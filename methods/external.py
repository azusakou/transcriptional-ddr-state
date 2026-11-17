#!/usr/bin/env python3
"""Reproduce external-context, treatment-summary and same-cell imaging analyses.

Run: python reanalyse_external.py --data-root /path/to/original/data --out /path/to/output
The data root contains datasets/{scperturb,geo_extracted,lineage_zenodo_14921691}.
No cell-level hypothesis tests or biological-replication claims are made.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json

import h5py
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.io import mmread

MODULES = {
    "ddr_composite": "TP53 CDKN1A MDM2 GADD45A GADD45B GADD45G DDB2 XPC BBC3 BAX PMAIP1 FAS RRM2B SESN1 BTG2 FDXR PLK3 PPM1D ATM ATR CHEK1 CHEK2 BRCA1 BRCA2 RAD51 PARP1 XRCC5 XRCC6 H2AFX MRE11 NBN FANCD2 CLSPN TOPBP1".split(),
    "TP53_axis": "TP53 CDKN1A MDM2 DDB2 GADD45A GADD45B RRM2B BBC3 BAX PMAIP1 PPM1D FDXR SESN1 BTG2 FAS".split(),
    "E2F_replication": "E2F1 E2F2 E2F3 MCM2 MCM3 MCM4 MCM5 MCM6 MCM7 PCNA RRM1 RRM2 TYMS TK1 CDC6 CCNE1 CCNA2 CDK2".split(),
    "ATF4_integrated_stress": "ATF4 DDIT3 ASNS TRIB3 SLC7A5 SLC7A11 HERPUD1 CEBPB CEBPG PPP1R15A DNAJC3 HSPA5".split(),
    "XBP1_UPR": "XBP1 ERN1 ATF6 HSPA5 DNAJB9 HERPUD1 PDIA4 PDIA6 EDEM1 SEC61A1 SEL1L".split(),
}
IR_MAP = {
    "GSM4331191": ("WT", "UT", "GSM4331191"),
    "GSM4331192": ("WT", "IR_5Gy_4h", "GSM4331192"),
    "GSM4331193": ("WT", "UT", "GSM4331193"),
    "GSM4331194": ("WT", "IR_5Gy_4h", "GSM4331194"),
    "GSM4331195": ("ATM_KD", "UT", "GSM4331195"),
    "GSM4331196": ("ATM_KD", "UT", "GSM4331196"),
    "GSM4331197": ("ATM_KD", "IR_5Gy_4h", "GSM4331197"),
    "GSM4331198": ("ATM_KD", "IR_5Gy_4h", "GSM4331198"),
}


def decode(x):
    return x.decode() if isinstance(x, bytes) else x


def column(group, key):
    x = group[key]
    if isinstance(x, h5py.Group):
        cats = np.array([decode(a) for a in x["categories"][:]], dtype=object)
        codes = x["codes"][:]
        return np.array([cats[i] if i >= 0 else None for i in codes], dtype=object)
    return np.array([decode(a) for a in x[:]], dtype=object)


def reference(values):
    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    if scale == 0:
        center, scale = float(np.mean(values)), float(np.std(values))
    return center, scale


def score_matrix(matrix, genes):
    """Gene-by-cell raw counts: log1p library-size normalization then means."""
    totals = np.asarray(matrix.sum(axis=0)).ravel()
    matrix = sparse.csr_matrix(matrix).multiply(1e4 / np.maximum(totals, 1)).tocsr()
    matrix.data = np.log1p(matrix.data)
    lookup = {g: i for i, g in enumerate(genes)}
    return pd.DataFrame(
        {
            name: np.asarray(
                matrix[[lookup[g] for g in markers if g in lookup]].mean(axis=0)
            ).ravel()
            for name, markers in MODULES.items()
        }
    )


def score_h5(path):
    with h5py.File(path, "r") as f:
        o = pd.DataFrame({c: column(f["obs"], c) for c in f["obs"]})
        key = next(k for k in ["gene_symbol", "gene_name", "_index"] if k in f["var"])
        genes = list(column(f["var"], key))
        x = f["X"]
        indices = {g: i for i, g in enumerate(genes)}
        wanted = sorted(set(sum(MODULES.values(), [])) & set(genes))
        positions = [indices[g] for g in wanted]
        shape = tuple(x.attrs["shape"])
        if decode(x.attrs["encoding-type"]) == "csc_matrix":
            vals = np.zeros((shape[0], len(wanted)))
            for j, k in enumerate(positions):
                a, b = x["indptr"][k : k + 2]
                vals[x["indices"][a:b], j] = x["data"][a:b]
        else:
            mat = sparse.csr_matrix(
                (x["data"][:], x["indices"][:], x["indptr"][:]), shape=shape
            )
            vals = mat[:, positions].toarray()
        vals = np.log1p(
            vals
            * (1e4 / np.maximum(pd.to_numeric(o.ncounts).to_numpy(float), 1))[:, None]
        )
        for name, markers in MODULES.items():
            o[name] = vals[:, [wanted.index(g) for g in markers if g in wanted]].mean(
                axis=1
            )
        o["source_row"] = np.arange(len(o))
        return o


def external(data, out):
    all_cells = []
    for ds in ["GSE145700", "GSE288487", "GSE255874"]:
        folder = data / "datasets" / "geo_extracted" / ds
        for path in sorted(
            folder.glob(
                "*.codingGene.txt.gz" if ds == "GSE145700" else "*_matrix.mtx.gz"
            )
        ):
            gsm = path.name.split("_")[0]
            if ds == "GSE145700":
                raw = pd.read_csv(path, sep="\t")
                cells = score_matrix(
                    raw.drop(columns=["GeneName", "GeneSymbol"]).to_numpy(float),
                    raw.GeneSymbol.astype(str),
                )
                genotype, condition, experiment = IR_MAP[gsm]
            else:
                stem = path.name.replace("_matrix.mtx.gz", "")
                with gzip.open(path.with_name(stem + "_features.tsv.gz"), "rt") as f:
                    genes = [line.strip().split("\t")[1] for line in f]
                cells = score_matrix(mmread(path).tocsr(), genes)
                genotype = "U-2 OS"
                if ds == "GSE288487":
                    experiment = (
                        "Exp1" if gsm in ["GSM8768903", "GSM8768904"] else "Exp2"
                    )
                    condition = (
                        "UT" if gsm in ["GSM8768903", "GSM8768905"] else "IR_4Gy_48h"
                    )
                else:
                    experiment = "rep2" if "replicate" in stem else "rep1"
                    condition = stem.split("_", 1)[1].replace("_replicate", "")
            cells["dataset_id"], cells["sample_id"] = ds, gsm
            cells["genotype"], cells["condition"], cells["experiment"] = (
                genotype,
                condition,
                experiment,
            )
            cells["source_row"] = np.arange(len(cells))
            all_cells.append(cells)
    cells = pd.concat(all_cells, ignore_index=True)
    refs, samples = [], []
    for ds, g in cells.groupby("dataset_id"):
        control = g[g.condition.isin(["UT", "EV_2N4N"])]
        for module in MODULES:
            center, scale = reference(control[module].to_numpy())
            refs.append(
                dict(
                    dataset_id=ds,
                    module=module,
                    center=center,
                    scale=scale,
                    n_reference_cells=len(control),
                )
            )
            cells.loc[g.index, module + "_z"] = (g[module] - center) / (scale + 1e-8)
    for keys, g in cells.groupby(
        ["dataset_id", "sample_id", "genotype", "condition", "experiment"]
    ):
        for module in MODULES:
            samples.append(
                dict(
                    zip(
                        [
                            "dataset_id",
                            "sample_id",
                            "genotype",
                            "condition",
                            "experiment",
                        ],
                        keys,
                    ),
                    module=module,
                    n_cells=len(g),
                    mean_score=g[module].mean(),
                    mean_control_z=g[module + "_z"].mean(),
                    fraction_z2=(g[module + "_z"] > 2).mean(),
                    fraction_z3=(g[module + "_z"] > 3).mean(),
                )
            )
    samples = pd.DataFrame(samples)
    effects = []
    for keys, g in samples.groupby(["dataset_id", "genotype", "module"]):
        control = g[g.condition.isin(["UT", "EV_2N4N"])]
        treated = g[~g.condition.isin(["UT", "EV_2N4N"])]
        for _, t in treated.iterrows():
            if keys[0] == "GSE145700":
                # GEO provides no pairing identifier. Each IR library is compared
                # with the equal-library mean of both genotype-matched controls.
                b = control.iloc[0].copy()
                for c in ["mean_score", "mean_control_z", "fraction_z2", "fraction_z3"]:
                    b[c] = control[c].mean()
                b["sample_id"] = ";".join(control.sample_id)
                b["n_cells"] = control.n_cells.sum()
                design = (
                    "IR sample versus unweighted mean of two unpaired control libraries"
                )
            else:
                candidates = control[control.experiment == t.experiment]
                if len(candidates) != 1:
                    raise ValueError("Ambiguous external reference")
                b = candidates.iloc[0]
                design = "matched experiment labels"
            effects.append(
                dict(
                    zip(["dataset_id", "genotype", "module"], keys),
                    condition=t.condition,
                    experiment=t.experiment,
                    control_sample=b.sample_id,
                    treated_sample=t.sample_id,
                    n_control=int(b.n_cells),
                    n_treated=int(t.n_cells),
                    comparison_design=design,
                    delta_mean_score=t.mean_score - b.mean_score,
                    delta_mean_control_z=t.mean_control_z - b.mean_control_z,
                    delta_fraction_z2=t.fraction_z2 - b.fraction_z2,
                    delta_fraction_z3=t.fraction_z3 - b.fraction_z3,
                )
            )
    effects = pd.DataFrame(effects)
    summaries = (
        effects.groupby(["dataset_id", "genotype", "module", "condition"])
        .agg(
            n_sample_contrasts=("experiment", "size"),
            mean_delta_score=("delta_mean_score", "mean"),
            min_delta_score=("delta_mean_score", "min"),
            max_delta_score=("delta_mean_score", "max"),
            mean_delta_control_z=("delta_mean_control_z", "mean"),
            mean_delta_fraction_z2=("delta_fraction_z2", "mean"),
            min_delta_fraction_z2=("delta_fraction_z2", "min"),
            max_delta_fraction_z2=("delta_fraction_z2", "max"),
        )
        .reset_index()
    )
    cells.to_csv(out / "external_scored_cells.csv.gz", index=False)
    samples.to_csv(out / "external_sample_scores.csv", index=False)
    effects.to_csv(out / "external_sample_effects.csv", index=False)
    summaries.to_csv(out / "external_effect_summary.csv", index=False)
    pd.DataFrame(refs).to_csv(out / "external_control_references.csv", index=False)


def chemical(data, out):
    ds = "SrivatsanTrapnell2020_sciplex2"
    o = score_h5(data / "datasets" / "scperturb" / (ds + ".h5ad"))
    o = o[o.perturbation.isin(["Nutlin", "SAHA"])].copy()
    control_mask = pd.to_numeric(o.dose_value, errors="coerce") == 0
    refs, wells = [], []
    for module in MODULES:
        center, scale = reference(o.loc[control_mask, module].to_numpy())
        o[module + "_high"] = (o[module] - center) / (scale + 1e-8) > 2
        refs.append(
            dict(
                dataset_id=ds,
                module=module,
                center=center,
                scale=scale,
                n_reference_cells=int(control_mask.sum()),
            )
        )
    for (drug, dose, well), g in o.groupby(["perturbation", "dose_value", "well"]):
        for module in MODULES:
            wells.append(
                dict(
                    dataset_id=ds,
                    drug=drug,
                    dose_uM=float(dose),
                    hash_well=well,
                    module=module,
                    n_cells=len(g),
                    mean_score=g[module].mean(),
                    fraction_z2=g[module + "_high"].mean(),
                )
            )
    wells = pd.DataFrame(wells)
    wells.to_csv(out / "sciplex_hash_well_scores.csv", index=False)
    dose = []
    for (drug, module), g in wells.groupby(["drug", "module"]):
        b = g[g.dose_uM == 0]
        for d, t in g[g.dose_uM > 0].groupby("dose_uM"):
            dose.append(
                dict(
                    drug=drug,
                    module=module,
                    dose_uM=d,
                    n_control_cells=int(b.n_cells.sum()),
                    n_treated_cells=int(t.n_cells.sum()),
                    n_control_hash_wells=len(b),
                    n_treated_hash_wells=len(t),
                    control_fraction=b.fraction_z2.mean(),
                    treated_fraction=t.fraction_z2.mean(),
                    delta_fraction_z2=t.fraction_z2.mean() - b.fraction_z2.mean(),
                    treated_well_min=t.fraction_z2.min(),
                    treated_well_max=t.fraction_z2.max(),
                    uncertainty="descriptive: hash labels do not identify independent culture replicates",
                )
            )
    pd.DataFrame(dose).to_csv(out / "sciplex_dose_effects.csv", index=False)
    pd.DataFrame(refs).to_csv(out / "chemical_control_references.csv", index=False)
    (out / "analysis_definitions.json").write_text(
        json.dumps(
            {
                "normalization": "log1p(10000 * count / library_total)",
                "discovery_library_total": "obs.ncounts; verified equal to X row sums for both chemical datasets",
                "chemical_control_thresholds": "pooled dose-zero Nutlin and SAHA sci-Plex2; source-matched McFarland analyses are supplied separately",
                "chemical_fraction_threshold": "control median + 2 * 1.4826 * MAD; SD fallback for zero MAD",
                "chemical_inclusion": "sci-Plex2 Nutlin and SAHA; compound-specific zero-dose controls",
                "chemical_effect_summary": "McFarland contrasts are supplied by source_context/reanalyse_functional_context.py",
                "sciplex_effect_summary": "unweighted hash-well means at each dose versus same-compound dose-zero hash-well mean; no inferential interval",
                "external_reference": "one shared untreated/EV control distribution per GEO dataset; no within-sample standardization",
                "GSE145700_comparisons": "two unpaired IR libraries per genotype compared separately with equal-library control mean; no validated pairing identifiers",
                "imaging_reference": "one field of view; exact frame and coordinate linkage; endpoint at Fiji slice114",
                "imaging_inclusion": ">=5 live-object-matched frames; endpoint is not counted as a live frame",
                "imaging_inference": "descriptive correlations only; observation-count-adjusted rank correlation is Pearson correlation of rank residuals after linear adjustment for ranked number of live frames",
                "module_genes": MODULES,
            },
            indent=2,
        )
    )


def partial_rank(x, y, z):
    x, y, z = [stats.rankdata(np.asarray(a, float)) for a in [x, y, z]]
    design = np.column_stack([np.ones(len(z)), z])
    return float(
        np.corrcoef(
            x - design @ np.linalg.lstsq(design, x, rcond=None)[0],
            y - design @ np.linalg.lstsq(design, y, rcond=None)[0],
        )[0, 1]
    )


def imaging(data, out):
    folder = data / "datasets" / "lineage_zenodo_14921691"
    track = pd.read_csv(folder / "Tracking_fiji.xls", sep="\t", encoding="latin1")
    track.columns = [
        "row_id",
        "track_id",
        "Slice",
        "X",
        "Y",
        "Distance",
        "Velocity",
        "Pixel",
    ]
    live = pd.read_csv(folder / "analysis_parameters_live.txt", sep="\t")
    end = pd.read_csv(folder / "analysis_parameters_iterative_staining.txt", sep="\t")
    for df in [live, end]:
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]
        df["X"], df["Y"] = df.Center_X, df.Center_Y
    track["Time"] = track.Slice - 1
    merged = track.merge(
        live, on=["Time", "X", "Y"], how="left", validate="many_to_one"
    )
    rows = []
    for tid, g in merged.groupby("track_id"):
        v = g[g.Object_ID.notna()].sort_values("Time")
        if len(v) < 5:
            continue
        rows.append(
            dict(
                track_id=tid,
                n_live_frames=len(v),
                duration_frames=v.Time.max() - v.Time.min() + 1,
                start_frame=v.Time.min(),
                end_frame=v.Time.max(),
                max_PCNA=v.PCNA_foci__Counts.max(),
                mean_PCNA=v.PCNA_foci__Counts.mean(),
                last_PCNA=v.PCNA_foci__Counts.iloc[-1],
                max_53BP1=v["53BP1_foci__Counts"].max(),
                mean_53BP1=v["53BP1_foci__Counts"].mean(),
                last_53BP1=v["53BP1_foci__Counts"].iloc[-1],
            )
        )
    summary = pd.DataFrame(rows)
    merged.loc[
        merged.Object_ID.notna(),
        [
            "track_id",
            "Time",
            "X",
            "Y",
            "Object_ID",
            "Area",
            "PCNA_foci__Counts",
            "53BP1_foci__Counts",
        ],
    ].to_csv(out / "imaging_per_frame.csv.gz", index=False)
    last_slice = int(live.Time.max() + 2)
    endpoint = track[track.Slice == last_slice].merge(
        end, on=["X", "Y"], validate="one_to_one"
    )
    matched = summary.merge(endpoint, on="track_id", validate="one_to_one")
    associations = []
    for x, y, subset in [
        ("max_PCNA", "max_53BP1", summary),
        ("last_53BP1", "gH2AX_foci__Counts", matched),
        ("max_53BP1", "gH2AX_foci__Counts", matched),
        ("last_PCNA", "gH2AX_foci__Counts", matched),
        ("mean_53BP1", "gH2AX_foci__Counts", matched),
    ]:
        associations.append(
            dict(
                feature_x=x,
                feature_y=y,
                n_tracks=len(subset),
                rho=float(stats.spearmanr(subset[x], subset[y]).statistic),
                observation_count_adjusted_rank_correlation=partial_rank(
                    subset[x], subset[y], subset.n_live_frames
                ),
                n_fields=1,
                interpretation="descriptive; no independent-field uncertainty",
            )
        )
    summary.to_csv(out / "imaging_track_summaries.csv", index=False)
    matched.to_csv(out / "imaging_live_endpoint_links.csv", index=False)
    pd.DataFrame(associations).to_csv(out / "imaging_associations.csv", index=False)
    (out / "imaging_counts.json").write_text(
        json.dumps(
            dict(
                n_tracking_rows=len(track),
                n_exported_tracks=track.track_id.nunique(),
                n_live_objects=len(live),
                n_live_frames=live.Time.nunique(),
                n_fields=1,
                n_wells=1,
                n_endpoint_objects=len(end),
                n_exact_endpoint_links=len(endpoint),
                n_tracks_at_least_five_live_frames=len(summary),
                n_eligible_endpoint_links=len(matched),
                endpoint_fiji_slice=last_slice,
                lineage_parent_identifiers_available=False,
            ),
            indent=2,
        )
    )


def manifest(data, out):
    rows = []
    source_map = json.loads(Path(__file__).with_name("source_url_map.json").read_text())
    downloads = source_map["download_manifest"]
    zen_files = source_map["zenodo_14921691_files"]
    for path in sorted((data / "datasets").rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        rel = path.relative_to(data)
        if "scperturb" in rel.parts:
            if path.suffix != ".h5ad":
                continue
            source = "https://zenodo.org/records/7041849"
            url = downloads[path.stem]
        elif "geo_extracted" in rel.parts:
            if not path.name.endswith(
                (
                    "_matrix.mtx.gz",
                    "_features.tsv.gz",
                    "_barcodes.tsv.gz",
                    ".codingGene.txt.gz",
                )
            ):
                continue
            ds = path.parent.name
            source = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={ds}"
            # Archive URLs are recorded only when present in the original
            # download manifest; the accession and exact local file suffice
            # to identify GSE255874, whose download was not in that manifest.
            url = downloads.get(ds, "")
        elif "lineage_zenodo_14921691" in rel.parts:
            if path.name not in zen_files:
                continue
            source = "https://zenodo.org/records/14921691"
            url = zen_files[path.name]
        else:
            continue
        h = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(8 * 1024 * 1024):
                h.update(chunk)
        rows.append(
            dict(
                relative_path=str(rel),
                bytes=path.stat().st_size,
                sha256=h.hexdigest(),
                source_record=source,
                source_download=url,
            )
        )
    pd.DataFrame(rows).to_csv(out / "source_file_manifest.csv", index=False)


def inventory(data, out):
    rows = []
    for path in sorted((data / "datasets" / "scperturb").glob("*.h5ad")):
        with h5py.File(path, "r") as f:
            gene_key = next(
                k for k in ["gene_symbol", "gene_name", "_index"] if k in f["var"]
            )
            shape = (len(column(f["obs"], "ncounts")), len(column(f["var"], gene_key)))
        rows.append(
            dict(
                resource_id=path.stem,
                source_record="Zenodo 7041849",
                n_cells=int(shape[0]),
                n_features=int(shape[1]),
                analysis_unit="transcriptomes",
                role=(
                    "forecasting and classification"
                    if path.stem == "McFarlandTsherniak2020"
                    else (
                        "classification and dose summaries"
                        if path.stem == "SrivatsanTrapnell2020_sciplex2"
                        else (
                            "genetic perturbation context"
                            if path.stem == "ReplogleWeissman2022_rpe1"
                            else "compendium description only"
                        )
                    )
                ),
            )
        )
    ext = pd.read_csv(
        out / "external_scored_cells.csv.gz", usecols=["dataset_id", "sample_id"]
    )
    for ds, g in ext.groupby("dataset_id"):
        rows.append(
            dict(
                resource_id=ds,
                source_record=ds,
                n_cells=len(g),
                n_features=np.nan,
                analysis_unit="transcriptomes",
                role="external transcriptional context",
            )
        )
    image = json.loads((out / "imaging_counts.json").read_text())
    rows.append(
        dict(
            resource_id="Zenodo 14921691",
            source_record="Zenodo 14921691",
            n_cells=image["n_tracks_at_least_five_live_frames"],
            n_features=np.nan,
            analysis_unit="same-cell tracks from one field",
            role="orthogonal imaging context; not transcript-score validation",
        )
    )
    pd.DataFrame(rows).to_csv(out / "resource_inventory.csv", index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--skip-manifest", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    external(args.data_root, args.out)
    chemical(args.data_root, args.out)
    imaging(args.data_root, args.out)
    inventory(args.data_root, args.out)
    if not args.skip_manifest:
        manifest(args.data_root, args.out)


if __name__ == "__main__":
    main()
