#!/usr/bin/env python3
"""Source-normal, experiment-matched descriptive effects and viability context.

PRISM associations are orthogonal assay corroboration in overlapping cell lines,
not an independent validation cohort, a repair assay, or a new experiment.
"""
from pathlib import Path
import argparse, json
import h5py, numpy as np, pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
MODULES = {
    "ddr_composite": "TP53 CDKN1A MDM2 GADD45A GADD45B GADD45G DDB2 XPC BBC3 BAX PMAIP1 FAS RRM2B SESN1 BTG2 FDXR PLK3 PPM1D ATM ATR CHEK1 CHEK2 BRCA1 BRCA2 RAD51 PARP1 XRCC5 XRCC6 H2AFX MRE11 NBN FANCD2 CLSPN TOPBP1".split(),
    "TP53_axis": "TP53 CDKN1A MDM2 DDB2 GADD45A GADD45B RRM2B BBC3 BAX PMAIP1 PPM1D FDXR SESN1 BTG2 FAS".split(),
    "E2F_replication": "E2F1 E2F2 E2F3 MCM2 MCM3 MCM4 MCM5 MCM6 MCM7 PCNA RRM1 RRM2 TYMS TK1 CDC6 CCNE1 CCNA2 CDK2".split(),
    "ATF4_integrated_stress": "ATF4 DDIT3 ASNS TRIB3 SLC7A5 SLC7A11 HERPUD1 CEBPB CEBPG PPP1R15A DNAJC3 HSPA5".split(),
    "XBP1_UPR": "XBP1 ERN1 ATF6 HSPA5 DNAJB9 HERPUD1 PDIA4 PDIA6 EDEM1 SEC61A1 SEL1L".split(),
    "cell_cycle_replication": "MKI67 TOP2A PCNA MCM2 MCM3 MCM4 MCM5 MCM6 MCM7 CDK1 CCNB1 CCNB2 CCNE1 CCNA2 CDC20 AURKA AURKB".split(),
    "integrated_stress": "ATF3 ATF4 ATF6 DDIT3 HSPA1A HSPA1B HSP90AA1 HMOX1 JUN FOS EGR1 IER3 XBP1 ERN1 EIF2AK3 DNAJB1 HSPA5".split(),
}
BATCH = {
    "Gemcitabine": "expt10",
    "Prexasertib": "expt10",
    "Idasanutlin": "expt1",
    "Bortezomib": "expt1",
}
FEATURE_FILE = {
    "Gemcitabine": "Gemcitabine_expt10",
    "Prexasertib": "Prexasertib_expt10",
    "Idasanutlin": "Idasanutlin_24hr_expt1",
    "Bortezomib": "Bortezomib_24hr_expt1",
}
GENE_IDENTITIES = json.loads((HERE / "gene_identity_records.json").read_text())


def resolved_markers(markers, genes):
    available = set(genes)
    result = {}
    for g in markers:
        candidates = sorted(
            available & set(GENE_IDENTITIES["records"][g]["valid_source_symbols"])
        )
        source = (
            g if g in available else candidates[0] if len(candidates) == 1 else None
        )
        if source is not None:
            result[g] = source
    if len(set(result.values())) != len(result):
        raise ValueError("Duplicate resolved marker identity")
    return result


def col(g, k):
    x = g[k]
    if isinstance(x, h5py.Group):
        cats = np.array(
            [v.decode() if isinstance(v, bytes) else v for v in x["categories"][:]],
            object,
        )
        return np.array([cats[i] if i >= 0 else None for i in x["codes"][:]], object)
    return np.array([v.decode() if isinstance(v, bytes) else v for v in x[:]], object)


def source_scores(data):
    frame = pd.read_csv(HERE / "source_cell_mapping.csv.gz", dtype={"source_time": str})
    with h5py.File(data / "datasets/scperturb/McFarlandTsherniak2020.h5ad") as f:
        key = next(k for k in ["gene_symbol", "gene_name", "_index"] if k in f["var"])
        genes = list(col(f["var"], key))
        lookup = {g: i for i, g in enumerate(genes)}
        resolved = {
            name: resolved_markers(markers, genes) for name, markers in MODULES.items()
        }
        wanted = sorted({s for mapping in resolved.values() for s in mapping.values()})
        x = f["X"]
        counts = np.zeros((len(frame), len(wanted)))
        for j, g in enumerate(wanted):
            a, b = x["indptr"][lookup[g] : lookup[g] + 2]
            counts[x["indices"][a:b], j] = x["data"][a:b]
        total = pd.to_numeric(col(f["obs"], "ncounts")).astype(float)
        frame["ncounts"] = total
        frame["ngenes"] = pd.to_numeric(col(f["obs"], "ngenes")).astype(float)
        expression = np.log1p(counts * (10000 / np.maximum(total, 1))[:, None])
        for name, markers in MODULES.items():
            frame[name] = expression[
                :, [wanted.index(g) for g in resolved[name].values()]
            ].mean(axis=1)
    keep = (
        (frame.cell_quality == "normal")
        & frame.source_experiment.isin(["expt1", "expt10"])
        & frame.source_condition.isin(["DMSO", *BATCH])
    )
    return frame[keep].copy()


def ref(x):
    center = np.median(x)
    scale = 1.4826 * np.median(np.abs(x - center))
    if scale == 0:
        center, scale = np.mean(x), np.std(x)
    return float(center), float(scale)


def effects(cells, mode):
    if mode == "24h":
        cells = cells[cells.source_time == "24"].copy()
    else:
        cells = cells[cells.source_time.isin(["6", "24"])].copy()
    control = cells[cells.source_condition == "DMSO"]
    references = []
    for name in MODULES:
        center, scale = ref(control[name].to_numpy())
        cells[name + "_high"] = (cells[name] - center) / (scale + 1e-8) > 2
        references.append(
            dict(
                mode=mode,
                module=name,
                center=center,
                scale=scale,
                n_reference_cells=len(control),
            )
        )
    rows = []
    for drug, batch in BATCH.items():
        ts = ["24"] if mode == "24h" or batch == "expt10" else ["6", "24"]
        relevant = cells[
            (cells.source_experiment == batch) & cells.source_time.isin(ts)
        ]
        group_cols = (
            ["cell_line", "source_time"]
            if mode == "matched_regimens"
            else ["cell_line"]
        )
        for group, t in relevant[relevant.source_condition == drug].groupby(group_cols):
            line = group[0]
            b = relevant[
                (relevant.source_condition == "DMSO") & (relevant.cell_line == line)
            ]
            if mode == "matched_regimens":
                b = b[b.source_time == group[1]]
            if len(t) < 10 or len(b) < 20:
                continue
            for module in MODULES:
                rows.append(
                    dict(
                        mode=mode,
                        drug=drug,
                        cell_line=line,
                        DepMap_ID=t.DepMap_ID.iloc[0],
                        singlet_ID=t.singlet_ID.iloc[0],
                        source_experiment=batch,
                        source_time=(
                            group[1] if mode == "matched_regimens" else ";".join(ts)
                        ),
                        module=module,
                        n_control=len(b),
                        n_treated=len(t),
                        control_mean=b[module].mean(),
                        treated_mean=t[module].mean(),
                        delta_mean_score=t[module].mean() - b[module].mean(),
                        control_fraction_z2=b[module + "_high"].mean(),
                        treated_fraction_z2=t[module + "_high"].mean(),
                        delta_fraction_z2=t[module + "_high"].mean()
                        - b[module + "_high"].mean(),
                    )
                )
    return pd.DataFrame(rows), references


def correlation(x, y, seed=20261117, bootstrap=4000, permutations=9999):
    good = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[good]
    y = np.asarray(y)[good]
    n = len(x)
    if n < 4 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return dict(
            n=n, rho=np.nan, ci_low=np.nan, ci_high=np.nan, p_permutation=np.nan
        )
    xr = stats.rankdata(x)
    yr = stats.rankdata(y)
    xr -= xr.mean()
    yr -= yr.mean()
    observed = float(xr @ yr / np.linalg.norm(xr) / np.linalg.norm(yr))
    rng = np.random.default_rng(seed)
    ix = rng.integers(n, size=(bootstrap, n))
    xb = stats.rankdata(x[ix], axis=1)
    yb = stats.rankdata(y[ix], axis=1)
    xb -= xb.mean(axis=1, keepdims=True)
    yb -= yb.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (
            (xb * yb).sum(axis=1)
            / np.linalg.norm(xb, axis=1)
            / np.linalg.norm(yb, axis=1)
        )
    b = b[np.isfinite(b)]
    yp = rng.permuted(np.broadcast_to(yr, (permutations, n)), axis=1)
    null = yp @ xr / np.linalg.norm(xr) / np.linalg.norm(yr)
    return dict(
        n=n,
        rho=observed,
        ci_low=np.percentile(b, 2.5),
        ci_high=np.percentile(b, 97.5),
        p_permutation=(1 + sum(abs(null) >= abs(observed) - 1e-14))
        / (permutations + 1),
    )


def bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.where(np.isfinite(p))[0]
    order = ok[np.argsort(p[ok])]
    if len(order):
        q[order] = np.minimum(
            1,
            np.minimum.accumulate(
                (p[order] * len(order) / np.arange(1, len(order) + 1))[::-1]
            )[::-1],
        )
    return q


def annotate(pairs):
    rows = []
    for drug, g in pairs.groupby("drug"):
        source = pd.read_csv(HERE / (FEATURE_FILE[drug] + "_CL_features.csv"))
        source = source[source.in_pool].copy()
        if source.DEPMAP_ID.duplicated().any():
            raise ValueError("Duplicate source phenotype identifier")
        linked = g.merge(
            source,
            left_on="DepMap_ID",
            right_on="DEPMAP_ID",
            how="left",
            validate="many_to_one",
        )
        if linked.DEPMAP_ID.isna().any():
            raise ValueError("Source phenotype identity unmatched")
        rows.append(linked)
    return pd.concat(rows, ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path)
    p.add_argument("--scored-cells", type=Path)
    args = p.parse_args()
    if args.scored_cells:
        cells = pd.read_csv(args.scored_cells, dtype={"source_time": str})
        if cells.source_row.duplicated().any() or not set(MODULES).issubset(cells):
            raise ValueError("Invalid compact source score table")
    else:
        if args.data_root is None:
            p.error("Provide --data-root or --scored-cells")
        cells = source_scores(args.data_root)
    cells.to_csv(HERE / "normal_matched_module_cells.csv.gz", index=False)
    refs = []
    both = []
    for mode in ["24h", "source_time_composition", "matched_regimens"]:
        pair, r = effects(cells, mode)
        refs.extend(r)
        both.append(pair)
    reg = both[-1]
    equal = []
    for _, g in reg.groupby(["drug", "cell_line", "module"]):
        row = g.iloc[0].copy()
        row["mode"] = "equal_time_aggregate"
        row["source_time"] = ";".join(g.source_time)
        row["n_regimens"] = len(g)
        for c in ["n_control", "n_treated"]:
            row[c] = g[c].sum()
        for c in [
            "control_mean",
            "treated_mean",
            "delta_mean_score",
            "control_fraction_z2",
            "treated_fraction_z2",
            "delta_fraction_z2",
        ]:
            row[c] = g[c].mean()
        equal.append(row)
    both.append(pd.DataFrame(equal))
    pair = pd.concat(both, ignore_index=True)
    pair.to_csv(HERE / "normal_matched_chemical_cell_line_effects.csv", index=False)
    pd.DataFrame(refs).to_csv(
        HERE / "normal_matched_control_references.csv", index=False
    )
    summaries = []
    for (mode, drug, module), g in pair.groupby(["mode", "drug", "module"]):
        x = g.delta_fraction_z2.to_numpy()
        rng = np.random.default_rng(20261117)
        boot = rng.choice(x, (4000, len(x)), replace=True).mean(axis=1)
        summaries.append(
            dict(
                mode=mode,
                dataset_id="McFarlandTsherniak2020",
                drug=drug,
                module=module,
                n_cell_lines=len(g),
                mean_delta_fraction_z2=x.mean(),
                n_control_cells=int(g.n_control.sum()),
                n_treated_cells=int(g.n_treated.sum()),
                ci_low=np.percentile(boot, 2.5),
                ci_high=np.percentile(boot, 97.5),
                uncertainty_unit="cell-line blocks, not biological culture replicates",
            )
        )
    pd.DataFrame(summaries).to_csv(
        HERE / "normal_matched_chemical_cell_line_summary.csv", index=False
    )
    linked = annotate(pair)
    linked.to_csv(HERE / "normal_matched_pairs_with_viability.csv", index=False)
    results = []
    for (mode, drug), g in linked[
        (linked.module == "ddr_composite") & (linked["mode"] != "matched_regimens")
    ].groupby(["mode", "drug"]):
        for feature in [
            "treated_fraction_z2",
            "control_fraction_z2",
            "delta_fraction_z2",
            "delta_mean_score",
        ]:
            z = correlation(g[feature].to_numpy(float), g.PRISM_AUC.to_numpy(float))
            results.append(
                dict(
                    mode=mode,
                    drug=drug,
                    expression_feature=feature,
                    outcome="exact-drug PRISM_AUC",
                    primary=(
                        mode == "equal_time_aggregate"
                        and feature == "treated_fraction_z2"
                    ),
                    **z
                )
            )
    results = pd.DataFrame(results)
    results["q_primary_four_tests"] = np.nan
    primary = results.primary
    results.loc[primary, "q_primary_four_tests"] = bh(
        results.loc[primary, "p_permutation"]
    )
    results.to_csv(HERE / "observed_prism_associations.csv", index=False)
    ida = linked[(linked.drug == "Idasanutlin") & (linked.module == "TP53_axis")].copy()
    ida["TP53_group"] = np.where(
        (ida.TP53_DAM_MUT == 0) & (ida.TP53_DAM_HOT == 0),
        "no_damaging_or_hotspot_mutation",
        "damaging_or_hotspot_mutation",
    )
    ida.to_csv(HERE / "idasanutlin_TP53_context.csv", index=False)
    ida.groupby(["mode", "TP53_group"]).agg(
        n_lines=("cell_line", "size"),
        n_PRISM=("PRISM_AUC", "count"),
        mean_baseline_p53=("control_mean", "mean"),
        mean_treated_p53=("treated_mean", "mean"),
        mean_delta_p53=("delta_mean_score", "mean"),
        median_delta_p53=("delta_mean_score", "median"),
        mean_PRISM_AUC=("PRISM_AUC", "mean"),
        median_PRISM_AUC=("PRISM_AUC", "median"),
    ).reset_index().to_csv(HERE / "idasanutlin_TP53_group_summary.csv", index=False)
    print(results[results.primary].to_string(index=False))


if __name__ == "__main__":
    main()
