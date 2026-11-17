#!/usr/bin/env python3
"""Frozen DDR expression scores versus same-cell functional repair reporters.

Replay: --input paired_cells.csv.gz --output DIRECTORY.
Rebuild: --data-root DIRECTORY --annotations figure6a_scrna_umap_plot.csv
         --output DIRECTORY.
Raw inputs are the two GEO10xcount H5s and two GEOH5ADs.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, platform
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.stats import rankdata, spearmanr

SEED = 20261117
DDR = "TP53 CDKN1A MDM2 GADD45A GADD45B GADD45G DDB2 XPC BBC3 BAX PMAIP1 FAS RRM2B SESN1 BTG2 FDXR PLK3 PPM1D ATM ATR CHEK1 CHEK2 BRCA1 BRCA2 RAD51 PARP1 XRCC5 XRCC6 H2AFX MRE11 NBN FANCD2 CLSPN TOPBP1".split()
P53 = "TP53 CDKN1A MDM2 DDB2 GADD45A GADD45B RRM2B BBC3 BAX PMAIP1 PPM1D FDXR SESN1 BTG2 FAS".split()
REPAIR = sorted(set(DDR) - set(P53))
CYCLE = "MKI67 TOP2A PCNA MCM2 MCM3 MCM4 MCM5 MCM6 MCM7 CDK1 CCNB1 CCNB2 CCNE1 CCNA2 CDC20 AURKA AURKB".split()
E2F = "E2F1 E2F2 E2F3 MCM2 MCM3 MCM4 MCM5 MCM6 MCM7 PCNA RRM1 RRM2 TYMS TK1 CDC6 CCNE1 CCNA2 CDK2".split()
SETS = {"DDR34": DDR, "p53_15": P53, "repair_19": REPAIR, "cycle": CYCLE, "E2F": E2F}
PATHWAYS = {
    "MMR": "GFP_BAR_003",
    "NER": "GFP_BAR_004",
    "DR_O6MeG": "GFP_BAR_005",
    "BER_8oxoG_C": "GFP_BAR_006",
    "BER_A_8oxoG": "GFP_BAR_007",
    "BER_UG": "GFP_BAR_008",
    "BER_Hx": "GFP_BAR_009",
    "NHEJ": "GFP_BAR_DBSR_2",
}
FRACTIONS = [x for x in PATHWAYS if x not in ["NER", "NHEJ"]]
COV = [
    "log_RNA_UMI",
    "log_RNA_genes",
    "mitochondrial_fraction",
    "log_reference_reporter",
    "cycle",
    "E2F",
]
KNOWN_KO = ["TK6 LIG4-/-", "HELA XPA-/-", "U2OS OGG-/-", "HAP1 MUTYH-/-"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def read_col(v):
    if isinstance(v, h5py.Group):
        cat = read_col(v["categories"])
        return np.array([cat[i] if i >= 0 else None for i in v["codes"][:]])
    return v.asstr()[:] if v.dtype.kind in "OSU" else v[:]


def read_frame(g):
    cols = [x for x in g.keys() if x != "_index"]
    return pd.DataFrame({c: read_col(g[c]) for c in cols}, index=read_col(g["_index"]))


def annotation_mapping(annotation_file, axes):
    # The authors concatenate24h then2h; the CSV retains two lexically ordered
    # barcode runs. Recover source before removing concat collision suffixes.
    a = pd.read_csv(annotation_file, index_col=0)
    keys = np.array([x[:16] for x in a.index])
    cuts = np.where(keys[1:] < keys[:-1])[0] + 1
    assert len(cuts) == 1, (
        cuts.tolist(),
        "Expected two source-specific ordered barcode runs",
    )
    pieces = []
    for t, lo, hi in [("24h", 0, int(cuts[0])), ("2h", int(cuts[0]), len(a))]:
        z = a.iloc[lo:hi].copy()
        z["source_figure_row"] = np.arange(lo, hi)
        z["figure_cell_id"] = z.index
        z["barcode"] = z.index.str.replace(r"-1-1$", "-1", regex=True)
        assert z["barcode"].is_unique and set(z["barcode"]).issubset(set(axes[t]))
        z["timepoint"] = t
        z["cell_line"] = z["label"]
        pieces.append(z.drop(columns=["label", "UMAP1", "UMP2"]))
    return pd.concat(pieces, ignore_index=True)


def build_input(args, out):
    records = {}
    axes = {}
    metadata = []
    coverage = []
    checks = []
    for t, gsm in [("2h", "GSM9718931"), ("24h", "GSM9718932")]:
        raw = args.data_root / f"{gsm}_P3_{t}_filtered_feature_bc_matrix.h5"
        with h5py.File(raw) as f:
            m = f["matrix"]
            names = m["features"]["name"].asstr()[:]
            bar = m["barcodes"].asstr()[:]
            x = sparse.csc_matrix(
                (m["data"][:], m["indices"][:], m["indptr"][:]), shape=m["shape"][:]
            ).T.tocsr()
        assert np.all(x.data >= 0) and np.all(x.data == np.floor(x.data))
        endog = ~np.char.startswith(names.astype(str), "GFP_BAR_")
        xe = x[:, endog]
        gn = names[endog]
        total = np.asarray(xe.sum(axis=1)).ravel()
        genes = np.asarray((xe > 0).sum(axis=1)).ravel()
        mt = np.asarray(
            xe[:, np.char.startswith(gn.astype(str), "MT-")].sum(axis=1)
        ).ravel() / np.maximum(total, 1)
        alltotal = np.asarray(x.sum(axis=1)).ravel()
        sc = pd.DataFrame(
            {
                "barcode": bar,
                "timepoint": t,
                "RNA_UMI": total,
                "RNA_genes": genes,
                "mitochondrial_fraction": mt,
                "full_UMI": alltotal,
            }
        )
        for label, gs in SETS.items():
            actual = [
                ("H2AX" if q == "H2AFX" and "H2AFX" not in set(gn) else q) for q in gs
            ]
            idx = [int(np.flatnonzero(gn == q)[0]) for q in actual if q in set(gn)]
            assert len(idx) == len(gs), (label, len(idx), len(gs))
            z = xe[:, idx].toarray().astype(float)
            sc[label] = np.log1p(z * 10000 / np.maximum(total[:, None], 1)).mean(axis=1)
            if label == "DDR34":
                sc["DDR34_full_denominator"] = np.log1p(
                    z * 10000 / np.maximum(alltotal[:, None], 1)
                ).mean(axis=1)
            coverage.extend(
                {
                    "timepoint": t,
                    "set": label,
                    "specified_gene": q,
                    "reference_symbol": v,
                    "present": v in set(gn),
                }
                for q, v in zip(gs, actual)
            )
        path = args.data_root / f"{gsm}_P3_{t}.h5ad"
        with h5py.File(path) as f:
            obs = read_frame(f["obs"])
            r = read_frame(f["obsm"]["repaired_counts"])
            u = read_frame(f["obsm"]["unrepaired_counts"])
        assert np.array_equal(bar, obs.index)
        sc = sc.set_index("barcode").join(obs)
        for p, rep in PATHWAYS.items():
            if p in FRACTIONS:
                den = r[rep] + u[rep]
                reconstructed = r[rep] / den
                error = float(np.nanmax(np.abs(reconstructed - obs[p])))
                assert error < 1e-10, (t, p, error)
                sc[p + "_n"] = den
                sc[p + "_repaired"] = r[rep]
                sc[p + "_unrepaired"] = u[rep]
            else:
                reconstructed = obs[rep] / obs["GFP_BAR_010"]
                error = float(np.nanmax(np.abs(reconstructed - obs[p])))
                assert error < 1e-5, (t, p, error)
            checks.append(
                {"timepoint": t, "endpoint": p, "max_abs_reconstruction_error": error}
            )
        sc = sc.reset_index()
        records[t] = sc
        axes[t] = bar
        metadata.append(
            {"filename": raw.name, "bytes": raw.stat().st_size, "sha256": sha(raw)}
        )
        h5ad = args.data_root / f"{gsm}_P3_{t}.h5ad"
        if h5ad.exists() and h5ad.stat().st_size in [622243676, 513556044]:
            metadata.append(
                {
                    "filename": h5ad.name,
                    "bytes": h5ad.stat().st_size,
                    "sha256": sha(h5ad),
                }
            )
    mapping = annotation_mapping(args.annotations, axes)
    mapping.to_csv(out / "cell_annotation_map.csv.gz", index=False)
    df = pd.concat(records.values(), ignore_index=True).merge(
        mapping, on=["barcode", "timepoint"], how="inner", validate="one_to_one"
    )
    assert len(df) == 18024
    df["sample_id"] = df["timepoint"] + ":" + df["barcode"]
    assert df.sample_id.is_unique
    df["known_repair_knockout"] = df.cell_line.isin(KNOWN_KO)
    df["transfected"] = df.GFP_BAR_010 >= 50
    df["log_RNA_UMI"] = np.log1p(df.RNA_UMI)
    df["log_RNA_genes"] = np.log1p(df.RNA_genes)
    df["log_reference_reporter"] = np.log1p(df.GFP_BAR_010)
    assert (
        (df.RNA_UMI > 5000).all()
        and (df.RNA_genes > 200).all()
        and (df.mitochondrial_fraction < 0.075).all()
    )
    pd.DataFrame(checks).to_csv(out / "endpoint_reconstruction.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "gene_coverage.csv", index=False)
    metadata.append(
        {
            "filename": args.annotations.name,
            "bytes": args.annotations.stat().st_size,
            "sha256": sha(args.annotations),
        }
    )
    (out / "input_checksums.json").write_text(json.dumps(metadata, indent=2) + "\n")
    df.to_csv(out / "paired_cells.csv.gz", index=False, float_format="%.17g")
    return pd.read_csv(
        out / "paired_cells.csv.gz", low_memory=False, float_precision="round_trip"
    )


def corr(x, y, cov=None):
    x = rankdata(x)
    y = rankdata(y)
    if cov is not None:
        z = np.column_stack(
            [np.ones(len(x))] + [rankdata(cov[:, i]) for i in range(cov.shape[1])]
        )
        x = x - z @ np.linalg.lstsq(z, x, rcond=None)[0]
        y = y - z @ np.linalg.lstsq(z, y, rcond=None)[0]
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def bh(p):
    p = np.array(p)
    idx = np.argsort(p)
    q = np.minimum.accumulate((p[idx] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty(len(p))
    out[idx] = np.minimum(q, 1)
    return out


def aggregate(vals, rng):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    k = len(vals)
    if k < 3:
        return dict(
            n_lines=k,
            rho=np.nan,
            median_rho=np.nan,
            ci_low=np.nan,
            ci_high=np.nan,
            p_signflip=np.nan,
        )
    z = np.arctanh(np.clip(vals, -0.999999, 0.999999))
    mu = z.mean()
    boot = z[rng.integers(0, k, size=(10000, k))].mean(axis=1)
    lo, hi = np.tanh(np.quantile(boot, [0.025, 0.975]))
    # Exact two-sided sign-flip sensitivity at the cell-line level.
    extreme = 0
    total = 2**k
    for start in range(0, total, 8192):
        ii = np.arange(start, min(start + 8192, total), dtype=np.uint64)
        sign = ((ii[:, None] >> np.arange(k, dtype=np.uint64)) & 1) * 2.0 - 1.0
        extreme += int(np.sum(np.abs(sign @ z / k) >= abs(mu) - 1e-12))
    return dict(
        n_lines=k,
        rho=float(np.tanh(mu)),
        median_rho=float(np.median(vals)),
        ci_low=float(lo),
        ci_high=float(hi),
        p_signflip=extreme / total,
    )


def analyse(df, out):
    df.groupby(["timepoint", "cell_line", "transfected"], dropna=False).size().rename(
        "n_cells"
    ).reset_index().to_csv(out / "cohort_counts.csv", index=False)
    rng = np.random.default_rng(SEED)
    results = []
    effects = []
    diagnostic = []
    specs = [
        ("primary", "DDR34", 10, False, False),
        ("precision20", "DDR34", 20, False, False),
        ("source_nonmissing", "DDR34", 1, False, False),
        ("without_known_KO", "DDR34", 10, True, False),
        ("lesion_depth_adjusted", "DDR34", 10, False, True),
        ("p53_subset", "p53_15", 10, False, False),
        ("repair_subset", "repair_19", 10, False, False),
        ("reporter_in_denominator", "DDR34_full_denominator", 10, False, False),
    ]
    for spec, score, min_n, dropko, lesion_cov in specs:
        for t in ["2h", "24h"]:
            base = df[(df.timepoint == t) & df.transfected].copy()
            if dropko:
                base = base[~base.known_repair_knockout]
            for endpoint in PATHWAYS:
                g = base[np.isfinite(base[endpoint])].copy()
                if endpoint in FRACTIONS:
                    g = g[g[endpoint + "_n"] >= min_n]
                needed = [score, endpoint] + COV
                g = g.dropna(subset=needed)
                values = {"unadjusted": [], "partial_rank": []}
                nused = 0
                for line, a in g.groupby("cell_line", sort=True):
                    if len(a) < 20 or a[endpoint].nunique() < 2:
                        continue
                    cov = a[COV].to_numpy()
                    if lesion_cov and endpoint in FRACTIONS:
                        cov = np.column_stack([cov, np.log1p(a[endpoint + "_n"])])
                    ru = corr(a[score], a[endpoint])
                    rp = corr(a[score], a[endpoint], cov)
                    effects.append(
                        dict(
                            analysis=spec,
                            timepoint=t,
                            endpoint=endpoint,
                            cell_line=line,
                            n_cells=len(a),
                            rho_unadjusted=ru,
                            rho_partial=rp,
                            median_endpoint=float(a[endpoint].median()),
                        )
                    )
                    values["unadjusted"].append(ru)
                    values["partial_rank"].append(rp)
                    nused += len(a)
                for adjustment, val in values.items():
                    results.append(
                        dict(
                            analysis=spec,
                            score=score,
                            timepoint=t,
                            endpoint=endpoint,
                            adjustment=adjustment,
                            n_cells=nused,
                            **aggregate(val, rng),
                        )
                    )
                if spec == "primary":
                    diagnostic.append(
                        dict(
                            timepoint=t,
                            endpoint=endpoint,
                            n_eligible=len(g),
                            pooled_rho=(
                                corr(g[score], g[endpoint]) if len(g) > 2 else np.nan
                            ),
                            median_endpoint=float(g[endpoint].median()),
                            min_denominator=(
                                float(g[endpoint + "_n"].min())
                                if endpoint in FRACTIONS and len(g)
                                else np.nan
                            ),
                        )
                    )
    result = pd.DataFrame(results)
    result["q_signflip"] = np.nan
    for key, z in result.groupby(["analysis", "adjustment"]):
        idx = z.index[np.isfinite(z.p_signflip)]
        result.loc[idx, "q_signflip"] = bh(result.loc[idx, "p_signflip"])
    result.to_csv(out / "repair_associations.csv", index=False, float_format="%.17g")
    pd.DataFrame(effects).to_csv(
        out / "within_line_correlations.csv", index=False, float_format="%.17g"
    )
    pd.DataFrame(diagnostic).to_csv(
        out / "pooled_diagnostics.csv", index=False, float_format="%.17g"
    )
    primary = result[
        (result.analysis == "primary") & (result.adjustment == "partial_rank")
    ]
    primary.to_csv(out / "primary_results.csv", index=False, float_format="%.17g")
    print(primary.to_string(index=False), flush=True)
    software = {
        "Python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "h5py": h5py.__version__,
    }
    (out / "software_versions.json").write_text(json.dumps(software, indent=2) + "\n")
    (out / "gene_sets.json").write_text(json.dumps(SETS, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--annotations", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.input:
        df = pd.read_csv(a.input, low_memory=False, float_precision="round_trip")
    else:
        if not a.data_root or not a.annotations:
            p.error("Rebuilding requires --data-root and --annotations")
        df = build_input(a, a.output)
    analyse(df, a.output)


if __name__ == "__main__":
    main()
