#!/usr/bin/env python3
"""Fixed-signature RNA versus same-cell gammaH2AX, GEO GSE250285.

Download the three source TSV.gz files listed in source_manifest.json into
source/, then run: python analysis.py
Requires numpy, pandas and scipy. Sorting plates are technical
units; the analysis does not infer independently replicated culture effects.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

SETS = {
    "DDR": "TP53 CDKN1A MDM2 GADD45A GADD45B GADD45G DDB2 XPC BBC3 BAX PMAIP1 FAS RRM2B SESN1 BTG2 FDXR PLK3 PPM1D ATM ATR CHEK1 CHEK2 BRCA1 BRCA2 RAD51 PARP1 XRCC5 XRCC6 H2AFX MRE11 NBN FANCD2 CLSPN TOPBP1".split(),
    "p53": "TP53 CDKN1A MDM2 DDB2 GADD45A GADD45B RRM2B BBC3 BAX PMAIP1 PPM1D FDXR SESN1 BTG2 FAS".split(),
    "repair": "DDB2 XPC RRM2B ATM ATR CHEK1 CHEK2 BRCA1 BRCA2 RAD51 PARP1 XRCC5 XRCC6 H2AFX MRE11 NBN FANCD2 CLSPN TOPBP1".split(),
    "cycle": "MKI67 TOP2A PCNA MCM2 MCM3 MCM4 MCM5 MCM6 MCM7 CDK1 CCNB1 CCNB2 CCNE1 CCNA2 CDC20 AURKA AURKB".split(),
}
GATES = ["negative", "low", "medium", "high"]


def partial_rank(x, y, covariates):
    """Correlation of ranked residuals; each covariate is ranked too."""
    a, b = rankdata(x), rankdata(y)
    design = np.column_stack([np.ones(len(a))] + [rankdata(v) for v in covariates])
    a = a - design @ np.linalg.lstsq(design, a, rcond=None)[0]
    b = b - design @ np.linalg.lstsq(design, b, rcond=None)[0]
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path(__file__).parent / "source")
    ap.add_argument(
        "--input",
        type=Path,
        help="Compact all_scored_cells.csv.gz; reproduces statistics without original matrices",
    )
    ap.add_argument("--output", type=Path, default=Path(__file__).parent)
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="Within-plate cell resampling, not biological-replicate uncertainty",
    )
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.input:
        meta = pd.read_csv(args.input, index_col=0)
        if not meta.index.is_unique:
            raise ValueError("Nonunique scored-cell identifiers")
        required = [
            "DDR",
            "p53",
            "repair",
            "cycle",
            "qc_pass",
            "endogenous_counts",
            "log_depth",
            "log_genes",
            "log_DAPI",
        ]
        if not set(required).issubset(meta):
            raise ValueError("Incomplete scored-cell input")
        check_file = args.input.parent / "checks_and_definitions.json"
        checks = (
            json.loads(check_file.read_text())
            if check_file.exists()
            else {"n_original_cells": len(meta)}
        )
        gene_file = args.input.parent / "gene_sets.json"
        coverage = (
            json.loads(gene_file.read_text())
            if gene_file.exists()
            else {"defined_sets": SETS, "input_mode": "Precomputed fixed scores"}
        )
    else:
        cfile = args.source / "GSE250285_counts_rpe_dox_gp.tsv.gz"
        mfile = args.source / "GSE250285_metadata_rpe_dox_gp.tsv.gz"
        counts = pd.read_csv(cfile, sep="\t", index_col=0)
        meta = pd.read_csv(mfile, sep="\t", index_col=0)
        if not counts.index.is_unique or not meta.index.is_unique:
            raise ValueError("Nonunique gene or cell identifier")
        if set(counts.columns) != set(meta.index):
            raise ValueError("Expression/phenotype cell identifiers differ")
        counts = counts.loc[:, meta.index]
        arr = counts.to_numpy(float)
        if not np.isfinite(arr).all() or np.any(arr < 0):
            raise ValueError("Invalid source count values")
        totals = arr.sum(axis=0)
        total_error = float(np.max(np.abs(totals - meta.nCount_RNA.to_numpy())))
        if not np.allclose(totals, meta.nCount_RNA, atol=1e-3, rtol=1e-8):
            raise ValueError("Source library sizes differ from matrix column sums")
        endogenous = ~counts.index.str.startswith("ERCC")
        total_endog = arr[endogenous].sum(axis=0)
        meta["endogenous_counts"] = total_endog
        meta["log_depth"] = np.log1p(total_endog)
        meta["log_genes"] = np.log1p(meta.nFeature_RNA)
        meta["log_DAPI"] = np.log1p(meta.DAPI_fluorescence)
        coverage = {}
        for name, markers in SETS.items():
            genes = [g for g in markers if g in counts.index]
            coverage[name] = {
                "defined": markers,
                "present": genes,
                "missing": [g for g in markers if g not in counts.index],
            }
            meta[name] = np.log1p(
                1e4 * counts.loc[genes].to_numpy() / total_endog
            ).mean(axis=0)
        meta["qc_pass"] = (
            (meta.nFeature_RNA > 500)
            & (meta.nFeature_RNA < 7500)
            & (meta.nCount_RNA > 1000)
            & (meta.nCount_RNA < 50000)
            & (meta.percent_MT < 10)
            & (meta.percent_ERCC < 10)
        )
        # Source metadata gives sorting plates, not independent-culture identities.
        checks = {
            "n_original_cells": len(meta),
            "n_genes": len(counts),
            "exact_cell_identifier_join": True,
            "n_noninteger_count_values": int(np.sum(np.abs(arr - np.rint(arr)) > 1e-7)),
            "n_nonzero_count_values": int(np.sum(arr > 0)),
            "max_abs_total_difference_vs_source": total_error,
            "n_ERCC_features_excluded_from_normalization": int((~endogenous).sum()),
            "n_qc_cells": int(meta.qc_pass.sum()),
            "n_treated_qc_cells": int((meta.qc_pass & (meta.treatment == "GP")).sum()),
            "n_vehicle_qc_cells": int((meta.qc_pass & (meta.treatment == "veh")).sum()),
            "source_file_sha256": {
                f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                for f in [cfile, mfile]
            },
            "count_units": "Source-provided processed transcript counts; noninteger values retained without rounding",
            "normalization": "log1p(10000 * source transcript count / complete endogenous library total)",
            "matching_unit": "A single index-sorted cell linked by exact column/metadata identifier",
            "replication": "Eight sorting plates; independent culture replication not established",
            "source_FUCCI_per_cell_available": False,
        }
    meta.groupby(["experimentid", "hras", "treatment"], observed=True).agg(
        n_original=("qc_pass", "size"), n_qc=("qc_pass", "sum")
    ).reset_index().to_csv(args.output / "plate_counts.csv", index=False)
    meta.index.name = "cell_id"
    meta.to_csv(args.output / "all_scored_cells.csv.gz")
    data = meta.loc[meta.qc_pass].copy()
    refs = []
    controls = data.loc[data.treatment == "veh"]
    for score in ["DDR", "p53", "repair"]:
        center = float(controls[score].median())
        scale = float(1.4826 * np.median(np.abs(controls[score] - center)))
        if scale <= 0:
            raise ValueError("Zero vehicle MAD")
        cutoff = center + 2 * scale
        data[score + "_high"] = data[score] > cutoff
        refs.append(
            {
                "score": score,
                "center": center,
                "scale": scale,
                "cutoff": cutoff,
                "n_controls": len(controls),
                "reference": "Fixed pooled QC-passing vehicle cells; descriptive gate contrasts only",
            }
        )
    for hras, c in controls.groupby("hras", observed=True):
        center = float(c.DDR.median())
        scale = float(1.4826 * np.median(np.abs(c.DDR - center)))
        mask = data.hras.eq(hras)
        data.loc[mask, "DDR_high_HRAS_reference"] = (
            data.loc[mask, "DDR"] > center + 2 * scale
        )
        refs.append(
            {
                "score": "DDR",
                "center": center,
                "scale": scale,
                "cutoff": center + 2 * scale,
                "n_controls": len(c),
                "reference": "HRAS-matched vehicle sensitivity: " + hras,
            }
        )
    pd.DataFrame(refs).to_csv(args.output / "vehicle_score_references.csv", index=False)
    treatment = data.loc[data.treatment == "GP"].copy()
    if (
        treatment[["yH2AX_fluorescence", "DAPI_fluorescence", "log_depth", "log_genes"]]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("Missing primary phenotype or covariate")
    effects, gate_effects, labels, interval_rows = [], [], [], []
    rng = np.random.default_rng(20261117)
    for plate, p in treatment.groupby("experimentid", observed=True):
        cov = [p.DAPI_fluorescence, p.log_depth, p.log_genes]
        for score in ["DDR", "p53", "repair"]:
            effects.append(
                {
                    "plate": plate,
                    "hras": p.hras.iloc[0],
                    "n": len(p),
                    "score": score,
                    "rho": float(spearmanr(p[score], p.yH2AX_fluorescence).statistic),
                    "rho_adjusted": partial_rank(p[score], p.yH2AX_fluorescence, cov),
                    "rho_adjusted_cycle": partial_rank(
                        p[score], p.yH2AX_fluorescence, cov + [p.cycle]
                    ),
                    "adjustment": "Ranked DAPI, log endogenous depth, log detected-gene count",
                    "additional_cycle": "Original fixed 17-gene transcriptional cycle proxy",
                }
            )
        for label in ["DDR_high", "DDR_high_HRAS_reference"]:
            high = p[label].astype(bool)
            labels.append(
                {
                    "plate": plate,
                    "hras": p.hras.iloc[0],
                    "label": label,
                    "n": len(p),
                    "n_DDR_high": int(high.sum()),
                    "rho_gamma": float(spearmanr(high, p.yH2AX_fluorescence).statistic),
                    "rho_gamma_adjusted": partial_rank(high, p.yH2AX_fluorescence, cov),
                    "gammaH2AX_high_median": float(
                        p.loc[high, "yH2AX_fluorescence"].median()
                    ),
                    "gammaH2AX_other_median": float(
                        p.loc[~high, "yH2AX_fluorescence"].median()
                    ),
                    "gammaH2AX_median_difference": float(
                        p.loc[high, "yH2AX_fluorescence"].median()
                        - p.loc[~high, "yH2AX_fluorescence"].median()
                    ),
                }
            )
        boot = []
        for _ in range(args.bootstrap):
            b = p.iloc[rng.integers(len(p), size=len(p))]
            bcov = [b.DAPI_fluorescence, b.log_depth, b.log_genes]
            boot.append(
                [
                    float(spearmanr(b.DDR, b.yH2AX_fluorescence).statistic),
                    partial_rank(b.DDR, b.yH2AX_fluorescence, bcov),
                    partial_rank(b.DDR, b.yH2AX_fluorescence, bcov + [b.cycle]),
                    partial_rank(b.DDR_high.astype(int), b.yH2AX_fluorescence, bcov),
                ]
            )
        for j, quantity in enumerate(
            ["rho", "rho_adjusted", "rho_adjusted_cycle", "DDR_high_rho_adjusted"]
        ):
            lo, hi = np.nanquantile(np.asarray(boot)[:, j], [0.025, 0.975])
            interval_rows.append(
                {
                    "plate": plate,
                    "hras": p.hras.iloc[0],
                    "n": len(p),
                    "quantity": quantity,
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                    "bootstrap_draws": args.bootstrap,
                    "scope": "Conditional within-plate cell-resampling interval; not biological-replicate uncertainty",
                }
            )
        for gate, g in p.groupby("yh2ax_level", observed=True):
            gate_effects.append(
                {
                    "plate": plate,
                    "hras": p.hras.iloc[0],
                    "gate": gate,
                    "n": len(g),
                    "gammaH2AX_median": float(g.yH2AX_fluorescence.median()),
                    "DAPI_median": float(g.DAPI_fluorescence.median()),
                    "DDR_mean": float(g.DDR.mean()),
                    "DDR_high_n": int(g.DDR_high.sum()),
                    "DDR_high_fraction": float(g.DDR_high.mean()),
                    "category_interpretation": "Source labels retained unchanged; inverse ordering versus numerical gammaH2AX; not used as ordinal outcome",
                }
            )
    pd.DataFrame(effects).to_csv(
        args.output / "within_plate_associations.csv", index=False
    )
    gate_table = pd.DataFrame(gate_effects)
    gate_table.to_csv(args.output / "source_gate_label_audit.csv", index=False)
    pd.DataFrame(labels).to_csv(
        args.output / "within_plate_DDR_label_associations.csv", index=False
    )
    pd.DataFrame(interval_rows).to_csv(
        args.output / "within_plate_conditional_intervals.csv", index=False
    )
    checks["categorical_gate_discrepancy"] = (
        "Source yh2ax_level has reversed ordering versus named yH2AX_fluorescence in all six treated plates. Values are retained; ordinal-gate discrimination/enrichment is not interpreted or silently relabelled."
    )
    checks["original_FCS_available"] = (
        "No FCS/index-sort raw files in the public GEO supplementary listing; quantitative phenotype fields are supplied in linked cell metadata."
    )
    checks["interval_scope"] = (
        "2000 within-plate cell-bootstrap draws, conditional on source assay and sorting; not independent-culture uncertainty."
    )
    # Equal-weight summary retains the distinction between technical and culture units.
    pde = pd.DataFrame(effects)
    primary = pde.loc[
        pde.score.eq("DDR"),
        ["plate", "hras", "n", "rho", "rho_adjusted", "rho_adjusted_cycle"],
    ].copy()
    primary["HRAS_condition"] = primary.hras.map(
        {"control": "HRAS uninduced", "dox": "HRAS G12V induced"}
    )
    label_frame = pd.DataFrame(labels).loc[
        lambda d: d.label.eq("DDR_high"),
        ["plate", "n_DDR_high", "rho_gamma_adjusted", "gammaH2AX_median_difference"],
    ]
    primary = primary.merge(label_frame, on="plate", validate="one_to_one")
    primary.to_csv(args.output / "primary_plate_summary.csv", index=False)
    summary = []
    for (condition, score), g in pde.groupby(["hras", "score"], observed=True):
        for endpoint in ["rho", "rho_adjusted", "rho_adjusted_cycle"]:
            summary.append(
                {
                    "hras": condition,
                    "score": score,
                    "quantity": endpoint,
                    "n_plates": len(g),
                    "n_cells": int(g.n.sum()),
                    "median": float(g[endpoint].median()),
                    "minimum": float(g[endpoint].min()),
                    "maximum": float(g[endpoint].max()),
                    "interpretation": "Descriptive across sorting plates; not biological-replicate uncertainty",
                }
            )
    pd.DataFrame(summary).to_csv(
        args.output / "condition_association_summary.csv", index=False
    )
    data.index.name = "cell_id"
    data.to_csv(args.output / "scored_cells.csv.gz")
    meta[["qc_pass"]].to_csv(args.output / "all_cell_QC.csv.gz", index_label="cell_id")
    (args.output / "checks_and_definitions.json").write_text(
        json.dumps(checks, indent=2) + "\n"
    )
    (args.output / "gene_sets.json").write_text(json.dumps(coverage, indent=2) + "\n")
    print(
        pde.loc[
            pde.score.eq("DDR"),
            ["plate", "hras", "n", "rho", "rho_adjusted", "rho_adjusted_cycle"],
        ].to_string(index=False)
    )
    print(
        pd.DataFrame(interval_rows)
        .loc[lambda d: d.quantity.eq("rho_adjusted")]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
