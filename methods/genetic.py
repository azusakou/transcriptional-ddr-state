#!/usr/bin/env python3
"""Prespecified genetic-perturbation context for a transcriptional DDR endpoint.
No individual cells are treated as independent experimental replicates.
"""
from pathlib import Path
import argparse, json
import h5py, numpy as np, pandas as pd
from scipy import stats

SETS = {
    "composite": "TP53 CDKN1A MDM2 GADD45A GADD45B GADD45G DDB2 XPC BBC3 BAX PMAIP1 FAS RRM2B SESN1 BTG2 FDXR PLK3 PPM1D ATM ATR CHEK1 CHEK2 BRCA1 BRCA2 RAD51 PARP1 XRCC5 XRCC6 H2AFX MRE11 NBN FANCD2 CLSPN TOPBP1".split(),
    "repair_checkpoint": "DDB2 XPC RRM2B ATM ATR CHEK1 CHEK2 BRCA1 BRCA2 RAD51 PARP1 XRCC5 XRCC6 H2AFX MRE11 NBN FANCD2 CLSPN TOPBP1".split(),
    "p53_response": "TP53 CDKN1A MDM2 DDB2 GADD45A GADD45B RRM2B BBC3 BAX PMAIP1 PPM1D FDXR SESN1 BTG2 FAS".split(),
    "cycle": "MKI67 TOP2A PCNA MCM2 MCM3 MCM4 MCM5 MCM6 MCM7 CDK1 CCNB1 CCNB2 CCNE1 CCNA2 CDC20 AURKA AURKB".split(),
    "stress": "ATF3 ATF4 ATF6 DDIT3 HSPA1A HSPA1B HSP90AA1 HMOX1 JUN FOS EGR1 IER3 XBP1 ERN1 EIF2AK3 DNAJB1 HSPA5".split(),
}
GROUPS = {
    "repair_checkpoint": "ATR ATRIP CHEK1 BRCA1 BRCA2 RAD51 RAD51C RAD51D FANCD2 FANCI RPA1 RPA2 RPA3 CLSPN TOPBP1 MRE11 MRE11A NBN".split(),
    "proteostasis": "XBP1 ERN1 ATF4 ATF6 HSPA5 EIF2AK3 HSF1 SEC61A1 PDIA4 PDIA6 EDEM1 SEL1L".split(),
    "mitosis": "CDK1 CCNB1 CDC20 AURKA AURKB CENPE KIF11 PLK1 TTK BUB1B".split(),
    "ribosome": "RPL5 RPL11 RPS6 RPL7 RPL3 RPS3 RPL4 RPS19 RPL23 RPS14".split(),
}


def col(g, k):
    v = g[k]
    if isinstance(v, h5py.Group):
        cats = v["categories"].asstr()[:]
        codes = v["codes"][:]
        return np.where(codes >= 0, cats[np.maximum(codes, 0)], "NA")
    return v.asstr()[:] if v.dtype.kind in "OS" else v[:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.input) as f:
        genes = col(f["var"], "gene_name").astype(str)
        lookup = {g: i for i, g in enumerate(genes)}
        obs = pd.DataFrame(
            {
                k: col(f["obs"], k)
                for k in [
                    "cell_barcode",
                    "perturbation",
                    "guide_id",
                    "batch",
                    "ncounts",
                    "ngenes",
                ]
            }
        )
        union = sorted(set(sum(SETS.values(), [])) & set(genes))
        ids = sorted(lookup[g] for g in union)
        ordered = [genes[i] for i in ids]
        counts = np.asarray(f["X"][:, ids], float)
        total = obs.ncounts.to_numpy(float)
        for start in range(0, len(obs), 4096):
            if not np.array_equal(
                np.asarray(f["X"][start : start + 4096, :], float).sum(1),
                total[start : start + 4096],
            ):
                raise ValueError("ncounts differs from rawX row sums")
        x = np.log1p(counts * 1e4 / total[:, None])
        coverage = {}
        score_ix = {}
        for name, markers in SETS.items():
            ix = [i for i, g in enumerate(ordered) if g in markers]
            score_ix[name] = ix
            obs[name] = x[:, ix].mean(1)
            coverage[name] = {"defined": markers, "present": [ordered[i] for i in ix]}
        out = []
        refs = []
        for batch, g in obs.groupby("batch"):
            c = g[g.perturbation == "control"]
            if len(c) < 20:
                continue
            ci = c.index.to_numpy()
            for target, h in g.groupby("perturbation"):
                if target == "control" or len(h) < 3:
                    continue
                hi = h.index.to_numpy()
                row = {
                    "target": target,
                    "batch": int(batch),
                    "n_cells": len(h),
                    "n_controls": len(c),
                    "n_guide_constructs": h.guide_id.nunique(),
                }
                for name, ix in score_ix.items():
                    # Exclude a directly targeted gene from BOTH perturbed and matched-control scores.
                    if target in SETS[name] and target in ordered:
                        subset = [j for j in ix if ordered[j] != target]
                        cs = x[ci][:, subset].mean(1)
                        hs = x[hi][:, subset].mean(1)
                    else:
                        cs = c[name].to_numpy()
                        hs = h[name].to_numpy()
                    row[name + "_mean_delta"] = float(hs.mean() - cs.mean())
                    if name in ["composite", "p53_response", "repair_checkpoint"]:
                        center = np.median(cs)
                        scale = 1.4826 * np.median(abs(cs - center))
                        if scale <= 0:
                            raise ValueError("degeneratecontrol")
                        cutoff = center + 2 * scale
                        row[name + "_fraction_delta"] = float(
                            (hs > cutoff).mean() - (cs > cutoff).mean()
                        )
                        refs.append(
                            {
                                "batch": int(batch),
                                "target": target,
                                "score": name,
                                "n_controls": len(c),
                                "cutoff": float(cutoff),
                                "excluded_direct_target": (
                                    target
                                    if target in SETS[name] and target in ordered
                                    else ""
                                ),
                            }
                        )
                out.append(row)
        effects = pd.DataFrame(out)
        effects.to_csv(a.output / "batch_target_effects.csv", index=False)
        agg = (
            effects.groupby("target")
            .agg(
                n_cells=("n_cells", "sum"),
                n_batches=("batch", "nunique"),
                **{c: (c, "mean") for c in effects.columns if c.endswith("_delta")}
            )
            .reset_index()
        )
        agg["predefined_group"] = agg.target.map(
            {g: k for k, ls in GROUPS.items() for g in ls}
        ).fillna("other_target")
        agg["eligible_summary"] = (agg.n_cells >= 10) & (agg.n_batches >= 2)
        agg.to_csv(a.output / "target_effects.csv", index=False)
        summary = []
        for group, g in agg[agg.eligible_summary].groupby("predefined_group"):
            summary.append(
                {
                    "group": group,
                    "n_targets": len(g),
                    "n_cells": int(g.n_cells.sum()),
                    **{
                        "median_" + c: g[c].median()
                        for c in agg.columns
                        if c.endswith("_delta")
                    },
                    "targets": ";".join(g.target),
                }
            )
        pd.DataFrame(summary).to_csv(a.output / "group_summary.csv", index=False)
        pd.DataFrame(refs).to_csv(
            a.output / "target_batch_control_references.csv", index=False
        )
        (a.output / "definitions.json").write_text(
            json.dumps(
                {
                    "scores": coverage,
                    "predefined_groups": GROUPS,
                    "batch_control_references_file": "target_batch_control_references.csv",
                    "all_row_totals_verified": True,
                    "summary_eligibility": "at least 10 target cells across at least 2 technical batches; target-batch estimates require at least 3 target cells and 20 nontargeting controls",
                    "direct_target_gene_excluded_from_both_conditions": True,
                    "unit": "gene-target summary; batches are technical GEM groups, not assumed independent cultures",
                    "interpretation": "genetic perturbation context, not physicaldamage or functionalrepairvalidation",
                },
                indent=2,
            )
            + "\n"
        )
        print(pd.DataFrame(summary).drop(columns=["targets"]).to_string(index=False))


if __name__ == "__main__":
    main()
