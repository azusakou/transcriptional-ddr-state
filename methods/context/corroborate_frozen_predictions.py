#!/usr/bin/env python3
"""Join frozen DDR forecasts to exact-drug PRISM outcomes; fit no models.

Six/24-hour forecasts are averaged equally within cell line and drug before a
single AUC is attached. All available eligible cell lines are retained.
"""
from pathlib import Path
import argparse, json
import pandas as pd, numpy as np
from scipy import stats
from reanalyse_functional_context import correlation, bh, FEATURE_FILE, HERE

MODELS = ["parsimonious_additive", "ddr_fraction_additive", "drug_only"]


def paired_rho_difference(x1, x0, y, seed=4417):
    good = np.isfinite(x1) & np.isfinite(x0) & np.isfinite(y)
    x1, x0, y = [np.asarray(a)[good] for a in [x1, x0, y]]
    n = len(y)
    rng = np.random.default_rng(seed)
    ix = rng.integers(n, size=(4000, n))

    def rowrho(x, y):
        x = stats.rankdata(x, axis=1)
        y = stats.rankdata(y, axis=1)
        x -= x.mean(1, keepdims=True)
        y -= y.mean(1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (
                (x * y).sum(1) / np.linalg.norm(x, axis=1) / np.linalg.norm(y, axis=1)
            )

    # Correlation with sensitivity (1-AUC): positive denotes more sensitivity.
    delta = rowrho(x1[ix], -y[ix]) - rowrho(x0[ix], -y[ix])
    delta = delta[np.isfinite(delta)]
    est = stats.spearmanr(x1, -y).statistic - stats.spearmanr(x0, -y).statistic
    return dict(
        n=n,
        delta_rho_with_sensitivity=est,
        ci_low=np.percentile(delta, 2.5),
        ci_high=np.percentile(delta, 97.5),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True)
    args = p.parse_args()
    pred = pd.read_csv(args.predictions, dtype={"source_time": str})
    pred = pred[
        (pred.analysis == "composite_z2")
        & (pred.cohort == "time_specific")
        & (pred.split == "frozen_outer")
        & pred.model.isin(MODELS)
    ]
    if pred.duplicated(["cell_line", "drug", "source_time", "model"]).any():
        raise ValueError("Duplicate regimen/model forecasts")
    agg = (
        pred.groupby(["cell_line", "drug", "model"])
        .agg(
            predicted=("predicted", "mean"),
            observed=("observed", "mean"),
            n_regimens=("source_time", "size"),
            source_time=("source_time", lambda x: ";".join(x)),
            baseline_n_cells=("baseline_n_cells", "sum"),
            treated_n_cells=("treated_n_cells", "sum"),
        )
        .reset_index()
    )
    ids = pd.read_csv(
        HERE / "source_cell_mapping.csv.gz",
        usecols=["cell_line", "DepMap_ID", "singlet_ID"],
    ).drop_duplicates()
    if ids.cell_line.duplicated().any():
        raise ValueError("Ambiguous cell line identity")
    agg = agg.merge(ids, on="cell_line", validate="many_to_one")
    joined = []
    for drug, g in agg.groupby("drug"):
        source = pd.read_csv(HERE / (FEATURE_FILE[drug] + "_CL_features.csv"))
        source = source[source.in_pool]
        g = g.merge(
            source,
            left_on="DepMap_ID",
            right_on="DEPMAP_ID",
            how="left",
            validate="many_to_one",
        )
        if g.DEPMAP_ID.isna().any():
            raise ValueError("Phenotype identifier mismatch")
        joined.append(g)
    pairs = pd.concat(joined, ignore_index=True)
    pairs.to_csv(HERE / "frozen_forecasts_with_PRISM.csv", index=False)
    rows = []
    for drug, g in pairs.groupby("drug"):
        for model, h in g.groupby("model"):
            stat = correlation(h.predicted.to_numpy(float), h.PRISM_AUC.to_numpy(float))
            rows.append(
                dict(
                    drug=drug,
                    model=model,
                    feature="held-out predicted DDR fraction",
                    primary=(model == "parsimonious_additive"),
                    **stat
                )
            )
        h = g[g.model == "parsimonious_additive"]
        rows.append(
            dict(
                drug=drug,
                model="observed",
                feature="held-out observed DDR fraction",
                primary=False,
                **correlation(h.observed.to_numpy(float), h.PRISM_AUC.to_numpy(float))
            )
        )
    results = pd.DataFrame(rows)
    results["q_primary_four_tests"] = np.nan
    results.loc[results.primary, "q_primary_four_tests"] = bh(
        results.loc[results.primary, "p_permutation"]
    )
    results["q_all_sixteen_tests"] = bh(results.p_permutation)
    results.to_csv(HERE / "frozen_forecast_PRISM_associations.csv", index=False)
    ida = pairs[
        (pairs.drug == "Idasanutlin")
        & (pairs.model == "parsimonious_additive")
        & pairs.PRISM_AUC.notna()
    ].copy()
    ida["genotype_group"] = np.where(
        (ida.TP53_DAM_MUT == 0) & (ida.TP53_DAM_HOT == 0),
        "no_damaging_or_hotspot_mutation",
        "damaging_or_hotspot_mutation",
    )
    strata = []
    for label, g in [("all", ida), *list(ida.groupby("genotype_group"))]:
        for feature in ["observed", "predicted"]:
            strata.append(
                dict(
                    group=label,
                    feature=feature,
                    n_lines=len(g),
                    rho_with_PRISM_AUC=stats.spearmanr(
                        g[feature], g.PRISM_AUC
                    ).statistic,
                )
            )
    design = np.column_stack(
        [
            np.ones(len(ida)),
            (ida.genotype_group == "damaging_or_hotspot_mutation").astype(int),
        ]
    )
    for feature in ["observed", "predicted"]:
        x = stats.rankdata(ida[feature])
        y = stats.rankdata(ida.PRISM_AUC)
        rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
        ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        strata.append(
            dict(
                group="rank residuals adjusting source binary TP53 genotype",
                feature=feature,
                n_lines=len(ida),
                rho_with_PRISM_AUC=np.corrcoef(rx, ry)[0, 1],
            )
        )
    pd.DataFrame(strata).to_csv(
        HERE / "idasanutlin_genotype_stratified_associations.csv", index=False
    )
    differences = []
    for drug, g in pairs.groupby("drug"):
        wide = g.pivot(index="cell_line", columns="model", values="predicted")
        auc = (
            g.drop_duplicates("cell_line")
            .set_index("cell_line")
            .PRISM_AUC.reindex(wide.index)
            .to_numpy(float)
        )
        for base in ["ddr_fraction_additive", "drug_only"]:
            differences.append(
                dict(
                    drug=drug,
                    model="parsimonious_additive",
                    reference_model=base,
                    **paired_rho_difference(
                        wide.parsimonious_additive.to_numpy(float),
                        wide[base].to_numpy(float),
                        auc,
                    )
                )
            )
    pd.DataFrame(differences).to_csv(
        HERE / "frozen_forecast_PRISM_paired_differences.csv", index=False
    )
    (HERE / "functional_analysis_definitions.json").write_text(
        json.dumps(
            {
                "model_predictions": (
                    str(args.predictions.relative_to(HERE.parents[1]))
                    if args.predictions.is_relative_to(HERE.parents[1])
                    else str(args.predictions)
                ),
                "filter": {
                    "analysis": "composite_z2",
                    "cohort": "time_specific",
                    "split": "frozen_outer",
                    "primary_model": "parsimonious_additive",
                },
                "aggregation": "equal mean of eligible 6h and 24h predictions within cell line and drug; attach one AUC afterward",
                "outcome": "original source exact-drug PRISM_AUC; no GDSC/nutlin proxy and no AUC averaging across assays",
                "forecast_test_family": "secondary four per-drug predicted-DDR vs PRISM AUC Spearman associations; two-sided 9999 within-drug line permutations, BH correction across four",
                "diagnostic_family": "all sixteen observed/predicted model-outcome tests; BH across sixteen also reported",
                "uncertainty": "4000 line-bootstrap resamples of paired frozen predictions and AUC; conditional on fitted models",
                "genotype": "original source TP53 damaging/hotspot binary annotations; not used to fit forecasts",
                "scope": "orthogonal viability assays in overlapping cancer cell lines from the same published resource; not independent cohort validation or DNA repair measurements",
                "source_PRISM_standard_protocol": "five-day viability readout; secondary eight-point dose response 10uM to610pM; original packaged AUC table lacks per-well exact dose/run metadata",
                "source_PRISM_protocol_url": "https://doi.org/10.1038/s43018-019-0018-6",
            },
            indent=2,
        )
    )
    print(results[results.primary].to_string(index=False))


if __name__ == "__main__":
    main()
