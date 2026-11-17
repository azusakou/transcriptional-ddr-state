#!/usr/bin/env python3
"""Source-QC- and experiment/time-matched forecasting with nested cell-line validation.
Companion nested_forecasting.py contains the fixed fractional-logit fitting engine.
"""
from __future__ import annotations
import argparse, json, hashlib, tempfile
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
import nested_forecasting as core

MAIN_MODELS = {
    **core.MODELS,
    "baseline_only": ([0, 1, 2, 3], False),
    "cycle_only_plus_drug": ([2], False),
    "no_DDR_TP53_plus_drug": ([2, 3], False),
    "weighted_parsimonious_additive": ([0, 1, 2, 3], False),
}
SENS_MODELS = {
    k: MAIN_MODELS[k]
    for k in ["drug_only", "ddr_fraction_additive", "parsimonious_additive"]
}
TARGETS = [
    ("composite_z2", "ddr_composite", 2, False),
    ("composite_z3", "ddr_composite", 3, False),
    ("custom_p53_z2", "TP53_response", 2, False),
    ("custom_repair_z2", "repair_checkpoint", 2, False),
    ("custom_repair_z3", "repair_checkpoint", 3, False),
    ("hallmark_DNA_repair_z2", "HALLMARK_DNA_REPAIR", 2, False),
    ("hallmark_DNA_repair_z3", "HALLMARK_DNA_REPAIR", 3, False),
    ("hallmark_p53_z2", "HALLMARK_P53_PATHWAY", 2, False),
    ("hallmark_p53_z3", "HALLMARK_P53_PATHWAY", 3, False),
    ("cycle_residual_z2", "ddr_composite", 2, True),
    ("source_stratified_z2", "ddr_composite", 2, False),
]


def extract_all(path, definitions, canonical, cache):
    modules = {k: v["defined"] for k, v in json.loads(definitions.read_text()).items()}
    modules.update(
        {
            k: v["geneSymbols"]
            for k, v in json.loads(canonical.read_text())["sets"].items()
        }
    )
    token = hashlib.sha256(
        (
            str(path.resolve()) + str(path.stat().st_mtime_ns) + json.dumps(modules)
        ).encode()
    ).hexdigest()[:20]
    cache.mkdir(parents=True, exist_ok=True)
    cf = cache / f"source_all_scores_{token}.npz"
    with h5py.File(path, "r") as f:
        obs = pd.DataFrame(
            {
                k: core.read_col(f["obs"], k)
                for k in [
                    "cell_line",
                    "perturbation",
                    "time",
                    "ncounts",
                    "cell_quality",
                ]
            }
        )
        obs.ncounts = pd.to_numeric(obs.ncounts)
        if cf.exists():
            return obs, np.load(cf)["scores"], modules
        genes = core.read_col(f["var"], "_index").astype(str)
        lookup = {g: j for j, g in enumerate(genes)}
        present = sorted(set(sum(modules.values(), [])) & set(genes))
        x = f["X"]
        arr = np.zeros((len(obs), len(present)), float)
        for j, g in enumerate(present):
            gi = lookup[g]
            start, end = map(int, x["indptr"][gi : gi + 2])
            arr[x["indices"][start:end], j] = x["data"][start:end]
        arr = np.log1p(arr / obs.ncounts.to_numpy(float)[:, None] * 1e4)
        scores = np.column_stack(
            [
                arr[:, [present.index(g) for g in gl if g in present]].mean(1)
                for gl in modules.values()
            ]
        )
    np.savez_compressed(cf, scores=scores)
    return obs, scores, modules


def cohort(obs, mapping, time_specific, earlier_controls=False):
    source = mapping.sort_values("source_row").reset_index(drop=True)
    assert np.array_equal(source.source_row.to_numpy(), np.arange(len(obs)))
    assert (source.cell_quality.to_numpy() == obs.cell_quality.to_numpy()).all()
    normal = obs.cell_quality.eq("normal").to_numpy()
    lines = obs.cell_line.to_numpy()
    per = obs.perturbation.to_numpy()
    rows = []
    controls = {}
    treated = {}
    for line in sorted(set(lines)):
        for drug in core.DRUGS:
            if earlier_controls and drug not in ["Idasanutlin", "Bortezomib"]:
                continue
            allti = np.flatnonzero(normal & (lines == line) & (per == drug))
            if earlier_controls:
                allti = np.array(
                    [i for i in allti if str(source.iloc[i].source_time) == "24"], int
                )
            if not len(allti):
                continue
            experiments = source.iloc[allti].source_experiment.unique()
            assert len(experiments) == 1
            experiment = experiments[0]
            times = sorted(source.iloc[allti].source_time.astype(str).unique())
            for timelist in [[t] for t in times] if time_specific else [times]:
                ti = np.array(
                    [i for i in allti if str(source.iloc[i].source_time) in timelist],
                    int,
                )
                cm = (
                    normal
                    & (lines == line)
                    & (per == "control")
                    & source.source_condition.eq("DMSO").to_numpy()
                    & source.source_experiment.eq(experiment).to_numpy()
                    & source.source_time.astype(str)
                    .isin(["6"] if earlier_controls else timelist)
                    .to_numpy()
                )
                ci = np.flatnonzero(cm)
                if len(ti) < 10 or len(ci) < 20:
                    continue
                regimen = (
                    drug + "_" + ",".join(timelist) + "h" if time_specific else drug
                )
                key = (line, regimen)
                controls[key] = ci
                treated[key] = ti
                rows.append(
                    {
                        "cell_line": line,
                        "drug": regimen,
                        "compound": drug,
                        "source_experiment": experiment,
                        "source_time": ",".join(timelist),
                        "baseline_time": (
                            "6" if earlier_controls else ",".join(timelist)
                        ),
                        "baseline_n_cells": len(ci),
                        "treated_n_cells": len(ti),
                    }
                )
    return pd.DataFrame(rows), controls, treated


def transformer_for(modules, target_name, cutoff, residualize, source_metadata=None):
    names = list(modules)
    jddr = names.index("ddr_composite")
    jp53 = names.index("TP53_response")
    jcy = names.index("cycle")
    js = names.index("stress")
    jt = names.index(target_name)

    # Older module labels use 'cycle'/'stress' in current scorer. Resolve exact established names.
    def transform(scores, pairs, controls, treated, trainlines):
        trainlines = set(trainlines)
        ci = np.unique(
            np.concatenate(
                [ix for (line, drug), ix in controls.items() if line in trainlines]
            )
        )
        arr = scores.copy()
        reg = None
        if residualize:
            reg = LinearRegression().fit(arr[ci, jcy : jcy + 1], arr[ci, jddr])
            arr[:, jddr] -= reg.predict(arr[:, jcy : jcy + 1])
        center = np.median(arr[ci], 0)
        scale = 1.4826 * np.median(abs(arr[ci] - center), 0)
        mask = scale <= 0
        center[mask] = arr[ci].mean(0)[mask]
        scale[mask] = arr[ci].std(0)[mask]
        scale = np.maximum(scale, 1e-8)
        z = (arr - center) / scale
        context_refs = {}
        if source_metadata is not None:
            sm = source_metadata.sort_values("source_row")
            exp = sm.source_experiment.to_numpy()
            times = sm.source_time.astype(str).to_numpy()
            for experiment, time in (
                pairs[["source_experiment", "source_time"]]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            ):
                mask = (exp == experiment) & (times == str(time))
                local = ci[mask[ci]]
                assert len(local) > 1
                cc = np.median(arr[local], 0)
                ss = 1.4826 * np.median(abs(arr[local] - cc), 0)
                bad = ss <= 0
                cc[bad] = arr[local].mean(0)[bad]
                ss[bad] = arr[local].std(0)[bad]
                ss = np.maximum(ss, 1e-8)
                z[mask] = (arr[mask] - cc) / ss
                context_refs[experiment + "_" + str(time) + "h"] = {
                    "n_training_controls": len(local),
                    "score_centers": dict(zip(names, cc.tolist())),
                    "score_scales": dict(zip(names, ss.tolist())),
                    "target_raw_cutoff": float(cc[jt] + cutoff * ss[jt]),
                }
        B = np.vstack(
            [
                np.r_[
                    (z[controls[(line, d)], jddr] > 2).mean(),
                    z[controls[(line, d)], jp53].mean(),
                    z[controls[(line, d)], jcy].mean(),
                    z[controls[(line, d)], js].mean(),
                ]
                for line, d in zip(pairs.cell_line, pairs.drug)
            ]
        )
        y = np.array(
            [
                (z[treated[(line, d)], jt] > cutoff).mean()
                for line, d in zip(pairs.cell_line, pairs.drug)
            ]
        )
        ref = {
            "n_training_control_cells": len(ci),
            "score_centers": dict(zip(names, center.tolist())),
            "score_scales": dict(zip(names, scale.tolist())),
            "target_name": target_name,
            "target_cutoff_z": cutoff,
            "target_raw_cutoff": float(center[jt] + cutoff * scale[jt]),
            "cycle_regression_intercept": (
                None if reg is None else float(reg.intercept_)
            ),
            "cycle_regression_slope": None if reg is None else float(reg.coef_[0]),
        }
        ref["target_reference_mode"] = (
            "source_experiment_and_time"
            if source_metadata is not None
            else "global_training_controls"
        )
        ref["context_references"] = context_refs
        return B, y, ref

    return transform


def performance(pred):
    rows = []
    for (analysis, cohort, split, model), g in pred.groupby(
        ["analysis", "cohort", "split", "model"]
    ):
        for drug in ["all"] + core.DRUGS:
            q = g if drug == "all" else g[g.drug == drug]
            if not len(q):
                continue
            met = core.metrics(q.observed.to_numpy(), q.predicted.to_numpy())
            slope, intercept = np.polyfit(q.predicted, q.observed, 1)
            rows.append(
                {
                    "analysis": analysis,
                    "cohort": cohort,
                    "split": split,
                    "model": model,
                    "drug": drug,
                    "n_lines": q.cell_line.nunique(),
                    "calibration_intercept": float(intercept),
                    "calibration_slope": float(slope),
                    "cell_weighted_mae": float(
                        np.average(
                            abs(q.observed - q.predicted), weights=q.treated_n_cells
                        )
                    ),
                    **met,
                }
            )
    return pd.DataFrame(rows)


def calibration(pred):
    rows = []
    for (cohort, split, model), g in pred[pred.analysis == "composite_z2"].groupby(
        ["cohort", "split", "model"]
    ):
        g = g.copy()
        g["bin"] = pd.qcut(g.predicted.rank(method="first"), 5, labels=False)
        for b, q in g.groupby("bin"):
            rows.append(
                {
                    "cohort": cohort,
                    "split": split,
                    "model": model,
                    "prediction_quintile": int(b) + 1,
                    "n_pairs": len(q),
                    "mean_predicted": q.predicted.mean(),
                    "mean_observed": q.observed.mean(),
                }
            )
    return pd.DataFrame(rows)


def extra_summaries(pred, params, frozen, out):
    rows = []
    for (analysis, cohort, split), g in pred[
        pred.model == "parsimonious_additive"
    ].groupby(["analysis", "cohort", "split"]):
        rows.append(
            {
                "analysis": analysis,
                "cohort": cohort,
                "split": split,
                "n_pairs": len(g),
                "n_lines": g.cell_line.nunique(),
                "high_cells": int(g.n_high.sum()),
                "treated_cells": int(g.treated_n_cells.sum()),
                "positive_pairs": int((g.n_high > 0).sum()),
                "positive_lines": g[g.n_high > 0].cell_line.nunique(),
                "mean_fraction": g.observed.mean(),
                "cell_weighted_fraction": g.n_high.sum() / g.treated_n_cells.sum(),
            }
        )
    pd.DataFrame(rows).to_csv(out / "endpoint_support.csv", index=False)
    rows = []
    for (cohort, split, model), g in pred[
        (pred.analysis == "composite_z2")
        & pred.model.isin(
            [
                "parsimonious_additive",
                "weighted_parsimonious_additive",
                "drug_only",
                "ddr_fraction_additive",
            ]
        )
    ].groupby(["cohort", "split", "model"]):
        g = g.copy()
        g["count_group"] = pd.qcut(
            g.treated_n_cells, 3, labels=["lower", "middle", "upper"], duplicates="drop"
        )
        for b, v in g.groupby("count_group", observed=True):
            rows.append(
                {
                    "cohort": cohort,
                    "split": split,
                    "model": model,
                    "count_group": str(b),
                    "n_pairs": len(v),
                    "min_treated_cells": v.treated_n_cells.min(),
                    "max_treated_cells": v.treated_n_cells.max(),
                    "mae": float(np.mean(abs(v.observed - v.predicted))),
                    "mean_observed_fraction": v.observed.mean(),
                }
            )
    pd.DataFrame(rows).to_csv(out / "cell_count_errors.csv", index=False)
    rows = []
    for q in params:
        if (
            q["analysis"] == "composite_z2"
            and q["cohort"] == "time_specific"
            and q["split"] == "frozen_outer"
            and q["model"] == "parsimonious_additive"
        ):
            for n, b in zip(q["feature_names"], q["coefficients"]):
                rows.append(
                    {
                        "fold": q["fold"],
                        "chosen_alpha": q["chosen_alpha"],
                        "feature": n,
                        "standardized_coefficient": b,
                    }
                )
    pd.DataFrame(rows).to_csv(out / "main_model_coefficients.csv", index=False)
    frozen[["cell_line", "fold"]].drop_duplicates().sort_values("cell_line").to_csv(
        out / "frozen_line_folds.csv", index=False
    )


def load_inputs(args):
    definitions = args.definitions
    canonical = args.canonical
    modules = {k: v["defined"] for k, v in json.loads(definitions.read_text()).items()}
    modules.update(
        {
            k: v["geneSymbols"]
            for k, v in json.loads(canonical.read_text())["sets"].items()
        }
    )
    if args.processed is not None:
        meta = pd.read_csv(
            args.processed / "cells.csv.gz",
            dtype={"source_time": str, "time": str},
            low_memory=False,
        )
        manifest = json.loads((args.processed / "input_manifest.json").read_text())
        assert list(modules) == manifest["module_column_order"]
        scores = np.load(args.processed / "module_scores.npz")["scores"]
        assert len(meta) == len(scores) == manifest["n_cells"]
        return (
            meta[["cell_line", "perturbation", "time", "ncounts", "cell_quality"]],
            scores,
            modules,
            meta,
        )
    if args.mapping is None:
        raise ValueError("--mapping is required with --input")
    obs, scores, modules = extract_all(args.input, definitions, canonical, args.cache)
    return (
        obs,
        scores,
        modules,
        pd.read_csv(args.mapping, dtype={"source_time": str}, low_memory=False),
    )


def main():
    ap = argparse.ArgumentParser()
    inp = ap.add_mutually_exclusive_group(required=True)
    inp.add_argument("--input", type=Path, help="Raw McFarland h5ad")
    inp.add_argument(
        "--processed",
        type=Path,
        help="Directory containing portable cells, module scores and input manifest",
    )
    ap.add_argument("--mapping", type=Path)
    ap.add_argument("--folds", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--cache", type=Path, default=Path(tempfile.gettempdir()) / "ddr_score_cache"
    )
    ap.add_argument("--definitions", type=Path, required=True)
    ap.add_argument("--canonical", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=4000)
    ap.add_argument("--skip-sensitivities", action="store_true")
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    obs, scores, modules, mapping = load_inputs(args)
    # Keep source names corresponding exactly to pre-existing score definitions.
    modules = {
        ("cycle" if k == "cycle" else "stress" if k == "stress" else k): v
        for k, v in modules.items()
    }
    frozen = pd.read_csv(args.folds)
    frozen = (
        frozen[
            (frozen.analysis == "normalized_composite_z2")
            & (frozen.model == "combined")
        ]
        if "analysis" in frozen
        else frozen
    )
    assert frozen.groupby("cell_line").fold.nunique().max() == 1
    foldmap = frozen.set_index("cell_line").fold.to_dict()
    allpred = []
    tuning = []
    params = []
    refs = []
    cohorts = []
    boot = []
    for time_specific, earlier_controls in [
        (False, False),
        (True, False),
        (True, True),
    ]:
        label = (
            "earlier_vehicle_6h_to_24h"
            if earlier_controls
            else "time_specific" if time_specific else "pooled_comparability"
        )
        pairs, controls, treated = cohort(obs, mapping, time_specific, earlier_controls)
        assert pairs.cell_line.isin(foldmap).all()
        folds = pairs.cell_line.map(foldmap).to_numpy(int)
        regimes = list(dict.fromkeys(pairs.drug))
        allocations = {"frozen_outer": folds}
        alt = np.zeros(len(pairs), int)
        for f, (_, test) in enumerate(
            GroupKFold(5, shuffle=True, random_state=20261117).split(
                pairs, groups=pairs.cell_line
            ),
            1,
        ):
            alt[test] = f

        if not earlier_controls:
            allocations["shuffled_outer_seed20261117"] = alt
        co = pairs.copy().rename(columns={"drug": "regimen", "compound": "drug"})
        co["cohort"] = label
        cohorts.append(co)
        print(
            "COHORT",
            label,
            "pairs",
            len(pairs),
            "lines",
            pairs.cell_line.nunique(),
            "regimes",
            regimes,
            "controls",
            len(np.unique(np.concatenate(list(controls.values())))),
            "treated",
            int(pairs.treated_n_cells.sum()),
            flush=True,
        )
        targets = (
            TARGETS[:1]
            if (not time_specific or earlier_controls or args.skip_sensitivities)
            else TARGETS
        )
        for analysis, target, cutoff, residualize in targets:
            for split, allocation in allocations.items():
                if analysis != "composite_z2" and split != "frozen_outer":
                    continue
                models = (
                    MAIN_MODELS
                    if analysis == "composite_z2" and not earlier_controls
                    else SENS_MODELS
                )
                transform = transformer_for(
                    modules,
                    target,
                    cutoff,
                    residualize,
                    source_metadata=(
                        mapping if analysis == "source_stratified_z2" else None
                    ),
                )
                p, t, m, r = core.run_nested(
                    scores,
                    pairs,
                    controls,
                    treated,
                    allocation,
                    split,
                    transformer=transform,
                    regimes=regimes,
                    models=models,
                )
                p = p.rename(columns={"drug": "regimen", "compound": "drug"})
                p["analysis"] = analysis
                p["cohort"] = label
                for collection in [t, m, r]:
                    for entry in collection:
                        entry.update({"analysis": analysis, "cohort": label})
                allpred.append(p)
                tuning += t
                params += m
                refs += r
                # Incremental outputs allow inspection of main QC results while prespecified sensitivity fits continue.
                pd.concat(allpred, ignore_index=True).to_csv(
                    out / "nested_predictions.csv", index=False
                )
                performance(pd.concat(allpred, ignore_index=True)).to_csv(
                    out / "performance.csv", index=False
                )
                bi = core.intervals(
                    p,
                    args.bootstrap,
                    unit_columns=["cell_line", "regimen"],
                    drugs=[d for d in core.DRUGS if d in set(p.drug)],
                )
                bi["analysis"] = analysis
                bi["cohort"] = label
                boot.append(bi)
                pd.concat(boot, ignore_index=True).to_csv(
                    out / "line_bootstrap.csv", index=False
                )
                print(
                    "RESULT",
                    analysis,
                    label,
                    split,
                    performance(p)
                    .query("drug=='all'")[["model", "r2", "mae"]]
                    .to_json(orient="records"),
                    flush=True,
                )
    pred = pd.concat(allpred, ignore_index=True)
    pd.DataFrame(tuning).to_csv(out / "inner_tuning.csv", index=False)
    (out / "model_parameters.json").write_text(json.dumps(params, indent=2) + "\n")
    (out / "fold_references.json").write_text(json.dumps(refs, indent=2) + "\n")
    pd.concat(cohorts, ignore_index=True).to_csv(out / "cohort_pairs.csv", index=False)
    calibration(pred).to_csv(out / "calibration_bins.csv", index=False)
    regmetrics = []
    for (analysis, cohortname, split, model, regimen), g in pred.groupby(
        ["analysis", "cohort", "split", "model", "regimen"]
    ):
        regmetrics.append(
            {
                "analysis": analysis,
                "cohort": cohortname,
                "split": split,
                "model": model,
                "regimen": regimen,
                **core.metrics(g.observed.to_numpy(), g.predicted.to_numpy()),
            }
        )
    pd.DataFrame(regmetrics).to_csv(out / "regimen_performance.csv", index=False)
    spec = {
        "main_model_chosen_before_QC_outcomes": "parsimonious_additive",
        "status": "post-review method refinement; outer outcomes previously inspected; shuffled split is stability only",
        "quality": "source author cell_quality==normal",
        "baseline_control": "exact source_experiment and source_time DMSO only; untreated no-vehicle and other experiments excluded",
        "target_reference": "unique training-line eligible matched control cells, global median/MAD; not drug- or experiment-specific cutoff",
        "primary_scientific_unit": "cell line x drug x time; regimen-specific unpenalized intercepts",
        "pooled_unit": "line x drug, represented source times pooled; comparability only",
        "alpha_candidates": core.ALPHAS,
        "inner_selection": "3fold GroupKFold byline, pooled held-inner-pair fractional logloss",
        "models": MAIN_MODELS,
        "main_features": core.FEATURES,
        "hierarchical_deviation_penalty": 4,
        "sensitivity_definitions": TARGETS,
        "sensitivity_models": SENS_MODELS,
        "normalization": "mean of log1p(count/ncounts*10000) over available genes",
        "source_mapping_file": (
            args.mapping.name if args.mapping is not None else "cells.csv.gz"
        ),
        "source_mapping_sha256": hashlib.sha256(
            (
                args.mapping
                if args.mapping is not None
                else args.processed / "cells.csv.gz"
            ).read_bytes()
        ).hexdigest(),
        "bootstrap": "4000 entire-line draws; fixed held-out predictions, not refitting/selection uncertainty",
        "all_gene_definitions": modules,
        "source_stratified_sensitivity": "separate training-control references per source experiment and time; same pairs, folds and models",
        "weighted_model": "treated-cell-count weights normalized to mean one; inner selection remains equal-pair fractional logloss",
    }
    (out / "analysis_specification.json").write_text(json.dumps(spec, indent=2) + "\n")
    extra_summaries(pred, params, frozen, out)
    print("ALL SOURCE-MATCHED ANALYSES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
