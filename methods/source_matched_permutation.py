#!/usr/bin/env python3
"""Conditional line-block permutation for the final source-matched additive model."""
from pathlib import Path
import argparse, json, tempfile
import numpy as np
import pandas as pd
from scipy.special import expit
import source_matched_forecasting as s
import nested_forecasting as c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path)
    ap.add_argument("--processed", type=Path)
    ap.add_argument("--mapping", type=Path)
    ap.add_argument("--folds", type=Path, required=True)
    ap.add_argument("--definitions", type=Path, required=True)
    ap.add_argument("--canonical", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--iterations", type=int, default=999)
    ap.add_argument(
        "--cache", type=Path, default=Path(tempfile.gettempdir()) / "ddr_score_cache"
    )
    a = ap.parse_args()
    out = a.output
    obs, scores, modules, mapping = s.load_inputs(a)
    pairs, controls, treated = s.cohort(obs, mapping, True)
    frozen = pd.read_csv(a.folds)
    frozen = (
        frozen[
            (frozen.analysis == "normalized_composite_z2")
            & (frozen.model == "combined")
        ]
        if "analysis" in frozen
        else frozen
    )
    foldmap = frozen.set_index("cell_line").fold.to_dict()
    folds = pairs.cell_line.map(foldmap).to_numpy(int)
    regimes = list(dict.fromkeys(pairs.drug))
    D = np.column_stack([pairs.drug.eq(d) for d in regimes]).astype(float)
    model_params = json.loads((out / "model_parameters.json").read_text())
    pred = pd.read_csv(out / "nested_predictions.csv")
    pred = pred[
        (pred.analysis == "composite_z2")
        & (pred.cohort == "time_specific")
        & (pred.split == "frozen_outer")
    ]
    key = ["cell_line", "regimen"]
    combined = pred[pred.model == "parsimonious_additive"].set_index(key)
    drug = pred[pred.model == "drug_only"].set_index(key).loc[combined.index]
    observed = float(
        np.mean((combined.observed - drug.predicted) ** 2)
        - np.mean((combined.observed - combined.predicted) ** 2)
    )
    transform = s.transformer_for(modules, "ddr_composite", 2, False)
    frames = []
    maxerr = 0.0
    patterns = {line: tuple(sorted(g.drug)) for line, g in pairs.groupby("cell_line")}
    line_regimen_rows = {
        (line, d): i for i, (line, d) in enumerate(zip(pairs.cell_line, pairs.drug))
    }
    for fold in sorted(set(folds)):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        B, y, ref = transform(
            scores, pairs, controls, treated, pairs.iloc[train].cell_line
        )
        param = [
            p
            for p in model_params
            if p["analysis"] == "composite_z2"
            and p["cohort"] == "time_specific"
            and p["split"] == "frozen_outer"
            and p["model"] == "parsimonious_additive"
            and p["fold"] == fold
        ][0]
        alpha = param["chosen_alpha"]
        X, penalty, names, sc = c.design(
            B, D, [0, 1, 2, 3], False, train, regimes=regimes
        )
        beta, _, _ = c.fit_model(X[train], y[train], penalty, alpha)
        p = expit(X[test] @ beta)
        actual = combined.loc[
            list(zip(pairs.iloc[test].cell_line, pairs.iloc[test].drug))
        ].predicted.to_numpy()
        maxerr = max(maxerr, float(np.max(abs(p - actual))))
        partitions = []
        for idx in [train, test]:
            strata = {}
            for line in sorted(pairs.iloc[idx].cell_line.unique()):
                strata.setdefault(patterns[line], []).append(line)
            partitions.append((idx, list(strata.values())))
        frames.append((train, test, B, y, alpha, partitions))
    assert maxerr < 1e-7, maxerr
    rng = np.random.default_rng(883)
    null = []
    drug_mse = float(np.mean((combined.observed - drug.predicted) ** 2))
    for it in range(a.iterations):
        error = 0.0
        count = 0
        for train, test, B, y, alpha, parts in frames:
            perm = B.copy()
            for idx, strata in parts:
                swap = {}
                for members in strata:
                    swap.update(dict(zip(members, rng.permutation(members))))
                perm[idx] = np.vstack(
                    [
                        B[line_regimen_rows[(swap[line], regimen)]]
                        for line, regimen in zip(
                            pairs.iloc[idx].cell_line, pairs.iloc[idx].drug
                        )
                    ]
                )
            X, penalty, names, sc = c.design(
                perm, D, [0, 1, 2, 3], False, train, regimes=regimes
            )
            beta, _, _ = c.fit_model(X[train], y[train], penalty, alpha)
            p = expit(X[test] @ beta)
            error += np.sum((y[test] - p) ** 2)
            count += len(test)
        null.append(drug_mse - error / count)
    result = {
        "model": "parsimonious_additive",
        "analysis": "composite_z2",
        "cohort": "time_specific",
        "split": "frozen_outer",
        "n_pairs": len(pairs),
        "n_lines": pairs.cell_line.nunique(),
        "observed_mse_improvement_vs_regimen_only": observed,
        "iterations": a.iterations,
        "p_one_sided": float(
            (1 + np.sum(np.array(null) >= observed)) / (a.iterations + 1)
        ),
        "null_mean": float(np.mean(null)),
        "null_95th_percentile": float(np.quantile(null, 0.95)),
        "maximum_unpermuted_prediction_reconstruction_error": maxerr,
        "method": "permute whole line-specific baseline vectors jointly across all available regimens; exchange only between lines with identical regimen-availability patterns and separately within train/test partitions of each outer fold; keep regimens, response blocks, cell counts, folds, and outer-selected alphas fixed; refit model coefficients",
        "inference_scope": "conditional block association test after fixed nested selections, not rerunning model selection under the null; complements paired line-bootstrap confidence intervals",
    }
    (out / "structured_permutation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    pd.DataFrame({"mse_improvement_vs_regimen_only": null}).to_csv(
        out / "permutation_null.csv", index=False
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
