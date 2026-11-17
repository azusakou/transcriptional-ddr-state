#!/usr/bin/env python3
"""Grouped fractional-logit fitting and conditional cell-line intervals."""
from __future__ import annotations
import argparse, json, hashlib, importlib.metadata
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import GroupKFold

GENES = {
    "ddr": "TP53 CDKN1A MDM2 GADD45A GADD45B GADD45G DDB2 XPC BBC3 BAX PMAIP1 FAS RRM2B SESN1 BTG2 FDXR PLK3 PPM1D ATM ATR CHEK1 CHEK2 BRCA1 BRCA2 RAD51 PARP1 XRCC5 XRCC6 H2AFX MRE11 NBN FANCD2 CLSPN TOPBP1".split(),
    "p53": "TP53 CDKN1A MDM2 DDB2 GADD45A GADD45B RRM2B BBC3 BAX PMAIP1 PPM1D FDXR SESN1 BTG2 FAS".split(),
    "cycle": "MKI67 TOP2A PCNA MCM2 MCM3 MCM4 MCM5 MCM6 MCM7 CDK1 CCNB1 CCNB2 CCNE1 CCNA2 CDC20 AURKA AURKB".split(),
    "stress": "ATF3 ATF4 ATF6 DDIT3 HSPA1A HSPA1B HSP90AA1 HMOX1 JUN FOS EGR1 IER3 XBP1 ERN1 EIF2AK3 DNAJB1 HSPA5".split(),
}
DRUGS = ["Gemcitabine", "Idasanutlin", "Prexasertib", "Bortezomib"]
MODELS = {
    "drug_only": ([], False),
    "ddr_fraction_additive": ([0], False),
    "parsimonious_additive": ([0, 1, 2, 3], False),
    "ddr_fraction_hierarchical": ([0], True),
    "parsimonious_hierarchical": ([0, 1, 2, 3], True),
}
FEATURES = [
    "baseline_ddr_fraction_z2",
    "baseline_TP53_response_mean_z",
    "baseline_cycle_replication_mean_z",
    "baseline_stress_mean_z",
]
ALPHAS = [0.3, 3.0, 30.0]


def read_col(group, key):
    a = group[key]

    def dec(v):
        return np.array(
            [x.decode() if isinstance(x, bytes) else x for x in v], dtype=object
        )

    if isinstance(a, h5py.Group):
        cat = dec(a["categories"][:])
        codes = a["codes"][:]
        out = np.full(len(codes), "NA", object)
        mask = codes >= 0
        out[mask] = cat[codes[mask]]
        return out
    return dec(a[:])


def design(B, D, cols, interactions, train, scaling=None, regimes=None):
    regimes = DRUGS if regimes is None else regimes
    k = len(regimes)
    b = B[:, cols]
    if scaling is None:
        center = b[train].mean(0)
        scale = b[train].std(0)
        scale[scale < 1e-12] = 1.0
    else:
        center, scale = np.array(scaling["center"]), np.array(scaling["scale"])
    b = (b - center) / scale
    names = ["regimen_intercept_" + d for d in regimes] + [FEATURES[c] for c in cols]
    parts = [D, b]
    penalty = [0.0] * k + [1.0] * len(cols)
    if interactions:
        # Centered indicators define drug-specific deviations from the across-drug shared slope.
        ib = ((D - 1 / k)[:, :, None] * b[:, None, :]).reshape(len(B), -1)
        parts.append(ib)
        penalty += [4.0] * (k * len(cols))
        names += [FEATURES[c] + "_deviation_" + d for d in regimes for c in cols]
    X = np.column_stack(parts)
    return (
        X,
        np.array(penalty),
        names,
        {"center": center.tolist(), "scale": scale.tolist()},
    )


def fit_model(X, y, penalty, alpha, weights=None):
    w = (
        np.ones(len(y))
        if weights is None
        else np.asarray(weights, float) / np.mean(weights)
    )
    # Equal-pair fractional-binomial likelihood. Drug intercepts are unpenalized.
    init = np.zeros(X.shape[1])
    n_regimes = int(np.sum(penalty == 0))
    for j in range(n_regimes):
        mask = X[:, j] > 0
        mean = np.average(y[mask], weights=w[mask])
        init[j] = np.log(
            np.clip(mean, 1e-7, 1 - 1e-7) / np.clip(1 - mean, 1e-7, 1 - 1e-7)
        )
    if X.shape[1] == n_regimes:
        return init, True, 0

    def fn(beta):
        eta = X @ beta
        p = expit(eta)
        value = np.sum(w * (np.logaddexp(0, eta) - y * eta)) + 0.5 * alpha * np.sum(
            penalty * beta**2
        )
        grad = X.T @ (w * (p - y)) + alpha * penalty * beta
        return value, grad

    result = minimize(
        fn,
        init,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 1500, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(str(result.message))
    return result.x, True, int(result.nit)


def logloss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def metrics(y, p):
    den = np.sum((y - y.mean()) ** 2)
    return {
        "n_pairs": len(y),
        "r2": float(1 - np.sum((y - p) ** 2) / den) if den > 0 else np.nan,
        "mae": float(np.mean(abs(y - p))),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "fractional_logloss": float(np.mean(logloss(y, p))),
        "pearson_r": float(np.corrcoef(y, p)[0, 1]),
        "mean_observed": float(y.mean()),
        "mean_predicted": float(p.mean()),
        "minimum_prediction": float(p.min()),
        "maximum_prediction": float(p.max()),
    }


def run_nested(
    scores,
    pairs,
    controls,
    treated,
    folds,
    split,
    transformer=None,
    regimes=None,
    models=None,
):
    assert transformer is not None, "A training-reference transformer is required"
    regimes = DRUGS if regimes is None else regimes
    models = MODELS if models is None else models
    D = np.column_stack([pairs.drug.eq(d) for d in regimes]).astype(float)
    predictions = []
    tuning = []
    params = []
    refs = []
    for fold in sorted(set(folds)):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        trainlines = set(pairs.iloc[train].cell_line)
        testlines = set(pairs.iloc[test].cell_line)
        assert not trainlines & testlines
        B, y, ref = transformer(scores, pairs, controls, treated, trainlines)
        refs.append({"split": split, "outer_fold": int(fold), "level": "outer", **ref})
        # Every inner label reference and biological-feature transformation is fitted independently.
        inner = []
        for innerfold, (itr, ival) in enumerate(
            GroupKFold(3).split(train, groups=pairs.iloc[train].cell_line), 1
        ):
            tr = train[itr]
            val = train[ival]
            Bi, yi, ri = transformer(
                scores, pairs, controls, treated, pairs.iloc[tr].cell_line
            )
            inner.append((tr, val, Bi, yi))
            refs.append(
                {
                    "split": split,
                    "outer_fold": int(fold),
                    "inner_fold": innerfold,
                    "level": "inner",
                    **ri,
                }
            )
        for model, (cols, interactions) in models.items():
            Dmodel = np.ones((len(pairs), 1)) if model == "baseline_only" else D
            modellabels = ["global"] if model == "baseline_only" else regimes
            trials = []
            if model == "drug_only":
                chosen = 0.0
            else:
                for alpha in ALPHAS:
                    losses = []
                    for innerfold, (tr, val, Bi, yi) in enumerate(inner, 1):
                        X, penalty, names, sc = design(
                            Bi, Dmodel, cols, interactions, tr, regimes=modellabels
                        )
                        beta, ok, nit = fit_model(
                            X[tr],
                            yi[tr],
                            penalty,
                            alpha,
                            weights=(
                                pairs.iloc[tr].treated_n_cells.to_numpy()
                                if model.startswith("weighted_")
                                else None
                            ),
                        )
                        p = expit(X[val] @ beta)
                        losses.extend(logloss(yi[val], p))
                        tuning.append(
                            {
                                "split": split,
                                "outer_fold": int(fold),
                                "inner_fold": innerfold,
                                "model": model,
                                "alpha": alpha,
                                "n_validation_pairs": len(val),
                                "fractional_logloss": float(
                                    np.mean(logloss(yi[val], p))
                                ),
                                "mse": float(np.mean((yi[val] - p) ** 2)),
                            }
                        )
                    trials.append((float(np.mean(losses)), alpha))
                best = min(q[0] for q in trials)
                chosen = max(a for loss, a in trials if loss <= best + 1e-8)
            X, penalty, names, sc = design(
                B, Dmodel, cols, interactions, train, regimes=modellabels
            )
            beta, ok, nit = fit_model(
                X[train],
                y[train],
                penalty,
                chosen,
                weights=(
                    pairs.iloc[train].treated_n_cells.to_numpy()
                    if model.startswith("weighted_")
                    else None
                ),
            )
            p = expit(X[test] @ beta)
            q = pairs.iloc[test].copy()
            q["split"] = split
            q["fold"] = int(fold)
            q["model"] = model
            q["alpha"] = chosen
            q["observed"] = y[test]
            q["predicted"] = p
            q["n_high"] = np.rint(y[test] * q.treated_n_cells).astype(int)
            predictions.append(q)
            params.append(
                {
                    "split": split,
                    "fold": int(fold),
                    "model": model,
                    "chosen_alpha": chosen,
                    "feature_columns": cols,
                    "interactions": interactions,
                    "base_scaling": sc,
                    "feature_names": names,
                    "penalty_multipliers": penalty.tolist(),
                    "coefficients": beta.tolist(),
                    "drug_intercepts_unpenalized": True,
                    "interaction_deviation_penalty_multiplier": 4,
                    "optimizer_iterations": nit,
                    "inner_candidate_loss": [
                        {"alpha": a, "loss": loss} for loss, a in trials
                    ],
                }
            )
        print(split, "fold", fold, "done", flush=True)
    return pd.concat(predictions, ignore_index=True), tuning, params, refs


def intervals(pred, nboot=4000, unit_columns=None, drugs=None):
    unit_columns = ["cell_line", "drug"] if unit_columns is None else unit_columns
    drugs = DRUGS if drugs is None else drugs
    models = list(pred.model.unique())
    rows = []
    rng = np.random.default_rng(6402)
    for split, ps in pred.groupby("split"):
        for drug in ["all"] + drugs:
            q = ps if drug == "all" else ps[ps.drug == drug]
            base = q[q.model == "drug_only"].set_index(unit_columns)
            lines = base.index.get_level_values(0).to_numpy()
            unique = np.array(sorted(set(lines)))
            blocks = {line: np.flatnonzero(lines == line) for line in unique}
            y = base.observed.to_numpy()
            p = {
                m: q[q.model == m]
                .set_index(unit_columns)
                .loc[base.index]
                .predicted.to_numpy()
                for m in models
            }
            values = {m: [] for m in models}
            deltas = {
                (m, r): []
                for m in models
                for r in ["drug_only", "ddr_fraction_additive"]
                if m != r
            }
            for k in range(nboot):
                idx = np.concatenate(
                    [blocks[x] for x in rng.choice(unique, len(unique), replace=True)]
                )
                v = y[idx]
                den = np.sum((v - v.mean()) ** 2)
                r2 = {m: 1 - np.sum((v - pm[idx]) ** 2) / den for m, pm in p.items()}
                mae = {m: np.mean(abs(v - pm[idx])) for m, pm in p.items()}
                for m in models:
                    values[m].append([r2[m], mae[m]])
                for m, r in deltas:
                    deltas[(m, r)].append([r2[m] - r2[r], mae[r] - mae[m]])
            point = {m: metrics(y, pm) for m, pm in p.items()}
            for m, vals in values.items():
                a = np.asarray(vals)
                for j, stat in enumerate(["r2", "mae"]):
                    rows.append(
                        {
                            "split": split,
                            "drug": drug,
                            "model": m,
                            "reference_model": "none",
                            "statistic": stat,
                            "estimate": point[m][stat],
                            "ci_low": float(np.quantile(a[:, j], 0.025)),
                            "ci_high": float(np.quantile(a[:, j], 0.975)),
                            "n_lines": len(unique),
                            "n_bootstrap": nboot,
                        }
                    )
            for (m, r), vals in deltas.items():
                a = np.asarray(vals)
                est = [
                    point[m]["r2"] - point[r]["r2"],
                    point[r]["mae"] - point[m]["mae"],
                ]
                for j, stat in enumerate(["delta_r2", "mae_improvement"]):
                    rows.append(
                        {
                            "split": split,
                            "drug": drug,
                            "model": m,
                            "reference_model": r,
                            "statistic": stat,
                            "estimate": est[j],
                            "ci_low": float(np.quantile(a[:, j], 0.025)),
                            "ci_high": float(np.quantile(a[:, j], 0.975)),
                            "n_lines": len(unique),
                            "n_bootstrap": nboot,
                        }
                    )
    return pd.DataFrame(rows)
