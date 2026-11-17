#!/usr/bin/env python3
"""Reproduce source-filtered, context-matched single-cell classification.
Supply --input scored cells or --rebuild-input with raw H5ADs and source mapping.
--output selects all results; --save-model also refits the complete-data artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import shutil
import sys
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261117
GENE_IDENTITY_PATH = Path(__file__).with_name("classifier_gene_identity_map.json")
DDR = set(
    "TP53 CDKN1A MDM2 GADD45A GADD45B GADD45G DDB2 XPC BBC3 BAX PMAIP1 FAS RRM2B SESN1 BTG2 FDXR PLK3 PPM1D ATM ATR CHEK1 CHEK2 BRCA1 BRCA2 RAD51 PARP1 XRCC5 XRCC6 H2AFX MRE11 NBN FANCD2 CLSPN TOPBP1".split()
)
TP53 = set(
    "TP53 CDKN1A MDM2 DDB2 GADD45A GADD45B RRM2B BBC3 BAX PMAIP1 PPM1D FDXR SESN1 BTG2 FAS".split()
)
GENE_SETS = {
    "target_DDR": DDR,
    "integrated_stress": set(
        "ATF3 ATF4 ATF6 DDIT3 HSPA1A HSPA1B HSP90AA1 HMOX1 JUN FOS EGR1 IER3 XBP1 ERN1 EIF2AK3 DNAJB1 HSPA5".split()
    ),
    "cell_cycle_replication": set(
        "MKI67 TOP2A PCNA MCM2 MCM3 MCM4 MCM5 MCM6 MCM7 CDK1 CCNB1 CCNB2 CCNE1 CCNA2 CDC20 AURKA AURKB".split()
    ),
    "E2F_replication": set(
        "E2F1 E2F2 E2F3 MCM2 MCM3 MCM4 MCM5 MCM6 MCM7 PCNA RRM1 RRM2 TYMS TK1 CDC6 CCNE1 CCNA2 CDK2".split()
    ),
    "AP1_immediate_early": set(
        "JUN JUNB JUND FOS FOSB FOSL1 FOSL2 ATF3 EGR1 IER2 IER3 DUSP1 DUSP5 ZFP36".split()
    ),
    "ATF4_integrated_stress": set(
        "ATF4 DDIT3 ASNS TRIB3 SLC7A5 SLC7A11 HERPUD1 CEBPB CEBPG PPP1R15A DNAJC3 HSPA5".split()
    ),
    "HSF1_heat_shock": set(
        "HSF1 HSPA1A HSPA1B HSPA5 HSP90AA1 HSP90AB1 DNAJB1 DNAJA1 HSPB1 HSPD1 HSPE1".split()
    ),
    "XBP1_UPR": set(
        "XBP1 ERN1 ATF6 HSPA5 DNAJB9 HERPUD1 PDIA4 PDIA6 EDEM1 SEC61A1 SEL1L".split()
    ),
    "NFkB_inflammatory_stress": set(
        "NFKB1 NFKBIA RELA TNFAIP3 IL6 CXCL8 ICAM1 BIRC3 JUNB FOS IER3".split()
    ),
}
# Remove target genes before score calculation, rather than removing only score names.
GENE_SETS = {
    k: (v if k == "target_DDR" else v - DDR - TP53) for k, v in GENE_SETS.items()
}
TASKS = {
    "McFarlandTsherniak2020": [
        "Bortezomib",
        "Gemcitabine",
        "Idasanutlin",
        "Prexasertib",
    ],
    "SrivatsanTrapnell2020_sciplex2": ["Nutlin", "SAHA"],
}
QC = ["log_ncounts", "log_ngenes"]
CONTEXT_COLUMNS = []
CYCLE = ["cell_cycle_replication", "E2F_replication"]
STRESS = [k for k in GENE_SETS if k not in ["target_DDR"] + CYCLE]
FEATURES = {
    "QC": QC,
    "QC_cellcycle": QC + CYCLE,
    "biology_QC_cellcycle": QC + CYCLE + STRESS,
}


def read_col(group, key):
    item = group[key]
    if isinstance(item, h5py.Group):
        cats = item["categories"].asstr()[:]
        codes = item["codes"][:]
        return np.array([cats[c] if c >= 0 else "NA" for c in codes])
    return item.asstr()[:] if item.dtype.kind in "OSU" else item[:]


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def build_input(data_root, out, source_mapping):
    mapping = pd.read_csv(source_mapping, low_memory=False)
    identity_manifest = json.loads(GENE_IDENTITY_PATH.read_text())
    all_cells, coverage, inputs, identity_rows = [], [], [], []
    for dataset, treatments in TASKS.items():
        path = data_root / f"{dataset}.h5ad"
        print("Extracting", dataset, flush=True)
        with h5py.File(path) as handle:
            obs = handle["obs"]
            perturb = read_col(obs, "perturbation")
            # In sciPlex2, dose-zero cells retain drug labels; generic 'control'
            # cells have no assigned dose and are not used as matched controls.
            included = treatments + (
                ["control"] if dataset == "McFarlandTsherniak2020" else []
            )
            keep = np.flatnonzero(np.isin(perturb, included))
            source_key = obs.attrs["_index"]
            if isinstance(source_key, bytes):
                source_key = source_key.decode()
            cell = pd.DataFrame(
                {
                    "dataset_id": dataset,
                    "source_row": keep,
                    "source_barcode": read_col(obs, source_key)[keep],
                    "perturbation": perturb[keep],
                    "cell_line": read_col(obs, "cell_line")[keep],
                    "ncounts": read_col(obs, "ncounts")[keep].astype(float),
                    "ngenes": read_col(obs, "ngenes")[keep].astype(float),
                }
            )
            for name in ["dose_value", "time", "channel", "well"]:
                if name in obs:
                    cell[name] = read_col(obs, name)[keep]
            cell["is_control"] = (
                (cell.perturbation == "control")
                if dataset == "McFarlandTsherniak2020"
                else (pd.to_numeric(cell.dose_value) == 0)
            )
            cell["sample_id"] = dataset + ":" + cell.source_barcode.astype(str)
            if cell.sample_id.duplicated().any():
                raise ValueError("Non-unique source barcode")
            genes = handle["var"]
            gene_key = next(
                k
                for k in ["gene_symbol", "gene_name", "_index", "index", "features"]
                if k in genes
            )
            names = list(read_col(genes, gene_key))
            lookup = {g: i for i, g in enumerate(names)}
            # Frozen HGNC approved/previous-symbol mappings cover all predefined
            # programmes. Each source identity contributes once within a programme.
            requested = sorted(set().union(*GENE_SETS.values()))
            symbol_map = identity_manifest["dataset_source_symbol_maps"][dataset]
            resolved = {}
            for gene in requested:
                symbol = gene if gene in lookup else symbol_map.get(gene)
                if symbol is not None and symbol in lookup:
                    resolved[gene] = symbol
                identity_rows.append(
                    {
                        "dataset_id": dataset,
                        "specified_gene": gene,
                        "source_symbol": resolved.get(gene, ""),
                        "source_column": (
                            lookup[resolved[gene]] if gene in resolved else -1
                        ),
                    }
                )
            selected = sorted(resolved)
            indices = np.array([lookup[resolved[g]] for g in selected])
            assert len(indices) == len(
                set(indices)
            ), "A source gene cannot contribute twice"
            target_sources = {resolved[g] for g in DDR if g in resolved}
            for programme, members in GENE_SETS.items():
                if programme != "target_DDR":
                    assert not target_sources.intersection(
                        resolved[g] for g in members if g in resolved
                    )
            x = handle["X"]
            shape = tuple(x.attrs["shape"])
            enc = x.attrs["encoding-type"]
            if isinstance(enc, bytes):
                enc = enc.decode()
            if enc == "csc_matrix":
                count = np.zeros((len(keep), len(selected)), dtype=float)
                row_map = np.full(shape[0], -1, dtype=int)
                row_map[keep] = np.arange(len(keep))
                for column, index in enumerate(indices):
                    begin, end = int(x["indptr"][index]), int(x["indptr"][index + 1])
                    rows = x["indices"][begin:end]
                    wanted = row_map[rows] >= 0
                    count[row_map[rows[wanted]], column] = x["data"][begin:end][wanted]
            elif enc == "csr_matrix":
                matrix = sparse.csr_matrix(
                    (x["data"][:], x["indices"][:], x["indptr"][:]), shape=shape
                )
                count = matrix[keep][:, indices].toarray().astype(float)
                del matrix
            else:
                raise ValueError(enc)
            if not np.all(cell.ncounts > 0):
                raise ValueError("Non-positive library size")
            normalized = np.log1p(count * (10000 / cell.ncounts.to_numpy())[:, None])
            for name, markers in GENE_SETS.items():
                ix = [j for j, gene in enumerate(selected) if gene in markers]
                if not ix:
                    raise ValueError(f"No genes for {dataset}: {name}")
                cell[name] = normalized[:, ix].mean(axis=1)
                coverage.append(
                    {
                        "dataset_id": dataset,
                        "programme": name,
                        "n_present": len(ix),
                        "genes_present": ";".join(selected[j] for j in ix),
                        "overlap_with_target": (
                            ";".join(sorted(markers & DDR))
                            if name != "target_DDR"
                            else "target"
                        ),
                    }
                )
            cell["log_ncounts"] = np.log1p(cell.ncounts)
            cell["log_ngenes"] = np.log1p(cell.ngenes)
            cell["task_name"] = cell.perturbation
            if dataset == "McFarlandTsherniak2020":
                cols = [
                    "source_row",
                    "source_experiment",
                    "source_condition",
                    "source_time",
                    "cell_quality",
                ]
                cell = cell.merge(mapping[cols], on="source_row", validate="one_to_one")
                cell = cell[cell.cell_quality == "normal"].copy()
                cell = cell[
                    (~cell.is_control) | cell.source_condition.str.casefold().eq("dmso")
                ].copy()
                cell["source_time"] = cell.source_time.astype(str)
                cell["reference_group"] = (
                    cell.source_experiment + "__" + cell.source_time
                )
                rng = np.random.default_rng(SEED)
                matched = []
                for _, group in cell.groupby(
                    ["source_experiment", "source_time", "cell_line"], sort=True
                ):
                    eligible = sorted(set(group.loc[~group.is_control, "perturbation"]))
                    controls = group.index[group.is_control].to_numpy()
                    if not eligible or not len(controls):
                        continue
                    order = rng.permutation(controls)
                    begin = int(rng.integers(len(eligible)))
                    cell.loc[order, "task_name"] = [
                        eligible[(j + begin) % len(eligible)] for j in range(len(order))
                    ]
                    matched.extend(group.index)
                cell = cell.loc[matched].copy().reset_index(drop=True)
            else:
                cell["source_experiment"] = "sciplex2"
                cell["source_time"] = "source_protocol"
                cell["source_condition"] = cell.perturbation
                cell["cell_quality"] = "source_provided"
                cell["reference_group"] = "sciplex2"
            cell["task_name"] = dataset + "__" + cell.task_name
            all_cells.append(cell)
        inputs.append(
            {
                "dataset_id": dataset,
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": checksum(path),
            }
        )
    table = pd.concat(all_cells, ignore_index=True)
    assert table.sample_id.is_unique
    table.to_csv(out / "scored_cells.csv.gz", index=False, float_format="%.10g")
    pd.DataFrame(coverage).to_csv(out / "gene_sets.csv", index=False)
    pd.DataFrame(identity_rows).to_csv(out / "gene_identity_mapping.csv", index=False)
    (out / "source_files.json").write_text(json.dumps(inputs, indent=2) + "\n")
    return table


def thresholds_and_labels(df, training):
    labels = np.zeros(len(df), dtype=int)
    thresholds = {}
    for dataset in df.reference_group.unique():
        controls = df.loc[
            training & (df.reference_group == dataset) & df.is_control, "target_DDR"
        ].to_numpy()
        if not len(controls):
            raise ValueError("A test dataset has no training controls")
        center = np.median(controls)
        mad = np.median(np.abs(controls - center))
        scale = 1.4826 * mad
        method = "median/MAD"
        if scale <= 1e-12:
            center, scale, method = (
                np.mean(controls),
                np.std(controls),
                "mean/SD (zero MAD)",
            )
        if scale <= 1e-12:
            raise ValueError("Degenerate target reference")
        cut = float(center + 2 * scale)
        thresholds[dataset] = {
            "center": float(center),
            "scale": float(scale),
            "cutoff_z2": cut,
            "n_training_controls": int(len(controls)),
            "method": method,
        }
        mask = df.reference_group == dataset
        labels[mask] = (df.loc[mask, "target_DDR"] > cut).astype(int)
    return labels, thresholds


def new_model():
    # Fixed in advance; no feature-set, hyperparameter or best-model selection on test data.
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=4,
        ),
    )


def metric(y, p):
    return {
        "n_cells": len(y),
        "n_positive": int(np.sum(y)),
        "positive_rate": float(np.mean(y)),
        "AUROC": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan,
        "AP": float(average_precision_score(y, p)) if np.sum(y) else np.nan,
    }


def run_cv(df, scheme, out):
    memberships = df[
        [
            "sample_id",
            "dataset_id",
            "cell_line",
            "perturbation",
            "is_control",
            "task_name",
            "reference_group",
            "source_experiment",
            "source_time",
        ]
    ].copy()
    memberships["test_fold"] = ""
    predictions = df[
        [
            "sample_id",
            "dataset_id",
            "cell_line",
            "perturbation",
            "is_control",
            "task_name",
            "reference_group",
            "source_experiment",
            "source_time",
        ]
    ].copy()
    predictions["observed"] = -1
    for key in FEATURES:
        predictions[key] = np.nan
    per_fold, per_task, preprocessing, importance = [], [], [], []
    if scheme == "leave_task_out":
        folds = [
            (
                task,
                np.flatnonzero(df.task_name != task),
                np.flatnonzero(df.task_name == task),
            )
            for task in sorted(df.task_name.unique())
        ]
    else:
        splitter = GroupKFold(n_splits=5)
        folds = [
            (str(i + 1), tr, te)
            for i, (tr, te) in enumerate(splitter.split(df, groups=df.cell_line))
        ]
    for fold, tr, te in folds:
        print(scheme, fold, len(tr), len(te), flush=True)
        assert not set(df.iloc[tr].sample_id) & set(df.iloc[te].sample_id)
        if scheme != "leave_task_out":
            assert not set(df.iloc[tr].cell_line) & set(df.iloc[te].cell_line)
        train_mask = np.zeros(len(df), dtype=bool)
        train_mask[tr] = True
        y, cuts = thresholds_and_labels(df, train_mask)
        memberships.loc[te, "test_fold"] = fold
        predictions.loc[te, "observed"] = y[te]
        for ds, cut in cuts.items():
            preprocessing.append(
                {"scheme": scheme, "fold": fold, "reference_group": ds, **cut}
            )
        for name, cols in FEATURES.items():
            model = new_model()
            model.fit(df.iloc[tr][cols], y[tr])
            p = model.predict_proba(df.iloc[te][cols])[:, 1]
            predictions.loc[te, name] = p
            per_fold.append(
                {
                    "scheme": scheme,
                    "fold": fold,
                    "feature_set": name,
                    **metric(y[te], p),
                }
            )
            for task in sorted(df.iloc[te].task_name.unique()):
                mask = df.iloc[te].task_name.to_numpy() == task
                per_task.append(
                    {
                        "scheme": scheme,
                        "fold": fold,
                        "task_name": task,
                        "feature_set": name,
                        **metric(y[te][mask], p[mask]),
                    }
                )
            if scheme == "leave_task_out" and name == "biology_QC_cellcycle":
                rng = np.random.default_rng(SEED)
                x_test = df.iloc[te][cols].copy()
                # Joint permutations retain correlation among all columns within each group.
                # Every test fold is one task, so permutations remain within that task.
                for group, group_cols in {
                    "stress_programmes_joint": STRESS,
                    "cellcycle_programmes_joint": CYCLE,
                    "QC_joint": QC,
                }.items():
                    base = metric(y[te], p)
                    for rep in range(10):
                        xp = x_test.copy()
                        perm = np.arange(len(xp))
                        references = df.iloc[te].reference_group.to_numpy()
                        for reference in np.unique(references):
                            ix = np.flatnonzero(references == reference)
                            perm[ix] = rng.permutation(ix)
                        xp.loc[:, group_cols] = x_test[group_cols].to_numpy()[perm]
                        pp = model.predict_proba(xp)[:, 1]
                        mm = metric(y[te], pp)
                        importance.append(
                            {
                                "task_name": fold,
                                "group": group,
                                "repeat": rep,
                                "AP_drop": base["AP"] - mm["AP"],
                                "AUROC_drop": base["AUROC"] - mm["AUROC"],
                            }
                        )
    predictions.to_csv(
        out / f"{scheme}_predictions.csv.gz", index=False, float_format="%.17g"
    )
    memberships.to_csv(out / f"{scheme}_fold_membership.csv.gz", index=False)
    pd.DataFrame(preprocessing).to_csv(out / f"{scheme}_thresholds.csv", index=False)
    pd.DataFrame(per_fold).to_csv(out / f"{scheme}_fold_metrics.csv", index=False)
    task_metrics = pd.DataFrame(per_task)
    task_metrics.to_csv(out / f"{scheme}_task_metrics.csv", index=False)
    if importance:
        imp = pd.DataFrame(importance)
        imp.to_csv(out / "grouped_permutation_importance.csv", index=False)
        imp.groupby(["task_name", "group"])[
            ["AP_drop", "AUROC_drop"]
        ].mean().reset_index().to_csv(
            out / "grouped_permutation_importance_by_task.csv", index=False
        )
        imp.groupby("group")[["AP_drop", "AUROC_drop"]].mean().reset_index().to_csv(
            out / "grouped_permutation_importance_summary.csv", index=False
        )
    pooled = [
        {
            "scheme": scheme,
            "feature_set": name,
            **metric(predictions.observed.to_numpy(), predictions[name].to_numpy()),
        }
        for name in FEATURES
    ]
    return pooled, task_metrics


def paired_sensitivity(task_metrics, out):
    rows = []
    for outcome in ["AP", "AUROC"]:
        wide = task_metrics.pivot(
            index="task_name", columns="feature_set", values=outcome
        )
        for baseline in ["QC", "QC_cellcycle"]:
            delta = (wide.biology_QC_cellcycle - wide[baseline]).to_numpy()
            signs = np.array(list(itertools.product([-1, 1], repeat=len(delta))))
            null = (signs * delta).mean(axis=1)
            dataset = np.array([x.split("__")[0] for x in wide.index])
            ds_names = sorted(set(dataset))
            ds_signs = np.array(list(itertools.product([-1, 1], repeat=len(ds_names))))
            ds_null = np.array(
                [
                    np.mean(
                        delta * np.array([sign[ds_names.index(d)] for d in dataset])
                    )
                    for sign in ds_signs
                ]
            )
            rng = np.random.default_rng(SEED)
            boot = np.mean(
                delta[rng.integers(0, len(delta), (10000, len(delta)))], axis=1
            )
            rows.append(
                {
                    "outcome": outcome,
                    "comparison": f"biology_QC_cellcycle minus {baseline}",
                    "n_tasks": len(delta),
                    "n_datasets": len(ds_names),
                    "mean_paired_delta": float(delta.mean()),
                    "minimum_task_delta": float(delta.min()),
                    "maximum_task_delta": float(delta.max()),
                    "task_resampling_lower95_descriptive": float(
                        np.quantile(boot, 0.025)
                    ),
                    "task_resampling_upper95_descriptive": float(
                        np.quantile(boot, 0.975)
                    ),
                    "exact_task_one_sided_P": float(
                        np.mean(null >= delta.mean() - 1e-14)
                    ),
                    "exact_task_two_sided_P": float(
                        np.mean(np.abs(null) >= abs(delta.mean()) - 1e-14)
                    ),
                    "exact_dataset_block_one_sided_P": float(
                        np.mean(ds_null >= delta.mean() - 1e-14)
                    ),
                    "exact_dataset_block_two_sided_P": float(
                        np.mean(np.abs(ds_null) >= abs(delta.mean()) - 1e-14)
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(out / "paired_task_sensitivity.csv", index=False)


def held_line_uncertainty(out):
    """Paired line-block resampling of fixed held-out predictions, without refitting."""
    df = pd.read_csv(
        out / "held_cell_line_McFarland_predictions.csv.gz",
        float_precision="round_trip",
    )
    groups = [g.index.to_numpy() for _, g in df.groupby("cell_line", sort=True)]
    y = df.observed.to_numpy()
    estimates = {k: [] for k in FEATURES}
    rng = np.random.default_rng(SEED)
    for _ in range(1000):
        ix = np.concatenate(
            [groups[j] for j in rng.integers(0, len(groups), len(groups))]
        )
        for name in FEATURES:
            estimates[name].append(metric(y[ix], df[name].to_numpy()[ix]))
    rows = []
    for outcome in ["AP", "AUROC"]:
        for name in FEATURES:
            values = np.array([m[outcome] for m in estimates[name]])
            rows.append(
                {
                    "metric": outcome,
                    "comparison": name,
                    "estimate": metric(y, df[name])[outcome],
                    "lower95": np.quantile(values, 0.025),
                    "upper95": np.quantile(values, 0.975),
                    "n_cell_lines": len(groups),
                    "bootstrap_resamples": 1000,
                }
            )
        for base in ["QC", "QC_cellcycle"]:
            values = np.array(
                [
                    a[outcome] - b[outcome]
                    for a, b in zip(estimates["biology_QC_cellcycle"], estimates[base])
                ]
            )
            rows.append(
                {
                    "metric": outcome,
                    "comparison": "biology_QC_cellcycle minus " + base,
                    "estimate": metric(y, df.biology_QC_cellcycle)[outcome]
                    - metric(y, df[base])[outcome],
                    "lower95": np.quantile(values, 0.025),
                    "upper95": np.quantile(values, 0.975),
                    "n_cell_lines": len(groups),
                    "bootstrap_resamples": 1000,
                }
            )
    pd.DataFrame(rows).to_csv(out / "held_cell_line_uncertainty.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="Directory for analysis results"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Normalized per-cell score table (CSV or gzipped CSV)",
    )
    parser.add_argument(
        "--data-root", type=Path, help="Directory containing the raw source H5AD files"
    )
    parser.add_argument(
        "--source-mapping", type=Path, help="McFarland source-cell mapping CSV"
    )
    parser.add_argument("--rebuild-input", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Also fit and save a model using all eligible cells",
    )
    args = parser.parse_args()
    out = args.output.resolve()
    if args.rebuild_input:
        if args.data_root is None or args.source_mapping is None:
            parser.error("--rebuild-input requires --data-root and --source-mapping")
        if args.input is not None:
            parser.error("Use either --input or --rebuild-input")
        out.mkdir(parents=True, exist_ok=True)
        build_input(args.data_root, out, args.source_mapping)
    elif args.input is not None:
        if not args.input.is_file():
            parser.error(f"Input file does not exist: {args.input}")
        out.mkdir(parents=True, exist_ok=True)
        target = out / "scored_cells.csv.gz"
        if args.input.resolve() != target.resolve():
            if args.input.suffix == ".gz":
                shutil.copy2(args.input, target)
            else:
                import gzip

                with args.input.open("rb") as source, gzip.open(target, "wb") as dest:
                    shutil.copyfileobj(source, dest)
    elif not (out / "scored_cells.csv.gz").is_file():
        parser.error(
            "Supply --input or --rebuild-input, or reuse an output directory containing scored_cells.csv.gz"
        )
    # Reload the portable input so numeric precision is identical on extraction and replay.
    df = pd.read_csv(out / "scored_cells.csv.gz", low_memory=False)
    if args.extract_only:
        return
    # Context levels are fixed source annotations; no outcome-derived encoding.
    context = pd.get_dummies(df.reference_group, prefix="context", dtype=float)
    for column in context:
        df[column] = context[column]
    for name in FEATURES:
        FEATURES[name] = FEATURES[name] + list(context.columns)
    df.groupby(
        ["dataset_id", "source_experiment", "source_time", "task_name", "is_control"],
        dropna=False,
    ).size().rename("n_cells").reset_index().to_csv(
        out / "cohort_counts.csv", index=False
    )
    assert df.sample_id.is_unique
    pooled, tasks = run_cv(df, "leave_task_out", out)
    paired_sensitivity(tasks, out)
    mcf = df[df.dataset_id == "McFarlandTsherniak2020"].reset_index(drop=True)
    extra, _ = run_cv(mcf, "held_cell_line_McFarland", out)
    pd.DataFrame(pooled + extra).to_csv(out / "pooled_metrics.csv", index=False)
    held_line_uncertainty(out)
    if args.save_model:
        labels, cut = thresholds_and_labels(df, np.ones(len(df), dtype=bool))
        final = new_model()
        final.fit(df[FEATURES["biology_QC_cellcycle"]], labels)
        artifact = {
            "pipeline": final,
            "feature_order": FEATURES["biology_QC_cellcycle"],
            "target_thresholds": cut,
            "normalization": "log1p(10000 * raw_count / total_count), mean over present genes",
            "gene_sets": {k: sorted(v) for k, v in GENE_SETS.items()},
            "gene_identity_map": json.loads(GENE_IDENTITY_PATH.read_text()),
            "seed": SEED,
            "context_levels": sorted(df.reference_group.unique()),
            "source_QC": "McFarland normal cells, DMSO-only controls matched by source experiment/time/cell line",
            "purpose": "Contemporaneous DDR-high classification; not cell fate or repair capacity",
        }
        joblib.dump(artifact, out / "final_classifier.joblib", compress=3)
    software = {
        "Python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "h5py": h5py.__version__,
        "joblib": joblib.__version__,
    }
    (out / "software_versions.json").write_text(json.dumps(software, indent=2) + "\n")
    (out / "requirements.txt").write_text(
        "\n".join(f"{k}=={v}" for k, v in software.items() if k != "Python") + "\n"
    )
    if args.save_model:
        model_digest = checksum(out / "final_classifier.joblib")
        (out / "model_sha256.txt").write_text(
            model_digest + "  final_classifier.joblib\n"
        )
    print(pd.DataFrame(pooled + extra).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
