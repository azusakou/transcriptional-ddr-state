#!/usr/bin/env python3
"""Recompute published statistics/figures or refit the grouped models."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile, urllib.request, zipfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
METHODS = ROOT / "methods"
sys.path.insert(0, str(METHODS))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ddr_matplotlib")
)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unpack(out):
    archive = ROOT / "data/processed.zip"
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    if digest(archive) != manifest["archive_sha256"]:
        raise ValueError("Evidence archive checksum mismatch")
    if "model_artifact" in manifest:
        model = manifest["model_artifact"]
        if digest(ROOT / "data" / model["path"]) != model["sha256"]:
            raise ValueError("Fitted classifier artifact checksum mismatch")
    target = out / "inputs"
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            path = Path(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe archive path")
        z.extractall(target)
    for record in manifest["files"]:
        if digest(target / record["path"]) != record["sha256"]:
            raise ValueError("Input checksum mismatch: " + record["path"])
    return target


def run(script, *args):
    subprocess.run([sys.executable, str(METHODS / script), *map(str, args)], check=True)


def forecast_statistics(folder, bootstrap):
    import numpy as np
    import pandas as pd
    import source_matched_forecasting as s
    import nested_forecasting as core

    pred = pd.read_csv(folder / "nested_predictions.csv")
    s.performance(pred).to_csv(folder / "performance.csv", index=False)
    main = pred.query(
        "analysis=='composite_z2' and cohort=='time_specific' and split=='frozen_outer'"
    ).copy()
    assert not main.duplicated(["cell_line", "regimen", "model"]).any()
    assert main.groupby("cell_line").fold.nunique().max() == 1
    assert main.predicted.between(0, 1).all()
    for _, g in main.groupby("model"):
        assert len(g) == 243 and g.cell_line.nunique() == 84
        assert np.allclose(g.observed * g.treated_n_cells, g.n_high)
    ci = core.intervals(main, bootstrap, unit_columns=["cell_line", "regimen"])
    ci["analysis"] = "composite_z2"
    ci["cohort"] = "time_specific"
    ci.to_csv(folder / "line_bootstrap.csv", index=False)
    s.calibration(pred).to_csv(folder / "calibration_bins.csv", index=False)
    stats = core.metrics(
        main[main.model == "parsimonious_additive"].observed.to_numpy(),
        main[main.model == "parsimonious_additive"].predicted.to_numpy(),
    )
    if abs(stats["r2"] - 0.505826051227275) > 1e-8:
        raise ValueError("Primary published R2 did not reproduce")
    if abs(stats["mae"] - 0.122645864999196) > 1e-8:
        raise ValueError("Primary published MAE did not reproduce")
    return stats


def classifier_statistics(folder):
    import pandas as pd
    import classifier as c

    pooled = []
    for scheme in ["leave_task_out", "held_cell_line_McFarland"]:
        p = pd.read_csv(
            folder / f"{scheme}_predictions.csv.gz", float_precision="round_trip"
        )
        fold = pd.read_csv(
            folder / f"{scheme}_fold_membership.csv.gz", dtype={"test_fold": str}
        )
        assert not p.sample_id.duplicated().any()
        p = p.merge(
            fold[["sample_id", "test_fold"]], on="sample_id", validate="one_to_one"
        )
        tasks = []
        folds = []
        for key in c.FEATURES:
            pooled.append(
                {
                    "scheme": scheme,
                    "feature_set": key,
                    **c.metric(p.observed.to_numpy(), p[key].to_numpy()),
                }
            )
            for f, g in p.groupby("test_fold"):
                folds.append(
                    {
                        "scheme": scheme,
                        "fold": f,
                        "feature_set": key,
                        **c.metric(g.observed.to_numpy(), g[key].to_numpy()),
                    }
                )
                for task, t in g.groupby("task_name"):
                    tasks.append(
                        {
                            "scheme": scheme,
                            "fold": f,
                            "task_name": task,
                            "feature_set": key,
                            **c.metric(t.observed.to_numpy(), t[key].to_numpy()),
                        }
                    )
        pd.DataFrame(tasks).to_csv(folder / f"{scheme}_task_metrics.csv", index=False)
        pd.DataFrame(folds).to_csv(folder / f"{scheme}_fold_metrics.csv", index=False)
    pd.DataFrame(pooled).to_csv(folder / "pooled_metrics.csv", index=False)
    return pooled


def fit_forecast(inputs, out, args):
    import numpy as np
    import pandas as pd
    import nested_forecasting as core
    import source_matched_forecasting as s

    d = inputs / "forecast"
    common = [
        "--folds",
        d / "frozen_line_folds.csv",
        "--definitions",
        d / "gene_sets.json",
        "--canonical",
        d / "canonical_gene_sets.json",
        "--output",
        out,
        "--bootstrap",
        args.bootstrap,
    ]
    if args.all_analyses:
        run("source_matched_forecasting.py", "--processed", d, *common)
        return
    a = SimpleNamespace(
        processed=d,
        input=None,
        mapping=None,
        definitions=d / "gene_sets.json",
        canonical=d / "canonical_gene_sets.json",
    )
    obs, scores, modules, mapping = s.load_inputs(a)
    pairs, controls, treated = s.cohort(obs, mapping, True)
    frozen = (
        pd.read_csv(d / "frozen_line_folds.csv").set_index("cell_line").fold.to_dict()
    )
    allocation = pairs.cell_line.map(frozen).to_numpy(int)
    p, t, params, refs = core.run_nested(
        scores,
        pairs,
        controls,
        treated,
        allocation,
        "frozen_outer",
        transformer=s.transformer_for(modules, "ddr_composite", 2, False),
        regimes=list(dict.fromkeys(pairs.drug)),
        models=s.MAIN_MODELS,
    )
    p = p.rename(columns={"drug": "regimen", "compound": "drug"})
    p["analysis"] = "composite_z2"
    p["cohort"] = "time_specific"
    for collection in [t, params, refs]:
        for entry in collection:
            entry.update(analysis="composite_z2", cohort="time_specific")
    keys = ["cell_line", "regimen", "model"]
    reference = (
        pd.read_csv(d / "nested_predictions.csv")
        .query(
            "analysis=='composite_z2' and cohort=='time_specific' and split=='frozen_outer'"
        )
        .set_index(keys)
    )
    matched = p.set_index(keys).loc[reference.index]
    error = float(np.max(np.abs(matched.predicted - reference.predicted)))
    if error > 1e-7:
        raise ValueError(f"Refit differs from published predictions: {error}")
    p.to_csv(out / "nested_predictions.csv", index=False)
    pd.DataFrame(t).to_csv(out / "inner_tuning.csv", index=False)
    (out / "model_parameters.json").write_text(json.dumps(params, indent=2) + "\n")
    (out / "fold_references.json").write_text(json.dumps(refs, indent=2) + "\n")
    print(
        f"Primary nested refit: maximum prediction difference {error:.3g}", flush=True
    )


def download_inputs(inputs, args):
    if args.data_root is None:
        raise ValueError("--download-inputs requires --data-root")
    records = json.loads((inputs / "classifier/source_files.json").read_text())
    import pandas as pd

    ext = pd.read_csv(inputs / "external/source_file_manifest.csv")
    for dataset in args.download_inputs:
        target = args.data_root / "datasets/scperturb" / f"{dataset}.h5ad"
        target.parent.mkdir(parents=True, exist_ok=True)
        record = next((r for r in records if r["dataset_id"] == dataset), None)
        if record is None:
            r = ext[ext.relative_path.str.endswith("/" + target.name)].iloc[0]
            record = {"bytes": int(r.bytes), "sha256": r.sha256}
        url = f"https://zenodo.org/api/records/7041849/files/{target.name}/content"
        if target.exists() and digest(target) == record["sha256"]:
            continue
        temporary = target.with_suffix(".download")
        print("Downloading public input:", dataset, flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, open(
            temporary, "wb"
        ) as f:
            shutil.copyfileobj(response, f)
        if (
            temporary.stat().st_size != record["bytes"]
            or digest(temporary) != record["sha256"]
        ):
            raise ValueError("Downloaded source failed checksum")
        temporary.replace(target)


def download_direct(inputs, args):
    if args.data_root is None:
        raise ValueError("--download-direct requires --data-root")
    for kind in args.download_direct:
        records = json.loads(
            (inputs / f"direct_{kind}/source_manifest.json").read_text()
        )
        destination = args.data_root / "datasets" / f"direct_{kind}"
        destination.mkdir(parents=True, exist_ok=True)
        for item in records:
            name = item.get("file", item.get("filename"))
            if kind == "damage" and name not in [
                "GSE250285_counts_rpe_dox_gp.tsv.gz",
                "GSE250285_metadata_rpe_dox_gp.tsv.gz",
            ]:
                continue
            url = item.get("url", item.get("source_url"))
            target = destination / name
            if target.exists() and digest(target) == item["sha256"]:
                continue
            temporary = target.with_suffix(target.suffix + ".download")
            print("Downloading public direct-validation input:", name, flush=True)
            request = urllib.request.Request(
                url, headers={"User-Agent": "GEO-reanalysis/1.0"}
            )
            with urllib.request.urlopen(request, timeout=120) as response, open(
                temporary, "wb"
            ) as f:
                shutil.copyfileobj(response, f)
            if digest(temporary) != item["sha256"]:
                raise ValueError(
                    "Downloaded direct-validation input failed checksum: " + name
                )
            temporary.replace(target)


def direct_statistics(inputs, results, args):
    for kind, name in [
        ("damage", "all_scored_cells.csv.gz"),
        ("repair", "paired_cells.csv.gz"),
    ]:
        folder = inputs / f"direct_{kind}"
        if folder.exists() and args.raw != kind:
            run(
                f"direct_{kind}.py",
                "--input",
                folder / name,
                "--output",
                results / f"direct_{kind}",
            )
        if folder.exists():
            import pandas as pd

            primary = (
                "primary_plate_summary.csv"
                if kind == "damage"
                else "primary_results.csv"
            )
            reference = pd.read_csv(folder / primary, float_precision="round_trip")
            actual = pd.read_csv(
                results / f"direct_{kind}" / primary, float_precision="round_trip"
            )
            pd.testing.assert_frame_equal(
                reference,
                actual,
                check_dtype=False,
                check_exact=False,
                atol=1e-12,
                rtol=1e-12,
            )
            print(
                f"Direct {kind}: published primary table reproduced ({len(actual)} rows)",
                flush=True,
            )


def raw_analysis(inputs, results, args):
    if args.data_root is None:
        raise ValueError("--raw requires --data-root")
    data = args.data_root.resolve()
    fd = inputs / "forecast"
    mapping = inputs / "source_context/source_cell_mapping.csv.gz"
    if args.raw == "forecast":
        run(
            "source_matched_forecasting.py",
            "--input",
            data / "datasets/scperturb/McFarlandTsherniak2020.h5ad",
            "--mapping",
            mapping,
            "--folds",
            fd / "frozen_line_folds.csv",
            "--definitions",
            fd / "gene_sets.json",
            "--canonical",
            fd / "canonical_gene_sets.json",
            "--output",
            results / "forecast",
            "--bootstrap",
            args.bootstrap,
            *([] if args.all_analyses else ["--skip-sensitivities"]),
        )
    elif args.raw == "classifier":
        run(
            "classifier.py",
            "--rebuild-input",
            "--data-root",
            data / "datasets/scperturb",
            "--source-mapping",
            mapping,
            "--output",
            results / "classifier",
            *(["--save-model"] if args.save_classifier else []),
        )
    elif args.raw == "external":
        run("external.py", "--data-root", data, "--out", results / "external")
    elif args.raw == "genetic":
        run(
            "genetic.py",
            "--input",
            data / "datasets/scperturb/ReplogleWeissman2022_rpe1.h5ad",
            "--output",
            results / "genetic",
        )
    elif args.raw == "damage":
        run(
            "direct_damage.py",
            "--source",
            data / "datasets/direct_damage",
            "--output",
            results / "direct_damage",
        )
    elif args.raw == "repair":
        run(
            "direct_repair.py",
            "--data-root",
            data / "datasets/direct_repair",
            "--annotations",
            data / "datasets/direct_repair/figure6a_scrna_umap_plot.csv",
            "--output",
            results / "direct_repair",
        )
    elif args.raw in ["context", "source-mapping"]:
        folder = results / "source_context"
        for f in (METHODS / "context").iterdir():
            if f.suffix in [".py", ".R", ".json"]:
                shutil.copy2(f, folder / f.name)
        if args.raw == "source-mapping":
            subprocess.run(
                [sys.executable, str(folder / "fetch_source_classifications.py")],
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(folder / "map_source_experiments.py"),
                    "--data-root",
                    str(data),
                ],
                check=True,
            )
        else:
            subprocess.run(
                [
                    sys.executable,
                    str(folder / "reanalyse_functional_context.py"),
                    "--data-root",
                    str(data),
                ],
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(folder / "corroborate_frozen_predictions.py"),
                    "--predictions",
                    str(results / "forecast/nested_predictions.csv"),
                ],
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(folder / "compare_ploidy_gates.py"),
                    "--scores",
                    str(results / "external/external_scored_cells.csv.gz"),
                ],
                check=True,
            )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "results")
    p.add_argument("--bootstrap", type=int, default=4000)
    p.add_argument("--refit", choices=["forecast", "classifier"])
    p.add_argument(
        "--all-analyses",
        action="store_true",
        help="Refit all predefined forecast endpoints/cohorts/splits",
    )
    p.add_argument(
        "--permutation",
        action="store_true",
        help="Recompute 999 conditional line-block permutations",
    )
    p.add_argument(
        "--save-classifier",
        action="store_true",
        help="Also refit/export the full-data RF artifact",
    )
    p.add_argument(
        "--raw",
        choices=[
            "forecast",
            "classifier",
            "external",
            "genetic",
            "context",
            "source-mapping",
            "damage",
            "repair",
        ],
    )
    p.add_argument(
        "--data-root",
        type=Path,
        help="Root containing datasets/scperturb and other manifest paths",
    )
    p.add_argument(
        "--download-inputs",
        nargs="+",
        choices=[
            "McFarlandTsherniak2020",
            "SrivatsanTrapnell2020_sciplex2",
            "ReplogleWeissman2022_rpe1",
        ],
    )
    p.add_argument("--download-direct", nargs="+", choices=["damage", "repair"])
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    inputs = unpack(out)
    results = out / "analysis"
    shutil.copytree(inputs, results, dirs_exist_ok=True)
    if args.download_inputs:
        download_inputs(inputs, args)
    if args.download_direct:
        download_direct(inputs, args)
    if args.raw:
        raw_analysis(inputs, results, args)
    if args.refit == "forecast":
        fit_forecast(inputs, results / "forecast", args)
    elif args.refit == "classifier":
        run(
            "classifier.py",
            "--input",
            inputs / "classifier/scored_cells.csv.gz",
            "--output",
            results / "classifier",
            *(["--save-model"] if args.save_classifier else []),
        )
    if args.permutation:
        fd = inputs / "forecast"
        run(
            "source_matched_permutation.py",
            "--processed",
            fd,
            "--folds",
            fd / "frozen_line_folds.csv",
            "--definitions",
            fd / "gene_sets.json",
            "--canonical",
            fd / "canonical_gene_sets.json",
            "--output",
            results / "forecast",
        )
    direct_statistics(inputs, results, args)
    primary = forecast_statistics(results / "forecast", args.bootstrap)
    secondary = classifier_statistics(results / "classifier")
    summary = {
        "primary_forecast": primary,
        "classifier_pooled": secondary,
        "bootstrap_draws": args.bootstrap,
        "mode": (
            "refit"
            if args.refit or args.raw
            else "statistics/figures from archived held-out predictions; no model refitting"
        ),
    }
    (out / "reproduced_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not args.no_figures:
        run("figures.py", "--data", results, "--output", out / "figures")
        if (METHODS / "direct_figure.py").exists():
            run("direct_figure.py", "--data", results, "--output", out / "figures")
        if (METHODS / "direct_figure.py").exists():
            run("direct_figure.py", "--data", results, "--output", out / "figures")
    print(
        f"Verified primary R2={primary['r2']:.9f}; MAE={primary['mae']:.9f}. Outputs: {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
