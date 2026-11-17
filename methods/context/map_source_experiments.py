#!/usr/bin/env python3
"""Match AnnData cells to original MIX-Seq experiment metadata, without models."""
from pathlib import Path
import argparse, json, re
import h5py, numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent


def col(group, key):
    x = group[key]
    if isinstance(x, h5py.Group):
        cats = np.array(
            [v.decode() if isinstance(v, bytes) else v for v in x["categories"][:]],
            object,
        )
        return np.array([cats[i] if i >= 0 else None for i in x["codes"][:]], object)
    return np.array([v.decode() if isinstance(v, bytes) else v for v in x[:]], object)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    a = p.parse_args()
    with h5py.File(a.data_root / "datasets/scperturb/McFarlandTsherniak2020.h5ad") as f:
        obs = pd.DataFrame({k: col(f["obs"], k) for k in f["obs"]})
    obs["source_row"] = np.arange(len(obs))
    obs["barcode_adata"] = obs["_index"].astype(str)
    obs["barcode16"] = obs.barcode_adata.str[:16]
    for c in ["num_SNPs", "tot_reads"]:
        obs[c] = pd.to_numeric(obs[c]).astype(int)
    obs["source_condition"] = obs.perturbation
    obs["source_time"] = obs.time.astype(str)
    obs["source_experiment"] = ""
    obs["barcode_original"] = ""
    obs["mapping_basis"] = ""
    unique_batches = {
        "Idasanutlin": "expt1",
        "Bortezomib": "expt1",
        "BRD3379": "expt3",
        "Dabrafenib": "expt3",
        "Navitoclax": "expt3",
        "Gemcitabine": "expt10",
        "Prexasertib": "expt10",
        "Everolimus": "expt10",
        "Afatinib": "expt10",
        "Taselisib": "expt10",
        "AZD5591": "expt10",
        "JQ1": "expt10",
        "sgGPX4-1": "expt2",
        "sgGPX4-2": "expt2",
        "sgLACZ": "expt2",
        "sgOR2J2": "expt2",
    }
    for drug, batch in unique_batches.items():
        m = obs.perturbation == drug
        obs.loc[m, "source_experiment"] = batch
        obs.loc[m, "mapping_basis"] = (
            "unique perturbation/time dataset in original experiment catalog"
        )
    m = (obs.perturbation == "Trametinib") & (obs.time == "6")
    obs.loc[m, "source_experiment"] = "expt1"
    obs.loc[m, "mapping_basis"] = (
        "unique perturbation/time dataset in original experiment catalog"
    )
    m = (obs.perturbation == "Trametinib") & (obs.time == "3, 6, 12, 24, 48")
    obs.loc[m, "source_experiment"] = "expt5"
    obs.loc[m, "mapping_basis"] = (
        "unique source time-course experiment; individual cell time not encoded in harmonized obs.time"
    )
    m = (
        (obs.perturbation == "Trametinib")
        & (obs.time == "24")
        & (~obs.channel.isin(["nan", None]))
    )
    obs.loc[m, "source_experiment"] = "expt10"
    obs.loc[m, "mapping_basis"] = (
        "source experiment catalog plus nonmissing sequencing channel"
    )
    reports = []
    keys = ["barcode16", "singlet_ID", "num_SNPs", "tot_reads"]
    for path in sorted(HERE.glob("*_classifications.csv")):
        src = pd.read_csv(path)
        src["barcode16"] = src.barcode.str[:16]
        for c in ["num_SNPs", "tot_reads"]:
            src[c] = pd.to_numeric(src[c]).astype(int)
        name = path.name.replace("_classifications.csv", "")
        batch = re.search(r"expt\d+", name).group()
        condition = name.split("_")[0]
        time = re.search(r"_(\d+)hr_", name)
        time = time.group(1) if time else "24"
        candidates = obs[
            (
                obs.perturbation
                == ("control" if condition in ["DMSO", "Untreated"] else condition)
            )
            & (obs.time == time)
        ]
        if src.duplicated(keys).any():
            raise ValueError("Ambiguous original barcode/SNP key")
        match = candidates.merge(
            src,
            how="inner",
            on=keys,
            suffixes=("_adata", "_source"),
            validate="many_to_one",
        )
        if len(match) != len(src) or match.source_row.duplicated().any():
            raise ValueError(
                f"Incomplete source mapping: {name}: source {len(src)}, matched {len(match)}"
            )
        if not (match.cell_quality_adata == match.cell_quality_source).all():
            raise ValueError("Quality mismatch")
        if not (match.DepMap_ID_adata == match.DepMap_ID_source).all():
            raise ValueError("Cell identity mismatch")
        # Continuous independent fingerprint verifies the integer key match.
        err = np.max(
            np.abs(
                pd.to_numeric(match.singlet_dev_adata)
                - pd.to_numeric(match.singlet_dev_source)
            )
        )
        if err > 1e-10:
            raise ValueError("SNP-model fingerprint mismatch")
        rows = match.source_row.to_numpy()
        if (
            (obs.loc[rows, "source_experiment"] != "")
            & (obs.loc[rows, "source_experiment"] != batch)
        ).any():
            raise ValueError("Experiment conflict")
        obs.loc[rows, "source_experiment"] = batch
        obs.loc[rows, "source_condition"] = condition
        obs.loc[rows, "source_time"] = time
        obs.loc[rows, "barcode_original"] = match.barcode.to_numpy()
        obs.loc[rows, "mapping_basis"] = (
            "original barcode16 + cell identity + SNP count + SNP read count; continuous SNP-deviance and QC verified"
        )
        reports.append(
            dict(
                source_dataset=name,
                n_original=len(src),
                n_matched=len(match),
                max_abs_singlet_dev_error=float(err),
                n_normal=int((src.cell_quality == "normal").sum()),
                n_normal_lines=int(
                    src.loc[src.cell_quality == "normal", "singlet_ID"].nunique()
                ),
            )
        )
    if (obs.source_experiment == "").any():
        raise ValueError(
            f'Unmapped rows: {obs.loc[obs.source_experiment=="",["perturbation","time","channel"]].value_counts().to_dict()}'
        )
    keep = [
        "source_row",
        "source_experiment",
        "source_condition",
        "source_time",
        "cell_quality",
        "cell_line",
        "singlet_ID",
        "DepMap_ID",
        "barcode_adata",
        "barcode_original",
        "mapping_basis",
    ]
    obs[keep].to_csv(HERE / "source_cell_mapping.csv.gz", index=False)
    pd.DataFrame(reports).to_csv(HERE / "source_mapping_verification.csv", index=False)
    obs.groupby(
        ["source_experiment", "source_condition", "source_time", "cell_quality"]
    ).size().rename("n_cells").reset_index().to_csv(
        HERE / "source_cohort_counts.csv", index=False
    )
    print(
        "All",
        len(obs),
        "rows mapped; controls",
        sum(obs.perturbation == "control"),
        "uniquely fingerprint-matched.",
    )


if __name__ == "__main__":
    main()
