#!/usr/bin/env python3
"""Compare measured DNA-content gates within perturbation and source replicate."""
from pathlib import Path
import argparse
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--scores",
        type=Path,
        default=HERE.parent / "external/external_scored_cells.csv.gz",
    )
    args = p.parse_args()
    cells = pd.read_csv(args.scores)
    cells = cells[cells.dataset_id == "GSE255874"]
    rows = []
    for perturbation in ["HRAS", "Pevo"]:
        for rep in ["rep1", "rep2"]:
            g = cells[cells.experiment == rep]
            lower = g[g.condition == perturbation + "_2N4N"]
            poly = g[g.condition == perturbation + "_Polyploid"]
            if lower.empty or poly.empty:
                raise ValueError("Missing perturbation/gate/replicate")
            rows.append(
                dict(
                    dataset_id="GSE255874",
                    perturbation=perturbation,
                    replicate=rep,
                    sample_2N4N=lower.sample_id.iloc[0],
                    sample_polyploid=poly.sample_id.iloc[0],
                    n_2N4N=len(lower),
                    n_polyploid=len(poly),
                    mean_score_2N4N=lower.ddr_composite.mean(),
                    mean_score_polyploid=poly.ddr_composite.mean(),
                    delta_mean_score_polyploid_minus_2N4N=poly.ddr_composite.mean()
                    - lower.ddr_composite.mean(),
                    fraction_z2_2N4N=(lower.ddr_composite_z > 2).mean(),
                    fraction_z2_polyploid=(poly.ddr_composite_z > 2).mean(),
                    delta_fraction_z2_polyploid_minus_2N4N=(
                        poly.ddr_composite_z > 2
                    ).mean()
                    - (lower.ddr_composite_z > 2).mean(),
                    reference="fixed pooled EV2N4N control median/MAD; shared by all source samples",
                    interpretation="within-perturbation DNA-content-gate contrast; descriptive two-replicate context, not a DNA-lesion or repair assay",
                )
            )
    pd.DataFrame(rows).to_csv(
        HERE / "within_perturbation_ploidy_contrasts.csv", index=False
    )
    print(
        pd.DataFrame(rows)[
            [
                "perturbation",
                "replicate",
                "n_2N4N",
                "n_polyploid",
                "delta_mean_score_polyploid_minus_2N4N",
                "delta_fraction_z2_polyploid_minus_2N4N",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
