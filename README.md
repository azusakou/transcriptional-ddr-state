# Transcriptional DDR-state analysis

Reproducible analysis of a transcriptional DNA-damage-response (DDR) state in public single-cell perturbation datasets. The main analysis predicts a treated cell population's DDR-high fraction from an experiment- and collection-time-matched DMSO population and treatment regimen. These controls are parallel vehicle cultures, not measured pretreatment states of the treated cells.

The repository contains one entry point, the analysis implementations, and a compact compressed evidence package. Large source expression matrices are downloaded separately. Primary forecast model parameters are provided in JSON; the fitted secondary random forest is supplied separately as a losslessly compressed 13 MB joblib artifact.

## Run

Use Python 3.12 and the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py --output results
```

The default command verifies every packaged input checksum, recalculates forecast metrics and 4,000 whole-cell-line bootstrap intervals from the archived held-out predictions, recalculates classifier AUROC/average precision, recalculates the two direct same-cell validation analyses, and renders the four main figures and Supplementary Figure S1 as vector PDFs. It does not refit models. Results are written only to the requested output directory. The primary analysis should recover **R² = 0.4925950995** and **MAE = 0.1248396179** across 243 cell-line × drug × time pairs from 84 cell lines.

## Refit from processed inputs

```bash
# Refit all primary forecast comparators with the frozen outer folds and nested tuning.
python reproduce.py --output forecast_refit --refit forecast

# Also fit every predefined signature, cutoff, source-reference, cohort and split sensitivity.
python reproduce.py --output forecast_all --refit forecast --all-analyses

# Recompute the 999 conditional line-block permutations; may be combined with either refit.
python reproduce.py --output forecast_permutation --permutation

# Refit the cell-state classifiers from the portable normalized feature table.
python reproduce.py --output classifier_refit --refit classifier

# Optionally export the full-data classifier fit in addition to cross-validation.
python reproduce.py --output classifier_refit --refit classifier --save-classifier
```

The primary forecast refit checks every prediction against the archived result, with a tolerance of 10⁻⁷. Small differences across numerical-library platforms are possible. The fitted random forest in `data/final_classifier.joblib` uses all eligible classifier cells and is separate from the held-out performance estimates. The artifact includes its exact feature order, imputer/scaler, 250 fitted trees, gene definitions, reference-group levels and full-data target cutoffs. Its lossless recompression was checked against the source artifact: all tree/preprocessing arrays and predictions for all 35,190 eligible cells are identical. The command above regenerates the full-data fit; serialization/compression may produce a different file checksum.

## Rebuild from public source matrices

The packaged `source_context` mapping retains the original zero-based H5AD observation row, source experiment/time, cell quality and source cell-line identifiers. Forecast processed inputs additionally preserve each original row ID. The source checksums, URLs, gene memberships, normalization and exact cohort criteria are recorded in the packaged manifests.

```bash
# Explicit optional download: approximately 1.6 GB for the two predictive resources.
python reproduce.py --output source_inputs --data-root raw_data \
  --download-inputs McFarlandTsherniak2020 SrivatsanTrapnell2020_sciplex2 --no-figures

python reproduce.py --output forecast_raw --data-root raw_data --raw forecast --all-analyses
python reproduce.py --output classifier_raw --data-root raw_data --raw classifier

# Reconstruct the source-experiment mapping from original public classification metadata.
python reproduce.py --output mapping_raw --data-root raw_data --raw source-mapping
```

Downloaded H5AD files are placed under `raw_data/datasets/scperturb/`. Source mapping reconstruction retrieves only the classification members of the original Figshare archives by HTTP byte ranges and checks the exact cell fingerprints. The included mapping is sufficient for the other raw analyses, so this reconstruction is optional.

The independent descriptive analyses can also be rerun from their public inputs:

```bash
python reproduce.py --output external_raw --data-root raw_data --raw external
python reproduce.py --output context_raw --data-root raw_data --raw context
python reproduce.py --output genetic_raw --data-root raw_data --raw genetic
```

For these commands, obtain the relevant inputs at the relative paths listed in `results/inputs/external/source_file_manifest.csv` after any default run. That table and `source_url_map.json` specify the GEO, Zenodo and original imaging records and checksums. `--raw context` uses the McFarland matrix and the included publisher-derived annotation tables. `--raw genetic` uses `datasets/scperturb/ReplogleWeissman2022_rpe1.h5ad`; it can be downloaded explicitly with `--download-inputs ReplogleWeissman2022_rpe1`. The original annotation retrieval and RDS-to-CSV conversion code is in `methods/context/`; the small original-file URLs and hashes are included in its source manifests. Raw analyses require more memory and disk space than the default replay.

## Same-cell damage and functional-repair datasets

The default command also reruns both direct endpoint analyses from compact cell-level tables. Their original 34-gene DDR signature is frozen: genes and weights are not selected or tuned against either physical endpoint. These data are separate from the forecast/classifier fitting resources.

- **GSE250285:** exact cell-ID links join RNA and quantitative gammaH2AX fluorescence. The original supplied transcript counts are normalized over endogenous transcripts, with ERCC spike-ins excluded; 32 of the 34 DDR genes are present. Published QC retains 1,868 of 2,998 cells. Primary associations adjust DAPI and RNA depth/complexity within six treated sorting plates, with a cell-cycle-proxy sensitivity. Conditional within-plate bootstrap intervals do not establish independent culture replication. Source gate-category labels are reversed relative to quantitative fluorescence, so the analysis uses the quantitative endpoint and records the category audit.
- **GSE330132:** 18,024 source-QC cell barcodes link RNA to eight host-cell-reactivation reporter endpoints at two collection times. Original source barcode/clone assignments and all reporter arithmetic are checked. Associations are estimated within cell lines after the specified covariate adjustment and summarized across cell lines; the exact endpoint definitions and denominator criteria are preserved. Reporter readouts are pathway-specific functional measurements and are distinct from gammaH2AX damage signalling.

To rebuild their scores from the original public files:

```bash
# Damage source counts + phenotype metadata: approximately 12 MB.
python reproduce.py --output damage_download --data-root raw_data --download-direct damage --no-figures
python reproduce.py --output damage_raw --data-root raw_data --raw damage

# Functional-repair source matrices, H5AD annotations and pinned author labels: approximately 1.33 GB.
python reproduce.py --output repair_download --data-root raw_data --download-direct repair --no-figures
python reproduce.py --output repair_raw --data-root raw_data --raw repair
```

The downloader verifies source SHA-256 hashes. Files are placed in `raw_data/datasets/direct_damage/` or `raw_data/datasets/direct_repair/`. The complete raw scoring and endpoint-reconstruction implementations are `methods/direct_damage.py` and `methods/direct_repair.py`; they do not import upstream assay software. Compact-input replay uses the same statistical implementation as each raw route. Exact URLs, fixed source commit, checksums, gene coverage, endpoint definitions, study citations and source-design limits are included in the respective evidence subdirectories.

## Scientific records

`data/processed.zip` contains only the processed scientific inputs and records needed to audit or regenerate this analysis. Its per-file hashes and the separate fitted RF artifact hash are listed in `data/manifest.json`.

- `forecast/`: normalized module scores for 23,704 source-normal candidate cells, cell/source metadata, exact gene sets, frozen cell-line folds, held-out predictions for every predefined analysis, inner tuning losses, training-control references, fitted coefficients and conditional permutation results. Primary eligibility requires at least 20 matched DMSO and 10 treated cells per pair.
- `classifier/`: normalized predictor/target scores, exclusive task assignments, held-out predictions, fold identities, training-control cutoffs and grouped permutation results. Target DDR genes are removed from biological predictor modules. Task and held-cell-line tests are reported separately.
- `source_context/`: original-source matching checks, source mutation/PRISM annotations and the descriptive functional comparisons.
- `direct_damage/` and `direct_repair/`: compact paired-cell input scores, original-source provenance and the direct endpoint analyses described above.
- `external/` and `genetic/`: compact scored/aggregated inputs for independent expression, live-to-endpoint imaging, DNA-content-gate and genetic-perturbation comparisons, with source provenance. Large, regenerable per-target/batch intermediates are omitted.

`methods/nested_forecasting.py` implements fractional-logit optimization, design matrices, inner grouped selection and line bootstrap; `source_matched_forecasting.py` implements source/QC matching and all fold-specific biological references. Primary biological covariates are the baseline composite-high fraction and p53, cell-cycle and stress module means, with unpenalized regimen intercepts. Regularization is selected only in inner grouped folds from the fixed candidates 0.3, 3 and 30. Every inner/outer threshold and feature transformation uses training controls/rows only. The primary strategy and candidate penalties are fixed in the supplied analysis specification; grouped cross-validation is distinct from independent prospective validation.

Bootstrap intervals resample complete cell lines with held-out predictions fixed; they do not include refitting or model-selection uncertainty. The conditional permutation preserves regimen-availability patterns and train/test partitions and fixes the selected regularization strengths. Alternative signatures and cutoffs can have different event support and are reported separately. The response is a transcriptional state, not a direct measurement of DNA lesions, repair capacity or cell survival.

## Sources and attribution

The expression resources originate from the study authors and the [scPerturb collection](https://zenodo.org/records/7041849). Exact study/record links and checksums are supplied with the processed inputs. The independent Hallmark signatures use the official [MSigDB human 2026.2 release](https://docs.gsea-msigdb.org/MSigDB/Release_Notes/MSigDB_2026.2.Hs/) and the [Hallmark collection publication](https://doi.org/10.1016/j.cels.2015.12.004); exact membership and observed-gene coverage are preserved in `forecast/canonical_gene_sets.json`. The scPerturb and imaging Zenodo records and the original MIX-seq Figshare annotation record specify CC BY 4.0; MSigDB specifies CC BY 4.0 for the selected Hallmark release. Machine-readable source-record license metadata and attribution are preserved in `upstream_sources.json`. GEO deposits are public, but a separate dataset-specific license is not inferred from an article's open-access license. The HCR-seq source software has an academic noncommercial license; that software is neither imported nor redistributed here. Source data retain their providers' terms and attribution requirements. No blanket software or third-party-data license is granted by this repository.
