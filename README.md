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

The default command verifies all packaged input checksums, recomputes forecast metrics and 4,000 whole-cell-line bootstrap intervals, reconstructs all saved fold coefficients, rebuilds the TP53 annotation, recomputes classifier AUROC/average precision and the 1,000-resample held-line intervals and every paired gammaH2AX, functional-repair and physical-lesion analysis, including their specified sensitivities. It renders four main and three supplementary figures, with vector PDF/SVG, PNG previews and source CSVs. It does not refit models. Results are written only to the requested output directory. The four main figures use a shared vector layout with uppercase panel labels and editable Arial text; supplementary figures retain their existing layouts.

The primary biology-informed model should recover **R² = 0.623163** and **MAE = 0.103320** across 243 pairs. Its 95% interval is 0.506239–0.706707; the paired gain over the original model is 0.117337 (0.043665–0.190968). The original four-state model remains **0.505826**. Known TP53 and metabolism alone reach 0.572688, and the four-state increment is uncertain (0.050475, −0.011599–0.122406). This changes the interpretation: the genetics/pharmacology explain much of the gain. The inner selector is 0.602347, with a separate MSE sensitivity. These extensions were developed in the existing resource; the fixed outer predictions and bootstrap intervals do not constitute independent prospective validation.

## Refit from processed inputs

```bash
# Refit the final biological model and its matching controls from compact inputs.
python reproduce.py --output extension_refit --refit extension

# Refit the original forecast comparators with the frozen outer folds and nested tuning.
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

The original forecast refit checks every prediction against the archived result, with a tolerance of 10⁻⁷. Small differences across numerical-library platforms are possible. The fitted random forest in `data/final_classifier.joblib` uses all eligible classifier cells and is separate from the held-out performance estimates. The artifact includes its exact feature order, imputer/scaler, 250 fitted trees, gene definitions, reference-group levels and full-data target cutoffs. Its lossless recompression was checked against the source artifact: all tree/preprocessing arrays and predictions for all 35,190 eligible cells are identical. The command above regenerates the full-data fit; serialization/compression may produce a different file checksum.

## Rebuild from public source matrices

The packaged `source_context` mapping retains the original zero-based H5AD observation row, source experiment/time, cell quality and source cell-line identifiers. Forecast processed inputs additionally preserve each original row ID. The source checksums, URLs, gene memberships, normalization and exact cohort criteria are recorded in the packaged manifests.

```bash
# Explicit optional download: approximately 1.6 GB for the two predictive resources.
python reproduce.py --output source_inputs --data-root raw_data \
  --download-inputs McFarlandTsherniak2020 SrivatsanTrapnell2020_sciplex2 --no-figures

# Rebuild original forecast scores from the source matrix.
python reproduce.py --output forecast_raw --data-root raw_data --raw forecast --all-analyses

# Re-extract the five pharmacology genes and refit the final biological models.
python reproduce.py --output extension_raw --data-root raw_data --raw extension
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

The default command reruns three direct endpoint analyses from compact cell-level tables. Their original 34-gene DDR signature is frozen: genes and weights are not selected or tuned against either physical endpoint. These data are separate from the forecast/classifier fitting resources. Their endpoints and experimental units are interpreted separately; successful transcriptional prediction is not itself validation of damage or repair capacity.

- **GSE250285:** exact cell-ID links join RNA and quantitative gammaH2AX fluorescence. The original supplied transcript counts are normalized over endogenous transcripts, with ERCC spike-ins excluded; 33 of the 34 DDR genes are present. Published QC retains 1,868 of 2,998 cells. Primary associations adjust DAPI and RNA depth/complexity within six treated sorting plates, with a cell-cycle-proxy sensitivity. Conditional within-plate bootstrap intervals do not establish independent culture replication. Source gate-category labels are reversed relative to quantitative fluorescence, so the analysis uses the quantitative endpoint and records the category audit.
- **GSE330132:** 18,024 source-QC cell barcodes link RNA to eight host-cell-reactivation reporter endpoints at two collection times. Original source barcode/clone assignments and all reporter arithmetic are checked. Associations are estimated within cell lines after the specified covariate adjustment and summarized across cell lines; the exact endpoint definitions and denominator criteria are preserved. Reporter readouts are pathway-specific functional measurements and are distinct from gammaH2AX damage signalling.

- **GSE268554 (Paired-Damage-seq):** 64,794 exact paired cell barcodes link RNA to physical lesion-region fragments across control and five recovery times. The default reruns all six conditions, original capture partitions, fixed p53/repair subsets and depth/read-matching sensitivities. These measurements describe relative lesion localization, not absolute lesions per cell or independent culture replication. The fixed score has negligible QC/cycle-adjusted correlations with relative induced-region signal across all six conditions (ρ −0.0033 to 0.0153); depth-matched sensitivities agree. These data do not establish absolute lesion burden or repair rates.

To rebuild their scores from the original public files:

```bash
# Paired physical-lesion source files: approximately 4.3 GB, optional.
python reproduce.py --output lesions_download --data-root raw_data --download-direct lesions --no-figures
python reproduce.py --output lesions_raw --data-root raw_data --raw lesions

# Damage source counts + phenotype metadata: approximately 12 MB.
python reproduce.py --output damage_download --data-root raw_data --download-direct damage --no-figures
python reproduce.py --output damage_raw --data-root raw_data --raw damage

# Functional-repair source matrices, H5AD annotations and pinned author labels: approximately 1.33 GB.
python reproduce.py --output repair_download --data-root raw_data --download-direct repair --no-figures
python reproduce.py --output repair_raw --data-root raw_data --raw repair
```

The downloader verifies source SHA-256 hashes. Files are placed in `raw_data/datasets/direct_damage/` or `raw_data/datasets/direct_repair/` or `raw_data/datasets/direct_lesions/`. The complete raw scoring and endpoint-reconstruction implementations are `methods/direct_damage.py`, `methods/direct_repair.py` and `methods/direct_lesions/`; they do not import upstream assay software. Compact-input replay uses the same statistical implementation as each raw route. Exact URLs, fixed source commit, checksums, gene coverage, endpoint definitions, study citations and source-design limits are included in the respective evidence subdirectories.

The uniform CCLE2019 TP53 input uses exact original sample IDs and mutation-profile coverage for all 84 lines. The fixed protein-altering/canonical-splice rule gives 57 calls. Literature-confirmed splice disruption by synonymous c.375G>A in NCI-H2347 gives 58; the raw-call sensitivity remains 0.613717. No absence-of-call is designated wild type. Gemcitabine features were fixed from [Kerr et al.](https://doi.org/10.1158/1078-0432.CCR-14-0542), [Nakano et al.](https://doi.org/10.1038/sj.bjc.6603559) and [Bjanes et al.](https://doi.org/10.1124/dmd.119.089334). The genetic source is [CCLE2019](https://doi.org/10.1038/s41586-019-1186-3); the annotation correction follows [Leroy et al.](https://doi.org/10.1002/humu.22556) and [Iwakawa et al.](https://doi.org/10.1111/j.1349-7006.2010.01622.x). Gene choice, missing-data treatment and annotation provenance are recorded separately from performance.

## Scientific records

`data/processed.zip` contains only the processed scientific inputs and records needed to audit or regenerate this analysis. Its per-file hashes and the separate fitted RF artifact hash are listed in `data/manifest.json`.

- `forecast/`: normalized module scores for 23,704 source-normal candidate cells, cell/source metadata, exact gene sets, frozen cell-line folds, held-out predictions for every predefined analysis, inner tuning losses, training-control references, fitted coefficients and conditional permutation results. Primary eligibility requires at least 20 matched DMSO and 10 treated cells per pair.
- `model_extension/`: the complete fixed candidate set, final genotype/pharmacology inputs, all fold parameters, unsuccessful candidates, paired intervals and source attribution. The literature splice-disruption correction is explicit; raw calls remain available.
- `classifier/`: normalized predictor/target scores, exclusive task assignments, held-out predictions, fold identities, training-control cutoffs and grouped permutation results. Target DDR genes are removed from biological predictor modules. Task and held-cell-line tests are reported separately.
- `direct_lesions/`: complete paired-cell input and 11 recomputed physical endpoint/sensitivity tables, fixed regions and source manifests.
- `source_context/`: original-source matching checks, source mutation/PRISM annotations and the descriptive functional comparisons.
- `direct_damage/` and `direct_repair/`: compact paired-cell input scores, original-source provenance and the direct endpoint analyses described above.
- `external/` and `genetic/`: compact scored/aggregated inputs for independent expression, live-to-endpoint imaging, DNA-content-gate and genetic-perturbation comparisons, with source provenance. Large, regenerable per-target/batch intermediates are omitted.

`methods/nested_forecasting.py` implements fractional-logit optimization, design matrices, inner grouped selection and line bootstrap; `source_matched_forecasting.py` implements source/QC matching and all fold-specific biological references. HGNC-approved and previous symbols are harmonized only when one biological identity maps to exactly one source column; synonym-only or ambiguous matches are excluded. The frozen memberships and gene weights are preserved. Original DDR coverage is 34/34 in McFarland, and the independent Hallmark DNA-repair/p53 sets cover 150/150 and 199/200 genes after this identity harmonization; PVT1 is absent. Exact substitutions, HGNC IDs, source symbols and official-table provenance are recorded in `forecast/gene_identity_map.json`. The four-state reference uses baseline composite-high fraction and p53, cell-cycle and stress module means, with unpenalized regimen intercepts. The final biological model additionally uses regimen-specific curated TP53 context and the three gemcitabine-only pharmacology groups defined above. Regularization is selected only in inner grouped folds from the fixed candidates 0.3, 3 and 30. Every inner/outer threshold and feature transformation uses training controls/rows only. The model definitions and candidate penalties are recorded in the supplied analysis specifications; grouped cross-validation is distinct from independent prospective validation.

Bootstrap intervals resample complete cell lines with held-out predictions fixed; they do not include refitting or model-selection uncertainty. The conditional permutation preserves regimen-availability patterns and train/test partitions and fixes the selected regularization strengths. Alternative signatures and cutoffs can have different event support and are reported separately. The response is a transcriptional state, not a direct measurement of DNA lesions, repair capacity or cell survival.

## Sources and attribution

The expression resources originate from the study authors and the [scPerturb collection](https://zenodo.org/records/7041849). Exact study/record links and checksums are supplied with the processed inputs. The independent Hallmark signatures use the official [MSigDB human 2026.2 release](https://docs.gsea-msigdb.org/MSigDB/Release_Notes/MSigDB_2026.2.Hs/) and the [Hallmark collection publication](https://doi.org/10.1016/j.cels.2015.12.004); exact membership and observed-gene coverage are preserved in `forecast/canonical_gene_sets.json`. The scPerturb and imaging Zenodo records and the original MIX-seq Figshare annotation record specify CC BY 4.0; MSigDB specifies CC BY 4.0 for the selected Hallmark release. Machine-readable source-record license metadata and attribution are preserved in `upstream_sources.json`. GEO deposits are public, but a separate dataset-specific license is not inferred from an article's open-access license. The HCR-seq source software has an academic noncommercial license; that software is neither imported nor redistributed here. Source data retain their providers' terms and attribution requirements. No blanket software or third-party-data license is granted by this repository.

The physical-lesion source is [Paired-Damage-seq, Nature Methods (2025)](https://doi.org/10.1038/s41592-025-02632-3). Raw workbook reconstruction additionally requires openpyxl; all default analyses use compact numerical inputs.
