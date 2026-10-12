# Verification record

Checked on 12 October 2026 UTC with Python 3.12 and the versions in requirements.txt.

- Eight correctness checks passed: cross-shard aggregation, duplicate detection, CSV loading, date moments, percentage ranges and date offsets, fold-local category schemas, group boundaries, forward temporal boundaries, and threshold counts. Some checks cover more than one behavior.
- An end-to-end synthetic fixture contained 600 training cases across 15 weeks, 10 test cases, and every table used by the supplied notebook. CatBoost and all three LightGBM configurations trained and produced valid probabilities.
- Three-fold grouped validation and a three-week temporal holdout both completed, including ensemble predictions, weekly diagnostics, figures, and correctly aligned submission case IDs.
- Preparing the same fixture from CSV and Parquet produced equivalent numeric features.
- The self-contained notebook's code was syntax-checked, its function cells were loaded, and its final execution cell was exercised against the fixture with a small iteration budget.

Synthetic records and synthetic score tables are not included in the project results. These checks establish software behavior only. The complete competition data was not available in this session, so neither the original historical AUC nor the new pipeline's real-data performance was reproduced.
