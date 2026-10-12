# Method and reproducibility notes

This repository is based on the newly supplied `final-try-0-589(2).ipynb`. It does not continue the earlier CatBoost-only notebook. The original input is preserved under `reference/`.

## Models retained

The original notebook trains CatBoost, alternating LightGBM GBDT configurations, LightGBM with GOSS, and class-balanced LightGBM. It averages predicted probabilities from all fitted models. The cleaned implementation retains these four families and equal-weight soft voting. It is **not logistic-regression stacking** and contains no XGBoost model.

Maximum, mean, and sample variance summaries are retained for numeric history features. Masked string features retain the original lexical maximum summary; this is a representative token, not an ordinal credit-risk scale. Improving categorical aggregation is an appropriate future ablation, not a finding of this repository.

## Explicit changes from the original

1. Data paths and run size are arguments; a 10-row test template no longer silently selects a 50,000-row training run.
2. Parquet or CSV shards are merged before aggregation. Duplicate depth-0 case IDs raise an error instead of silently selecting a row.
3. Current Polars expressions replace old `apply` calls. Date means are converted back to dates; date variances remain numeric values in squared days.
4. The percentage-range feature is parsed with string expressions, and the original range column is actually dropped.
5. Predictor filtering uses development rows only. Category vocabularies are fitted there and shared with validation/test transforms. Numeric missing values remain numeric missing values; categorical missing/unknown values receive explicit tokens.
6. CPU is the portable default and applies to both libraries. GPU is an explicit option and depends on the local builds. CatBoost's incompatible `max_leaves` setting is removed for its default symmetric trees; no minimum-tree request can exceed the chosen iteration cap.
7. Hyperparameter aliases use `reg_alpha` and `reg_lambda`. The GOSS iteration cap is twice the base cap. CatBoost depth is explicitly 6; the balanced LightGBM uses the same base cap, so this is not an exact reproduction of the old iteration budgets.
8. Early stopping uses an inner tuning subset of development weeks. The selected model is refitted on development data before the evaluation fold is predicted. Evaluation labels are not passed to the estimator's `eval_set`.
9. OOF ensemble predictions use only the models from the fold that held each row out. Submission predictions average models from all folds. The original notebook did not report an OOF ensemble AUC.
10. Mean fold AUC, pooled OOF metrics, weekly AUC, class counts, and a full confusion matrix are saved. The maximum fold AUC is not presented as overall performance.

## Validation choices

**Grouped** is the default to preserve the original week-grouped five-fold design. Weeks are disjoint across training and validation within each fold. Training may include weeks later than a validation week. This does not measure strictly future performance.

**Temporal** reserves the latest 13 observed weeks by default. Model selection uses earlier development weeks; the held-out future weeks remain untouched until evaluation. This is a separate evaluation protocol, not a replacement label for grouped-CV results.

The program reports standard ROC-AUC, average precision, log loss, Brier score, and confusion counts at a fixed threshold of 0.05. Brier score is a descriptive diagnostic; no probability calibration model is fitted. Class-balanced predictions may need calibration before any operational use. Weekly AUC is left undefined for a single-class week. The official Kaggle stability metric is not calculated.

## Results provenance

`results/historical_fold_scores.csv` is transcribed from the **saved outputs** of the replacement notebook. That notebook's dry-run branch used the first 50,000 rows and five week-grouped folds. Its ordering was not a deterministic random sample. These historical results have not been rerun against the complete competition data here. They are not the full-data results reported in the earlier course report, not a leaderboard score, and not an ensemble score.

New runs write their own results to `outputs/`, with Python/package versions, seeds, fold weeks, case counts, and input files. The included correctness tests use synthetic records and do not validate historical accuracy. The cleaned implementation changes preprocessing and validation, so agreement with the historical AUC is neither promised nor assumed.

## Attribution

The supplied course report lists Ziqian Zhang, Yinyu Yao, Kaixun Chen, Biying Xiu, and Zhetao Zhang. It is included for context and is distinct from Zhetao's independently listed credit-risk publication. Specific personal contributions and upstream notebook provenance cannot be established from the supplied files alone; no individual implementation claim or software license is asserted here. Repository cleanup and the new reproducibility code were prepared with AI assistance.

Technical references: [Polars lazy Parquet loading](https://docs.pola.rs/api/python/stable/reference/api/polars.scan_parquet.html), [LightGBM classifier](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html), [CatBoost classifier](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier), [scikit-learn grouped folds](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
