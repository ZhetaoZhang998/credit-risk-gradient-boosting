# Data setup

The source is the [Home Credit Credit Risk Model Stability competition](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data). Access and download through your own Kaggle account under the competition's terms. No competition records are redistributed here.

Extract the archive rather than passing the ZIP to the pipeline. The original notebook uses **Parquet**, not a single train.csv/test.csv pair. The cleaned pipeline also accepts the competition's multi-table CSV layout.

Place the extracted competition folder under `data/raw/`. The loader recognizes:

- `data/raw/parquet_files/train/train_base.parquet` and the related Parquet tables;
- `data/raw/parquet_files/test/test_base.parquet` and related test tables;
- alternatively, the same multi-table layout in `data/raw/csv_files/train/` and `data/raw/csv_files/test/`.

Use `--data-root` if your extracted folder is elsewhere. On Windows, quote the path:

```powershell
python credit_risk.py --data-root "D:/HomeCredit" --max-cases 20000 --iterations 150
```

Training IDs are sampled once from the base table using the recorded seed. Related tables are filtered by those IDs, so a customer keeps all their available records. Files belonging to one table are concatenated **before** grouping. This prevents the per-shard aggregation followed by case-ID deduplication in the original code from discarding records when a case spans files.

The normal run expects every table used in the replacement notebook. `--allow-missing-tables` is for an explicitly reduced experiment; omitted tables are recorded in `data_metadata.json`. Do not describe such a run as using the complete feature set.

The supplied `sample_submission.csv` contains 10 case IDs. These are unlabeled test examples. They can exercise submission formatting but cannot estimate predictive accuracy. Evaluation uses labeled training cases in separate folds or a later-week holdout.

The defaults use 20,000 randomly selected cases to keep the first run manageable. `--max-cases 0` uses all training cases and can require substantial memory. Raw file size, the joined feature frame, and model training memory are different quantities. Start with the sample and inspect memory before a full run.
