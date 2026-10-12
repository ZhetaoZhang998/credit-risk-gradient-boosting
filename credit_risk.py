"""Credit-risk pipeline adapted from the supplied final-try-0-589 notebook.

Run `python credit_risk.py --help`. Model families follow the new notebook;
data handling and validation changes are documented in docs/METHOD.md.
"""

# SECTION: Imports and configuration
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    log_loss, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

TABLES = [
    ("static_cb_0", 0), ("static_0_*", 0),
    ("applprev_1_*", 1), ("tax_registry_a_1", 1),
    ("tax_registry_b_1", 1), ("tax_registry_c_1", 1),
    ("credit_bureau_a_1_*", 1), ("credit_bureau_b_1", 1),
    ("other_1", 1), ("person_1", 1), ("deposit_1", 1),
    ("debitcard_1", 1), ("credit_bureau_a_2_*", 2),
    ("credit_bureau_b_2", 2),
]
MODEL_NAMES = ["catboost", "lightgbm", "lightgbm_goss", "lightgbm_balanced"]
EXCLUDED = {"target", "case_id", "week_num", "WEEK_NUM", "date_decision"}


# SECTION: Data preparation
def find_split_dir(root: Path, split: str) -> tuple[Path, str]:
    for directory in (root / "parquet_files" / split,
                      root / "csv_files" / split, root / split, root):
        for ext in ("parquet", "csv"):
            if (directory / f"{split}_base.{ext}").exists():
                return directory, ext
    raise FileNotFoundError(
        f"No {split}_base.parquet or {split}_base.csv under {root}. "
        "Unzip the competition data first; see docs/DATA.md."
    )


def scan_table(paths: list[Path], case_ids: pl.DataFrame | None = None) -> pl.LazyFrame:
    """Concatenate raw shards before aggregation, retaining every record."""
    if not paths:
        raise FileNotFoundError("No files matched a required table.")
    scans = []
    for path in sorted(paths):
        if path.suffix == ".parquet":
            df = pl.scan_parquet(path)
        else:
            df = pl.scan_csv(path, infer_schema_length=10000)
        schema = df.collect_schema()
        casts = []
        for col, dtype in schema.items():
            if col in ("case_id", "WEEK_NUM", "num_group1", "num_group2"):
                casts.append(pl.col(col).cast(pl.Int64))
            elif col == "date_decision" or col.endswith("D"):
                if dtype == pl.String:
                    casts.append(pl.col(col).str.to_date(strict=False))
                else:
                    casts.append(pl.col(col).cast(pl.Date))
            elif col.endswith(("P", "A")):
                casts.append(pl.col(col).cast(pl.Float64))
            elif col.endswith("M"):
                casts.append(pl.col(col).cast(pl.String))
        df = df.with_columns(casts)
        if case_ids is not None:
            df = df.join(case_ids.lazy(), on="case_id", how="semi")
        scans.append(df)
    return pl.concat(scans, how="diagonal_relaxed")


def aggregate_table(df: pl.LazyFrame, depth: int) -> pl.DataFrame:
    if depth == 0:
        result = df.collect(engine="streaming")
        if result["case_id"].n_unique() != result.height:
            raise ValueError("A depth-0 table contains duplicate case_id rows.")
        return result
    exprs = []
    for col, dtype in df.collect_schema().items():
        if col == "case_id":
            continue
        if col.endswith(("P", "M", "A", "D", "T", "L")) or "num_group" in col:
            exprs.append(pl.col(col).max().alias(f"max_{col}"))
        if col.endswith(("P", "A")) and dtype.is_numeric():
            exprs.extend([
                pl.col(col).mean().alias(f"mean_{col}"),
                pl.col(col).var(ddof=1).alias(f"var_{col}"),
            ])
        elif col.endswith("D"):
            # Date means are dates; date variance is in squared days, not a date.
            days = pl.col(col).cast(pl.Int32)
            exprs.extend([
                days.mean().cast(pl.Int32).cast(pl.Date).alias(f"mean_{col}"),
                days.var(ddof=1).alias(f"var_{col}_days_squared"),
            ])
    return df.group_by("case_id").agg(exprs).collect(engine="streaming")


def transform_features(df: pl.DataFrame) -> pl.DataFrame:
    names = df.columns
    if "riskassesment_302T" in names:
        value = pl.col("riskassesment_302T").cast(pl.String)
        low = value.str.extract(r"([0-9]+(?:\.[0-9]+)?)\s*%", 1).cast(pl.Float32)
        high = value.str.extract(r"-\s*([0-9]+(?:\.[0-9]+)?)\s*%", 1).cast(pl.Float32)
        df = df.with_columns(
            (high - low).alias("riskassesment_302T_rng"),
            ((high + low) / 2).alias("riskassesment_302T_mean"),
        ).drop("riskassesment_302T")
    df = df.with_columns(
        pl.col("date_decision").dt.year().cast(pl.Int16).alias("year"),
        pl.col("date_decision").dt.day().cast(pl.UInt8).alias("day"),
    )
    for col, dtype in df.schema.items():
        if col != "date_decision" and dtype == pl.Date:
            df = df.with_columns(
                (pl.col(col) - pl.col("date_decision")).dt.total_days().alias(col)
            )
    df = df.drop("date_decision")
    renames = {old: new for old, new in [("MONTH", "month"), ("WEEK_NUM", "week_num")]
               if old in df.columns}
    df = df.rename(renames)
    # Predictors use float32; identifiers remain int64 to avoid overflow.
    numeric = [c for c, d in df.schema.items()
               if d.is_numeric() and c not in {"case_id", "week_num", "target"}]
    df = df.with_columns([pl.col(c).cast(pl.Float32) for c in numeric])
    return df


def prepare_data(root: Path, split: str, max_cases: int, seed: int,
                 allow_missing_tables: bool = False) -> tuple[pd.DataFrame, dict]:
    directory, ext = find_split_dir(root, split)
    base = scan_table([directory / f"{split}_base.{ext}"]).collect()
    if base["case_id"].n_unique() != base.height:
        raise ValueError("The base table must have one row per case_id.")
    total_cases = base.height
    if split == "train" and max_cases > 0 and base.height > max_cases:
        base = base.sample(n=max_cases, seed=seed, shuffle=True)
    ids = base.select("case_id")
    missing = []
    input_files = [str(directory / f"{split}_base.{ext}")]
    for suffix_index, (pattern, depth) in enumerate(TABLES):
        paths = sorted(directory.glob(f"{split}_{pattern}.{ext}"))
        if not paths:
            if not allow_missing_tables:
                raise FileNotFoundError(f"Missing required table: {split}_{pattern}.{ext}")
            missing.append(pattern)
            continue
        table = aggregate_table(scan_table(paths, ids), depth)
        base = base.join(table, on="case_id", how="left", suffix=f"_{suffix_index}")
        input_files.extend(str(p) for p in paths)
        del table
        gc.collect()
    features = transform_features(base).sort("case_id").to_pandas()
    metadata = {"split": split, "total_base_cases": total_cases,
                "prepared_cases": len(features), "sample_seed": seed,
                "max_cases": max_cases, "input_format": ext,
                "missing_tables": missing, "input_files": input_files}
    return features, metadata


# SECTION: Fold-local feature schema and validation
@dataclass
class FeatureSchema:
    null_limit: float = 0.95
    columns: list[str] = field(default_factory=list)
    categories: dict[str, list[str]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "FeatureSchema":
        self.columns = []
        self.categories = {}
        for col in df.columns:
            if col in EXCLUDED or df[col].isna().mean() > self.null_limit:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                values = sorted(df[col].dropna().astype(str).unique())
                if len(values) <= 1 or len(values) > 200:
                    continue
                self.categories[col] = values
            self.columns.append(col)
        if not self.columns:
            raise ValueError("No usable predictors remain after feature filtering.")
        return self

    def transform(self, df: pd.DataFrame, catboost: bool = False) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self.columns:
            source = df[col] if col in df else pd.Series(np.nan, index=df.index)
            if col in self.categories:
                values = source.astype("string").fillna("__MISSING__")
                vocab = self.categories[col] + ["__MISSING__", "__UNKNOWN__"]
                values = values.where(values.isin(vocab), "__UNKNOWN__")
                out[col] = values.astype(str) if catboost else pd.Categorical(values, categories=vocab)
            else:
                out[col] = pd.to_numeric(source, errors="coerce").replace(
                    [np.inf, -np.inf], np.nan).astype("float32")
        return out


def temporal_split(df: pd.DataFrame, holdout_weeks: int) -> tuple[np.ndarray, np.ndarray]:
    weeks = np.sort(df["week_num"].unique())
    if holdout_weeks < 1 or len(weeks) <= holdout_weeks + 1:
        raise ValueError("Need at least two development weeks before the holdout.")
    mask = df["week_num"].isin(weeks[-holdout_weeks:]).to_numpy()
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def evaluation_splits(df: pd.DataFrame, validation: str, folds: int,
                      holdout_weeks: int):
    if df["week_num"].isna().any():
        raise ValueError("week_num may not contain missing values.")
    if df["case_id"].duplicated().any():
        raise ValueError("Duplicate cases would contaminate validation.")
    if df["target"].isna().any() or set(df["target"].unique()) != {0, 1}:
        raise ValueError("target must contain both classes, encoded as 0 and 1.")
    if validation == "temporal":
        return [temporal_split(df, holdout_weeks)]
    if df["week_num"].nunique() < folds:
        raise ValueError("There are fewer distinct weeks than folds.")
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=False)
    return list(cv.split(df, df["target"], groups=df["week_num"]))


# SECTION: Models and evaluation
def make_model(name: str, iterations: int, seed: int, device: str, threads: int,
               fold: int, tune: bool):
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=iterations, depth=6, boosting_type="Plain",
            learning_rate=0.05, l2_leaf_reg=10, random_seed=seed,
            eval_metric="AUC", task_type=device.upper(), thread_count=threads,
            allow_writing_files=False, verbose=False,
            use_best_model=tune,
        )
    import lightgbm as lgb
    params = dict(
        boosting_type="goss" if name == "lightgbm_goss" else "gbdt",
        colsample_bynode=0.8, colsample_bytree=0.8, extra_trees=True,
        learning_rate=0.03 if name == "lightgbm_goss" or (name == "lightgbm" and fold % 2) else 0.05,
        reg_alpha=0.1, reg_lambda=10,
        max_depth=16 if name == "lightgbm_goss" or (name == "lightgbm" and fold % 2) else 20,
        num_leaves=54 if name == "lightgbm_goss" or (name == "lightgbm" and fold % 2) else 64,
        n_estimators=iterations, objective="binary", metric="auc",
        random_state=seed, device_type=device, n_jobs=threads, verbosity=-1,
    )
    if device == "cpu":
        params.update(deterministic=True, force_col_wise=True)
    if name == "lightgbm_balanced":
        params["is_unbalance"] = True
    return lgb.LGBMClassifier(**params)


def fit_model(name: str, df: pd.DataFrame, iterations: int, seed: int,
              device: str, threads: int, fold: int):
    # Select tree count on earlier/later subsets of development data only.
    tune_weeks = min(6, max(1, df["week_num"].nunique() // 5))
    fit_idx, tune_idx = temporal_split(df, tune_weeks)
    earlier, tuning = df.iloc[fit_idx], df.iloc[tune_idx]
    if earlier["target"].nunique() < 2 or tuning["target"].nunique() < 2:
        raise ValueError("Both classes are required in inner training and tuning subsets.")
    schema = FeatureSchema().fit(earlier)
    is_cat = name == "catboost"
    x_fit = schema.transform(earlier, catboost=is_cat)
    x_tune = schema.transform(tuning, catboost=is_cat)
    model = make_model(name, iterations, seed, device, threads, fold, tune=True)
    if is_cat:
        model.fit(x_fit, earlier["target"], cat_features=list(schema.categories),
                  eval_set=(x_tune, tuning["target"]), early_stopping_rounds=50)
        selected = max(1, model.get_best_iteration() + 1)
    else:
        import lightgbm as lgb
        model.fit(x_fit, earlier["target"], eval_set=[(x_tune, tuning["target"])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        selected = max(1, model.best_iteration_)
    del x_fit, x_tune, model
    gc.collect()
    # Refit on all development rows; the evaluation fold is never an eval_set.
    schema = FeatureSchema().fit(df)
    model = make_model(name, selected, seed, device, threads, fold, tune=False)
    x = schema.transform(df, catboost=is_cat)
    if is_cat:
        model.fit(x, df["target"], cat_features=list(schema.categories))
    else:
        model.fit(x, df["target"])
    return model, schema, selected


def metrics(y, p, threshold: float) -> dict:
    if pd.Series(y).nunique() != 2:
        raise ValueError("An evaluation fold must contain both target classes.")
    tn, fp, fn, tp = confusion_matrix(y, np.asarray(p) >= threshold, labels=[0, 1]).ravel()
    return {"auc": float(roc_auc_score(y, p)),
            "average_precision": float(average_precision_score(y, p)),
            "brier_score": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "threshold": threshold, "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp), "n_cases": len(y)}


def weekly_metrics(df: pd.DataFrame, name: str, p) -> list[dict]:
    values = pd.DataFrame({"week_num": df["week_num"].to_numpy(),
                           "target": df["target"].to_numpy(), "p": p})
    result = []
    for week, rows in values.groupby("week_num", sort=True):
        auc = float(roc_auc_score(rows.target, rows.p)) if rows.target.nunique() == 2 else np.nan
        result.append({"model": name, "week_num": int(week), "n_cases": len(rows),
                       "n_defaults": int(rows.target.sum()), "auc": auc,
                       "gini": 2 * auc - 1 if np.isfinite(auc) else np.nan})
    return result


def run_experiment(train: pd.DataFrame, test: pd.DataFrame | None,
                   output: Path, validation: str = "grouped", folds: int = 5,
                   holdout_weeks: int = 13, iterations: int = 300,
                   seed: int = 42, device: str = "cpu", threads: int = 2,
                   threshold: float = 0.05, dataset_label: str = "home_credit") -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    splits = evaluation_splits(train, validation, folds, holdout_weeks)
    oof = {name: np.full(len(train), np.nan) for name in MODEL_NAMES + ["ensemble"]}
    test_preds = {name: [] for name in MODEL_NAMES}
    fold_rows, split_records = [], []
    for fold, (idx_train, idx_valid) in enumerate(splits):
        development, heldout = train.iloc[idx_train].copy(), train.iloc[idx_valid].copy()
        split_records.append({"fold": fold,
            "train_weeks": sorted(int(v) for v in development.week_num.unique()),
            "valid_weeks": sorted(int(v) for v in heldout.week_num.unique())})
        predictions = []
        for name in MODEL_NAMES:
            print(f"Fold {fold + 1}/{len(splits)}: {name}", flush=True)
            start = time.perf_counter()
            cap = iterations * 2 if name == "lightgbm_goss" else iterations
            model, schema, selected = fit_model(
                name, development, cap, seed, device, threads, fold)
            p = model.predict_proba(schema.transform(heldout, catboost=name == "catboost"))[:, 1]
            oof[name][idx_valid] = p
            predictions.append(p)
            fold_rows.append({"fold": fold, "model": name,
                "selected_iterations": selected,
                "fit_and_predict_seconds": time.perf_counter() - start,
                **metrics(heldout.target, p, threshold)})
            if test is not None:
                test_preds[name].append(model.predict_proba(
                    schema.transform(test, catboost=name == "catboost"))[:, 1])
            del model, schema
            gc.collect()
        p_ensemble = np.mean(predictions, axis=0)
        oof["ensemble"][idx_valid] = p_ensemble
        fold_rows.append({"fold": fold, "model": "ensemble",
                          **metrics(heldout.target, p_ensemble, threshold)})
    evaluated = np.isfinite(oof["ensemble"])
    rows = train.loc[evaluated]
    summary, weekly = [], []
    predictions_frame = rows[["case_id", "week_num", "target"]].copy()
    for name, values in oof.items():
        p = values[evaluated]
        summary.append({"model": name, "dataset": dataset_label,
                        "validation": validation, **metrics(rows.target, p, threshold)})
        weekly.extend(weekly_metrics(rows, name, p))
        predictions_frame[name] = p
    result = pd.DataFrame(summary)
    result.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(weekly).to_csv(output / "weekly_metrics.csv", index=False)
    predictions_frame.to_csv(output / "validation_predictions.csv", index=False)
    if test is not None:
        p_test = np.mean([np.mean(test_preds[n], axis=0) for n in MODEL_NAMES], axis=0)
        pd.DataFrame({"case_id": test.case_id.to_numpy(), "score": p_test}).to_csv(
            output / "submission.csv", index=False)
    versions = {}
    for name in ["numpy", "pandas", "polars", "pyarrow", "scikit-learn", "lightgbm", "catboost"]:
        versions[name] = importlib.metadata.version(name)
    metadata = {"dataset_label": dataset_label, "validation": validation,
        "seed": seed, "iteration_cap": iterations, "device": device, "threads": threads,
        "threshold": threshold, "prepared_train_cases": len(train),
        "evaluated_cases": int(evaluated.sum()), "folds": split_records,
        "python": platform.python_version(), "packages": versions,
        "note": "Grouped CV is week-disjoint, not forward-in-time. No official Kaggle stability score is calculated."}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot_results(result, pd.DataFrame(weekly), output)
    return result


def plot_results(summary: pd.DataFrame, weekly: pd.DataFrame, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    palette = ["#8dafa6", "#b6c9c1", "#40675e", "#be9a7a", "#282d2b"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor="#faf9f5")
    for ax in axes:
        ax.set_facecolor("#faf9f5")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].barh(summary.model, summary.auc, color=palette)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Validation ROC-AUC")
    axes[0].set_title("Model comparison", loc="left")
    for name, color in zip(MODEL_NAMES + ["ensemble"], palette):
        line = weekly.loc[weekly.model == name].sort_values("week_num")
        axes[1].plot(line.week_num, line.auc, color=color, label=name)
    axes[1].set_title("Weekly discrimination", loc="left")
    axes[1].set_xlabel("Application week")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{summary.dataset.iloc[0]} · {summary.validation.iloc[0]} validation", fontsize=11)
    fig.tight_layout()
    fig.savefig(output / "model_comparison.png", dpi=180)
    plt.close(fig)


# SECTION: Command line
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("outputs/run"))
    parser.add_argument("--max-cases", type=int, default=20000, help="0 uses all training cases")
    parser.add_argument("--validation", choices=["grouped", "temporal"], default="grouped")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--holdout-weeks", type=int, default=13)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--allow-missing-tables", action="store_true",
                        help="Only for an explicitly reduced experiment")
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("threshold must be between 0 and 1")
    if args.max_cases < 0 or args.iterations < 1 or args.folds < 2 or args.threads < 1:
        parser.error("Invalid case limit, iterations, folds, or threads")
    train, train_meta = prepare_data(args.data_root, "train", args.max_cases,
                                     args.seed, args.allow_missing_tables)
    test, test_meta = (None, None) if args.skip_test else prepare_data(
        args.data_root, "test", 0, args.seed, args.allow_missing_tables)
    result = run_experiment(train, test, args.output, args.validation, args.folds,
        args.holdout_weeks, args.iterations, args.seed, args.device, args.threads, args.threshold)
    (args.output / "data_metadata.json").write_text(json.dumps(
        {"train": train_meta, "test": test_meta}, indent=2), encoding="utf-8")
    print(result[["model", "auc", "average_precision", "brier_score"]].to_string(index=False))
    print(f"Saved outputs to {args.output}")


if __name__ == "__main__":
    main()
