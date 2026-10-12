"""Checks for data integrity and evaluation; fixtures contain no real borrowers."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import polars as pl
from credit_risk import (
    FeatureSchema, aggregate_table, evaluation_splits, scan_table,
    temporal_split, transform_features, metrics,
)


class PipelineTests(unittest.TestCase):
    def test_cross_shard_aggregation_keeps_all_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / 'a.parquet', Path(tmp) / 'b.parquet']
            pl.DataFrame({'case_id': [1, 1], 'valueA': [2., 4.]}).write_parquet(paths[0])
            pl.DataFrame({'case_id': [1, 2], 'valueA': [12., 3.]}).write_parquet(paths[1])
            result = aggregate_table(scan_table(paths), 1).sort('case_id')
            self.assertEqual(result['max_valueA'][0], 12.)
            self.assertEqual(result['mean_valueA'][0], 6.)
            self.assertEqual(result['var_valueA'][0], 28.)

    def test_depth_zero_duplicates_raise(self):
        df = pl.DataFrame({'case_id': [1, 1], 'valueA': [2., 4.]}).lazy()
        with self.assertRaises(ValueError):
            aggregate_table(df, 0)

    def test_csv_input_and_date_variance(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'sample.csv'
            p.write_text('case_id,valueA,eventD\n1,2,2024-01-01\n1,4,2024-01-03\n')
            out = aggregate_table(scan_table([p]), 1)
            self.assertEqual(out['mean_valueA'][0], 3.)
            self.assertEqual(out['var_eventD_days_squared'][0], 2.)
            self.assertEqual(out['mean_eventD'].dtype, pl.Date)

    def test_percentage_range_and_relative_dates(self):
        df = pl.DataFrame({'case_id': [1], 'WEEK_NUM': [5], 'MONTH': [202401],
            'date_decision': ['2024-01-10'], 'eventD': ['2024-01-03'],
            'riskassesment_302T': ['20% - 40%']}).with_columns(
                pl.col('date_decision').str.to_date(), pl.col('eventD').str.to_date())
        out = transform_features(df)
        self.assertNotIn('riskassesment_302T', out.columns)
        self.assertEqual(out['riskassesment_302T_rng'][0], 20.)
        self.assertEqual(out['riskassesment_302T_mean'][0], 30.)
        self.assertEqual(out['eventD'][0], -7.)

    def test_schema_uses_training_rows_and_excludes_identifiers(self):
        train = pd.DataFrame({'case_id': [1, 2, 3, 4], 'target': [0, 1, 0, 1],
            'week_num': [1, 1, 2, 2], 'category': ['a', 'b', None, 'a'],
            'numeric': [1., np.nan, 3., 4.], 'empty': [np.nan] * 4})
        schema = FeatureSchema().fit(train)
        test = pd.DataFrame({'category': ['new', None], 'numeric': [5., np.nan]})
        x = schema.transform(test)
        self.assertEqual(schema.columns, ['category', 'numeric'])
        self.assertEqual(x.category.astype(str).tolist(), ['__UNKNOWN__', '__MISSING__'])
        self.assertTrue(np.isnan(x.numeric.iloc[1]))
        self.assertEqual(list(x.category.cat.categories), ['a', 'b', '__MISSING__', '__UNKNOWN__'])

    def test_group_folds_have_disjoint_weeks(self):
        df = pd.DataFrame({'case_id': np.arange(60), 'target': [0, 1] * 30,
                           'week_num': np.repeat(np.arange(10), 6)})
        visited = []
        for a, b in evaluation_splits(df, 'grouped', 5, 2):
            self.assertFalse(set(df.week_num.iloc[a]) & set(df.week_num.iloc[b]))
            visited.extend(b)
        self.assertEqual(sorted(visited), list(range(len(df))))

    def test_temporal_split_never_trains_on_future_weeks(self):
        df = pd.DataFrame({'week_num': [9, 2, 1, 8, 3, 7]})
        a, b = temporal_split(df, 2)
        self.assertLess(df.week_num.iloc[a].max(), df.week_num.iloc[b].min())
        self.assertEqual(set(df.week_num.iloc[b]), {8, 9})

    def test_confusion_matrix_counts_at_fixed_threshold(self):
        m = metrics([0, 0, 1, 1], [.01, .10, .03, .80], .05)
        self.assertEqual((m['tn'], m['fp'], m['fn'], m['tp']), (1, 1, 1, 1))


if __name__ == '__main__':
    unittest.main()
