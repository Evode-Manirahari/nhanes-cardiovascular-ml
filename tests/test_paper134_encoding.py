"""Checks for the categorical handling in src/paper134_models.py.

Run: .venv/bin/python -m unittest tests/test_paper134_encoding.py -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import paper134_models as M  # noqa: E402
from paper134_models import Prep  # noqa: E402

WORKBOOK = ROOT / "results" / "nhanes_evidence_audit.xlsx"
EXPECTED = {"RIAGENDR", "RIDRETH1", "INDHHINC", "PAQ180", "CVDFITLV", "BPQ020", "MCQ160B",
            "MCQ160E", "MCQ160F", "DIQ010", "XY_Estimation",
            "DERIVED_HL_MED", "DERIVED_HTN_MED", "DERIVED_DM_MED"}


class TestCategoricalList(unittest.TestCase):
    def test_fourteen_categorical_predictors(self):
        self.assertEqual(set(M.CATEGORICAL), EXPECTED)
        self.assertEqual(len(M.CATEGORICAL), 14)

    @unittest.skipUnless(WORKBOOK.exists(), "variable audit workbook not available")
    def test_matches_variable_audit_workbook(self):
        w = pd.read_excel(WORKBOOK, sheet_name="Paper 134")
        marked = set(w.loc[w["One-hot?"].astype(str).str.lower() == "yes",
                           "Working code / derivation"])
        # the workbook maps the heart-attack row to MCQ160C; the pipeline uses MCQ160E
        marked = {M.REPLACED.get(c, c) for c in marked}
        self.assertEqual(marked, set(M.CATEGORICAL))

    def test_all_categorical_are_model_inputs(self):
        X, _, _, _ = M.load()
        self.assertTrue(set(M.CATEGORICAL) <= set(X.columns))
        self.assertEqual(X.shape[1], 134)
        # no measured (many-valued) predictor is treated as a category, and no
        # predictor with only two or three values is left as a number
        self.assertLessEqual(max(X[c].nunique() for c in M.CATEGORICAL), 15)
        numeric = [c for c in X.columns if c not in M.CATEGORICAL]
        self.assertGreater(min(X[c].nunique() for c in numeric), 3)


class TestEncoder(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({"RIAGENDR": [1, 2, 2, 2, np.nan],
                                   "PAQ180": [1, 2, 3, 2, 2],
                                   "LBXGH": [5.0, np.nan, 6.1, 5.5, 7.0]})
        self.test = pd.DataFrame({"RIAGENDR": [np.nan, 1], "PAQ180": [4, 1],  # 4 is unseen
                                  "LBXGH": [np.nan, 5.2]})
        self.prep = Prep(list(self.train.columns)).fit(self.train, impute=False)

    def test_columns_and_grouping(self):
        m, groups = self.prep.native(self.train)
        self.assertEqual(list(m.columns), ["LBXGH", "RIAGENDR=1", "RIAGENDR=2",
                                           "PAQ180=1", "PAQ180=2", "PAQ180=3"])
        self.assertEqual(groups, ["LBXGH", "RIAGENDR", "RIAGENDR", "PAQ180", "PAQ180", "PAQ180"])

    def test_one_indicator_per_person(self):
        m, _ = self.prep.native(self.train)
        self.assertTrue((m[["RIAGENDR=1", "RIAGENDR=2"]].sum(axis=1) == 1).all())
        self.assertTrue((m[["PAQ180=1", "PAQ180=2", "PAQ180=3"]].sum(axis=1) == 1).all())

    def test_missing_category_gets_most_frequent(self):
        m, _ = self.prep.native(self.train)
        self.assertEqual(m.loc[4, "RIAGENDR=2"], 1.0)  # 2 is the most frequent in training
        t, _ = self.prep.native(self.test)
        self.assertEqual(t.loc[0, "RIAGENDR=2"], 1.0)  # filled from TRAINING, not test

    def test_unseen_category_is_all_zeros(self):
        t, _ = self.prep.native(self.test)
        self.assertEqual(t.loc[0, ["PAQ180=1", "PAQ180=2", "PAQ180=3"]].sum(), 0.0)
        self.assertNotIn("PAQ180=4", t.columns)

    def test_numeric_missing_values_kept_for_boosted_trees(self):
        m, _ = self.prep.native(self.train)
        self.assertTrue(np.isnan(m.loc[1, "LBXGH"]))

    def test_real_data_fold(self):
        from sklearn.model_selection import StratifiedKFold
        X, y, _, _ = M.load()
        tr, te = next(StratifiedKFold(5, shuffle=True, random_state=M.SEED).split(X, y))
        prep = Prep(list(X.columns)).fit(X.iloc[tr], impute=False)
        a, groups = prep.native(X.iloc[tr])
        b, _ = prep.native(X.iloc[te])
        self.assertEqual(list(a.columns), list(b.columns))
        g = np.asarray(groups)
        for c in M.CATEGORICAL:
            cols = a.columns[g == c]
            self.assertTrue(set(np.unique(a[cols])) <= {0.0, 1.0})
            self.assertTrue((a[cols].sum(axis=1) == 1).all(), c)
            self.assertTrue((b[cols].sum(axis=1) <= 1).all(), c)
        self.assertEqual(len(set(groups)), 134)


if __name__ == "__main__":
    unittest.main()
