"""Checks for the paper-faithful replication dataset (src/paper134_replication.py).

Run: .venv/bin/python -m unittest tests/test_paper134_replication.py -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import paper134_build as B  # noqa: E402
import paper134_models as M  # noqa: E402
import paper134_replication as P  # noqa: E402

RAW = ROOT / "data" / "raw"


def raw(stem):
    df = pd.read_sas(RAW / f"{stem}.xpt", format="xport")
    num = df.select_dtypes("number").columns
    df[num] = df[num].mask(df[num].abs() < 1e-70, 0.0)
    return df.drop_duplicates("SEQN").set_index("SEQN")


def same(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.isclose(a, b, rtol=1e-12, atol=0, equal_nan=True)


@unittest.skipUnless(P.OUT.exists(), "run src/paper134_replication.py first")
class TestReplicationDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = pd.read_pickle(P.OUT).set_index("SEQN")
        cls.f = pd.read_pickle(P.FULL).set_index("SEQN").reindex(cls.r.index)
        cls.cy = cls.r["SDDSRVYR"]

    def test_same_people_and_outcome(self):
        self.assertTrue(self.r.index.equals(self.f.index))
        self.assertTrue(self.r["death_5yr"].equals(self.f["death_5yr"]))
        self.assertEqual(int(self.r["death_5yr"].sum()), 1293)

    def test_only_documented_columns_differ(self):
        allowed = set(P.SECOND_EXAM) | set(P.YES_NO) | set(P.MEDICATION) | {"MET_MIN_30D_CAND"}
        for c in self.f.columns:
            if c in allowed:
                continue
            with self.subTest(column=c):
                self.assertTrue(same(self.r[c], self.f[c]).all())

    def test_2001_2002_labs_come_from_second_exam_files(self):
        for col, (stem, src) in P.SECOND_EXAM.items():
            with self.subTest(column=col):
                s = raw(stem)[src].reindex(self.r.index)
                k = self.cy == 2
                self.assertTrue(same(self.r.loc[k, col], s[k]).all())
                k = self.cy == 1  # 1999-2000 unchanged
                self.assertTrue(same(self.r.loc[k, col], self.f.loc[k, col]).all())
                k = self.cy == 3
                if col in P.NO_2003_2004:
                    self.assertTrue(self.r.loc[k, col].isna().all())
                else:
                    self.assertTrue(same(self.r.loc[k, col], self.f.loc[k, col]).all())

    def test_yes_no_and_medication_coding(self):
        mcq = pd.concat([raw(f)[["MCQ160E"]] for f in ("MCQ", "MCQ_B", "MCQ_C")])
        orig = {**{c: self.f[c] for c in P.YES_NO if c != "MCQ160E"},
                "MCQ160E": mcq["MCQ160E"].reindex(self.r.index)}
        for c in P.YES_NO:
            with self.subTest(column=c):
                self.assertTrue(self.r[c].isin([0, 1]).all())
                self.assertTrue(((self.r[c] == 1) == (orig[c] == 1)).all())
        for c in P.MEDICATION:
            with self.subTest(column=c):
                self.assertFalse(self.r[c].isna().any())
                k = self.f[c].notna()
                self.assertTrue((self.r.loc[k, c] == self.f.loc[k, c]).all())

    def test_met_zero_only_for_people_without_usable_activity_data(self):
        diff = ~same(self.r["MET_MIN_30D_CAND"], self.f["MET_MIN_30D_CAND"])
        self.assertEqual(int(diff.sum()), 5)
        self.assertTrue((self.r.loc[diff, "MET_MIN_30D_CAND"] == 0).all())
        self.assertTrue(self.f.loc[diff, "MET_MIN_30D_CAND"].isna().all())

    def test_table_s1_counts(self):
        cross = pd.read_csv(M.CROSS)
        s1 = B.s1_lookup(cross, B.read_table_s1())
        old = M.CAND
        try:
            M.CAND = P.OUT
            X, _, _, names = M.load()
        finally:
            M.CAND = old
        diffs = {names[f]: int(X[f].notna().sum()) - s1[r["Paper feature"]]
                 for f, (_, r) in zip(X.columns, cross.iterrows())
                 if int(X[f].notna().sum()) != s1[r["Paper feature"]]}
        # income and Troponin-T are deliberately not forced to match
        self.assertEqual(diffs, {"Income": 24, "Troponin-T": 1})


if __name__ == "__main__":
    unittest.main()
