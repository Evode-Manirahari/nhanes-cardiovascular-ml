"""Checks for the hand-written statistics in src/paper134_models.py and
src/paper134_analysis.py against scipy or known answers.

Run: .venv/bin/python -m unittest tests/test_paper134_stats.py -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import paper134_analysis as A  # noqa: E402
from paper134_models import kaplan_meier, logrank  # noqa: E402


class TestStatistics(unittest.TestCase):
    def setUp(self):
        r = np.random.default_rng(0)
        self.n = 600
        self.g = r.integers(0, 2, self.n)
        self.t = np.ceil(r.exponential(40 + 30 * self.g)).clip(1, 60)
        self.e = ((self.t < 60) & (r.random(self.n) < 0.8)).astype(int)

    def test_kaplan_meier_matches_scipy(self):
        times, surv = kaplan_meier(self.t, self.e)
        ref = stats.ecdf(stats.CensoredData(uncensored=self.t[self.e == 1],
                                            right=self.t[self.e == 0])).sf
        np.testing.assert_allclose(surv[1:], ref.evaluate(times[1:]), rtol=1e-10)

    def test_logrank_matches_scipy(self):
        samples = [stats.CensoredData(uncensored=self.t[(self.g == k) & (self.e == 1)],
                                      right=self.t[(self.g == k) & (self.e == 0)])
                   for k in (0, 1)]
        ref = stats.logrank(*samples).pvalue
        self.assertAlmostEqual(logrank(self.t, self.e, self.g), ref, delta=ref * 0.02 + 1e-12)

    def test_cumulative_incidence_without_censoring(self):
        # With complete follow-up the Aalen-Johansen estimate is the plain proportion.
        r = np.random.default_rng(1)
        t = r.integers(1, 61, 500).astype(float)
        cause = r.choice([0, 1, 2], 500, p=[0.7, 0.1, 0.2])
        t[cause == 0] = 60
        _, f1 = A.aalen_johansen(t, cause, 1)
        _, f2 = A.aalen_johansen(t, cause, 2)
        self.assertAlmostEqual(f1[-1], (cause == 1).mean(), places=12)
        self.assertAlmostEqual(f2[-1], (cause == 2).mean(), places=12)

    def test_gray_test_detects_only_the_affected_cause(self):
        r = np.random.default_rng(2)
        g = r.integers(0, 2, 4000)
        cause = np.where(r.random(4000) < 0.05 + 0.10 * g, 1,
                         np.where(r.random(4000) < 0.10, 2, 0))
        t = np.where(cause > 0, r.integers(1, 60, 4000), 60).astype(float)
        self.assertLess(A.gray_test(t, cause, g, 1), 1e-6)
        self.assertGreater(A.gray_test(t, cause, g, 2), 0.001)

    def test_calibration_of_true_probabilities(self):
        r = np.random.default_rng(3)
        p = r.uniform(0.02, 0.6, 40000)
        y = (r.random(40000) < p).astype(int)
        slope, citl = A.calibration_w(y, p)
        self.assertAlmostEqual(slope, 1.0, delta=0.05)
        self.assertAlmostEqual(citl, 0.0, delta=0.05)
        # predictions that are too extreme have slope < 1; too high have CITL < 0
        extreme = 1 / (1 + np.exp(-2 * A.logit(p)))
        self.assertLess(A.calibration_w(y, extreme)[0], 0.6)
        self.assertLess(A.calibration_w(y, np.clip(p * 2, 0, 0.99))[1], -0.3)
        # equal weights change nothing
        s2, c2 = A.calibration_w(y, p, np.full(len(y), 7.0))
        self.assertAlmostEqual(s2, slope, places=6)
        self.assertAlmostEqual(c2, citl, places=6)

    def test_nri_idi(self):
        r = np.random.default_rng(4)
        y = r.integers(0, 2, 2000)
        old = r.uniform(0.2, 0.8, 2000)
        self.assertEqual(A.nri_idi(y, old, old), (0.0, 0.0))
        better = np.where(y == 1, old + 0.1, old - 0.1)
        nri, idi = A.nri_idi(y, better, old)
        self.assertAlmostEqual(nri, 2.0)
        self.assertAlmostEqual(idi, 0.2)

    def test_ppv40_operating_point(self):
        y = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
        p = np.array([.9, .8, .7, .6, .5, .4, .3, .2, .1, .05])
        op = A.ppv40_point(y, p)
        # lowest threshold with PPV >= 40% is 0.3 (3 of 7 flagged), sensitivity 100%
        self.assertAlmostEqual(op["Threshold"], 0.3)
        self.assertAlmostEqual(op["Sensitivity"], 1.0)
        self.assertGreaterEqual(op["PPV"], 0.40)


if __name__ == "__main__":
    unittest.main()
