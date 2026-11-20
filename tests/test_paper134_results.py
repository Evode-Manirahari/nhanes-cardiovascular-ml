"""Checks on the model results, the figures and the derived medication flags.

Needs results/models_v0.2/ (written by src/paper134_analysis.py) and the raw data.

Run: .venv/bin/python -m unittest tests/test_paper134_results.py -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import paper134_analysis as A  # noqa: E402
import paper134_models as M  # noqa: E402

OUT = ROOT / "results" / "models_v0.2"
MAIN = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
PAPER_AUC = {"Logistic Regression": 0.8577, "Random Forest": 0.8597,
             "XGBoost": 0.8660, "LightGBM": 0.8621}


@unittest.skipUnless((OUT / "summary.json").exists(), "run src/paper134_analysis.py first")
class TestSavedResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oof = pd.read_csv(OUT / "oof_predictions.csv", index_col=0)
        cls.cv = pd.read_csv(OUT / "cv_metrics.csv").set_index("Model")
        with (OUT / "summary.json").open() as f:
            cls.info = json.load(f)

    def test_results_come_from_a_full_run(self):
        self.assertFalse(self.info["fast_mode"])
        self.assertEqual(len(self.info["categorical_predictors"]), 14)

    def test_every_person_scored_once_out_of_fold(self):
        X, y, _, _ = M.load()
        self.assertEqual(list(self.oof.index), list(X.index))
        self.assertTrue((self.oof["death_5yr"] == y).all())
        self.assertFalse(self.oof.isna().any().any())
        folds = np.full(len(X), -1)
        for f, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=M.SEED).split(X, y)):
            folds[te] = f
        self.assertTrue((self.oof["fold"].to_numpy() == folds).all())
        rate = self.oof.groupby("fold")["death_5yr"].mean()
        self.assertLess(rate.max() - rate.min(), 0.002)  # stratified

    def test_reported_metrics_match_independent_recomputation(self):
        y = self.oof["death_5yr"].to_numpy()
        for m in self.cv.index:
            p = self.oof[m].to_numpy()
            with self.subTest(model=m):
                self.assertTrue(((p >= 0) & (p <= 1)).all())
                for col, val in [("ROC-AUC", roc_auc_score(y, p)),
                                 ("PR-AUC", average_precision_score(y, p)),
                                 ("Brier", brier_score_loss(y, p)),
                                 ("Precision@0.5", precision_score(y, p >= 0.5)),
                                 ("Recall@0.5", recall_score(y, p >= 0.5)),
                                 ("F1@0.5", f1_score(y, p >= 0.5))]:
                    # 6 places: predictions pass through a CSV, which can reorder exact ties
                    self.assertAlmostEqual(self.cv.loc[m, col], val, places=6, msg=col)

    def test_operating_points_are_what_they_claim(self):
        y = self.oof["death_5yr"].to_numpy()
        for m in MAIN:
            p = self.oof[m].to_numpy()
            with self.subTest(model=m):
                flag = p >= self.cv.loc[m, "Thr@PPV40"]
                self.assertGreaterEqual(y[flag].mean(), 0.40)
                self.assertAlmostEqual(y[flag].sum() / y.sum(), self.cv.loc[m, "Sens@PPV40"], places=10)
                flag = p >= self.cv.loc[m, "Thr@Sens85"]
                self.assertGreaterEqual(y[flag].sum() / y.sum(), 0.85)

    def test_models_beat_simple_baselines(self):
        y = self.oof["death_5yr"].to_numpy()
        X, _, _, _ = M.load()
        age_auc = roc_auc_score(y, X["RIDAGEYR"])
        null_brier = brier_score_loss(y, np.full(len(y), y.mean()))
        for m in MAIN:
            with self.subTest(model=m):
                self.assertGreater(self.cv.loc[m, "ROC-AUC"], age_auc + 0.03)
                self.assertLess(self.cv.loc[m, "Brier"], null_brier)

    def test_close_to_paper_table_2(self):
        for m, auc in PAPER_AUC.items():
            with self.subTest(model=m):
                self.assertAlmostEqual(self.cv.loc[m, "ROC-AUC"], auc, delta=0.015)

    def test_agreement_table_is_mean_of_folds(self):
        by = pd.read_csv(OUT / "shap_agreement_by_fold.csv")
        ag = pd.read_csv(OUT / "shap_agreement.csv")
        self.assertEqual(sorted(by["fold"].unique()), [1, 2, 3, 4, 5])
        mean = by.groupby(["Model A", "Model B"], sort=False)["Top-1 feature agreement"].mean()
        np.testing.assert_allclose(ag["Top-1 feature agreement"], mean.to_numpy(), rtol=1e-12)
        self.assertTrue(ag[["Top-1 feature agreement", "Top-5 overlap"]].stack().between(0, 1).all())

    def test_same_model_agrees_more_than_different_models(self):
        ag = pd.read_csv(OUT / "shap_agreement.csv")
        floor = ag.loc[ag["Type"] == "Same model, new random seed", "Top-1 feature agreement"].min()
        cross = ag.loc[ag["Type"] == "Different models", "Top-1 feature agreement"].max()
        self.assertGreater(floor, cross)

    def test_importance_table_complete(self):
        imp = pd.read_csv(OUT / "importance_all_measures.csv", index_col=0)
        self.assertEqual(len(imp), 134)
        for m in MAIN:
            for k in ("SHAP", "Permutation", "Built-in"):
                self.assertEqual(sorted(imp[f"{m} | {k} rank"]), sorted(
                    imp[f"{m} | {k}"].rank(ascending=False).astype(int)))
            self.assertAlmostEqual(imp[f"{m} | Built-in"].sum(), 1.0, places=9)
            self.assertTrue((imp[f"{m} | SHAP"] >= 0).all())

    def test_every_figure_exists_and_is_not_blank(self):
        from matplotlib.image import imread
        figs = sorted(OUT.glob("fig_*.png"))
        self.assertEqual(len(figs), 11)
        for f in figs:
            with self.subTest(figure=f.name):
                img = imread(f)[:, :, :3]
                self.assertGreater(min(img.shape[:2]), 500)
                self.assertGreater(img.std(), 0.02)  # not a single flat colour


REPRO = ROOT / "results" / "models_v0.2_reproduction"


@unittest.skipUnless((REPRO / "summary.json").exists() and (OUT / "summary.json").exists(),
                     "needs a second from-scratch run in results/models_v0.2_reproduction "
                     "(see the module docstring of this test)")
class TestReproducibility(unittest.TestCase):
    """Two complete runs of src/paper134_analysis.py, each from scratch, must agree.

    To refresh the second run:
        .venv/bin/python -c "import sys; sys.path.insert(0,'src'); sys.argv=['x']; \
import paper134_analysis as A; A.OUT=A.ROOT/'results'/'models_v0.2_reproduction'; A.main()"
    """

    def test_every_table_matches_between_two_runs(self):
        tables = sorted(OUT.glob("*.csv"))
        self.assertGreaterEqual(len(tables), 27)
        for f in tables:
            with self.subTest(table=f.name):
                a, b = pd.read_csv(f), pd.read_csv(REPRO / f.name)
                self.assertEqual(a.shape, b.shape)
                num = list(a.select_dtypes("number").columns)
                self.assertTrue(a.drop(columns=num).astype(str).equals(
                    b.drop(columns=num).astype(str)))
                # Parallel forest predictions can change permutation scores by
                # ~7e-8 through floating-point summation order. All other saved
                # measures agree exactly between these two full runs.
                np.testing.assert_allclose(a[num].to_numpy(float), b[num].to_numpy(float),
                                           rtol=1e-6, atol=1e-7)

    def test_summary_matches(self):
        with (OUT / "summary.json").open() as original, (REPRO / "summary.json").open() as rerun:
            self.assertEqual(json.load(original), json.load(rerun))


class TestImputationStability(unittest.TestCase):
    def test_imputed_values_do_not_react_to_negligible_changes(self):
        X, y, _, _ = M.load()
        tr, _ = next(StratifiedKFold(5, shuffle=True, random_state=M.SEED).split(X, y))
        tr = tr[:2500]
        A1 = X.iloc[tr]
        noise = 1 + 1e-10 * np.random.default_rng(0).standard_normal(A1.shape)
        cat = [c for c in A1.columns if c in M.CATEGORICAL]
        A2 = A1 * noise
        A2[cat] = A1[cat]
        cols = list(X.columns)
        r1 = M.Prep(cols).fit(A1, full=False).imputed(A1)[0]
        r2 = M.Prep(cols).fit(A2, full=False).imputed(A1)[0]
        num = [c for c in cols if c not in M.CATEGORICAL]
        spread = (A1[num].max() - A1[num].min()).replace(0, 1)
        change = ((r1[num] - r2[num]).abs() / spread).max().max()
        self.assertLess(change, 1e-4)  # the unstabilised imputer moved values by >50% of range
        self.assertTrue(np.isfinite(r1.to_numpy()).all())
        obs = A1[num].notna().to_numpy()
        np.testing.assert_array_equal(r1[num].to_numpy()[obs], A1[num].to_numpy()[obs])


class TestModelBehaviour(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y, _, _ = M.load()
        tr, te = next(StratifiedKFold(5, shuffle=True, random_state=M.SEED).split(X, y))
        prep = M.Prep(list(X.columns)).fit(X.iloc[tr], impute=False)
        cls.Xtr, cls.groups = prep.native(X.iloc[tr])
        cls.Xte = prep.native(X.iloc[te])[0].iloc[:300]
        cls.ytr, cls.feats = y.iloc[tr], list(X.columns)
        cls.model = M.fit_xgb(cls.Xtr, cls.ytr)

    def test_xgboost_training_is_repeatable(self):
        again = M.fit_xgb(self.Xtr, self.ytr)
        np.testing.assert_array_equal(self.model.predict_proba(self.Xte),
                                      again.predict_proba(self.Xte))

    def test_shap_values_add_up_to_the_prediction(self):
        import xgboost as xgb
        con = self.model.get_booster().predict(
            xgb.DMatrix(self.Xte), pred_contribs=True,
            iteration_range=(0, self.model.best_iteration + 1))
        shap_part = A.xgb_contribs(self.model, self.Xte)
        np.testing.assert_allclose(shap_part, con[:, :-1])
        logit = con[:, -1] + shap_part.sum(axis=1)
        np.testing.assert_allclose(1 / (1 + np.exp(-logit)),
                                   self.model.predict_proba(self.Xte)[:, 1], atol=1e-5)

    def test_one_hot_columns_are_summed_back_to_134_predictors(self):
        raw = A.xgb_contribs(self.model, self.Xte)
        agg = M.aggregate(raw, self.groups, self.feats)
        self.assertEqual(agg.shape, (len(self.Xte), 134))
        np.testing.assert_allclose(agg.sum(axis=1), raw.sum(axis=1), atol=1e-5)
        g = np.asarray(self.groups)
        j = self.feats.index("PAQ180")
        np.testing.assert_allclose(agg[:, j], raw[:, g == "PAQ180"].sum(axis=1), atol=1e-6)

    def test_permutation_importance_is_zero_for_an_unused_predictor(self):
        used = self.model.get_booster().get_score(importance_type="total_gain")
        g = np.asarray(self.groups)
        unused = [f for f in self.feats if not any(c in used for c in self.Xtr.columns[g == f])]
        if not unused:
            self.skipTest("model uses every predictor")
        y_te = M.load()[1].loc[self.Xte.index]
        pi = A.permutation_importance(self.model, self.Xte, y_te, self.groups, unused[:2],
                                      np.random.default_rng(0), 1)
        np.testing.assert_allclose(pi, 0.0, atol=1e-12)


class TestFigureCode(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = A.OUT
        A.OUT = Path(self.tmp.name)

    def tearDown(self):
        A.OUT = self.old
        self.tmp.cleanup()

    def test_agreement_figure_draws_the_numbers_it_is_given(self):
        import matplotlib.pyplot as plt
        agg = pd.DataFrame({
            "Model A": ["XGBoost", "Random Forest", "XGBoost"],
            "Model B": ["LightGBM", "XGBoost", "XGBoost (seed 2)"],
            "Type": ["Different models", "Different models", "Same model, new random seed"],
            "Top-1 feature agreement": [0.40, 0.55, 0.93],
            "Top-1 SD across folds": [0.05, 0.04, 0.02]})
        captured = {}
        orig = plt.close
        plt.close = lambda fig=None: captured.setdefault("fig", fig)
        try:
            A.fig_agreement(agg)
        finally:
            plt.close = orig
        ax = captured["fig"].axes[0]
        widths = sorted(round(p.get_width(), 6) for p in ax.patches)
        self.assertEqual(widths, [40.0, 55.0, 93.0])
        self.assertEqual([t.get_text() for t in ax.get_yticklabels()],
                         ["XGBoost vs LightGBM", "RF vs XGBoost", "XGBoost vs XGBoost (seed 2)"])
        self.assertGreater((A.OUT / "fig_shap_agreement.png").stat().st_size, 10_000)
        orig(captured["fig"])

    def test_survival_curves_end_at_the_table_values(self):
        t = np.array([10, 20, 60, 60, 60, 30, 60, 60], float)
        e = np.array([1, 1, 0, 0, 0, 1, 0, 0])
        _, s = M.kaplan_meier(t, e)
        self.assertAlmostEqual(s[-1], 1 - e.mean())  # no censoring before 60 months
        cause = np.where(e == 1, [1, 2, 0, 0, 0, 1, 0, 0], 0)
        _, f = A.aalen_johansen(t, cause, 1)
        self.assertAlmostEqual(f[-1], (cause == 1).mean())


@unittest.skipUnless((ROOT / "results" / "nhanes_master_v1.pkl").exists(), "master dataset needed")
class TestMedicationFlags(unittest.TestCase):
    """Rebuild the three medication flags from the raw prescription files, independently
    of notebook 02, and compare with the values used as predictors."""

    PATTERNS = {
        "DERIVED_HL_MED": r"ANTIHYPERLIPIDEMIC|CHOLESTEROL ABSORPTION|HMG.COA REDUCTASE|"
                          r"FIBRIC ACID|BILE ACID",
        "DERIVED_HTN_MED": r"ANTIHYPERTENSIVE|ACE INHIBITOR|ANGIOTENSIN|BETA.*BLOCK|"
                           r"CALCIUM CHANNEL BLOCK|DIURETIC|RENIN INHIBITOR",
        "DERIVED_DM_MED": r"ANTIDIABETIC|INSULIN|BIGUANIDE|SULFONYLUREA|THIAZOLIDINEDIONE"}

    def test_flags_rebuilt_from_prescription_files(self):
        raw = ROOT / "data" / "raw"
        X, _, _, _ = M.load()

        def dec(s):
            return s.map(lambda v: v.decode("utf-8", "replace").strip() if isinstance(v, bytes) else v)

        drug = pd.read_sas(raw / "RXQ_DRUG.xpt", format="xport")
        drug["RXDDRGID"] = dec(drug["RXDDRGID"]).str.lower()
        classes = drug[[c for c in drug.columns if c.startswith(("RXDDCN", "RXDICN"))]].apply(dec)
        rx = []
        for f, use in [("RXQ_RX", "RXD030"), ("RXQ_RX_B", "RXD030"), ("RXQ_RX_C", "RXDUSE")]:
            d = pd.read_sas(raw / f"{f}.xpt", format="xport")[["SEQN", use, "RXDDRGID"]]
            d = d.rename(columns={use: "use"})
            d["RXDDRGID"] = dec(d["RXDDRGID"]).str.lower()
            rx.append(d)
        rx = pd.concat(rx)
        rx = rx[rx["SEQN"].isin(X.index)]
        use = rx.groupby("SEQN")["use"].first()
        # someone who reports a prescription that cannot be identified has unknown status
        known = set(drug["RXDDRGID"])
        unidentified = rx.assign(u=~rx["RXDDRGID"].isin(known) & (rx["use"] == 1)
                                 ).groupby("SEQN")["u"].any()
        for flag, pattern in self.PATTERNS.items():
            ids = set(drug.loc[classes.apply(lambda c: c.astype(str).str.contains(
                pattern, case=False, regex=True)).any(axis=1), "RXDDRGID"])
            takes = rx.assign(h=rx["RXDDRGID"].isin(ids)).groupby("SEQN")["h"].any()
            rebuilt = pd.Series(np.where(takes, 1.0, np.where(
                ~use.isin([1, 2]) | unidentified, np.nan, 0.0)), index=takes.index).reindex(X.index)
            self.assertEqual(int(rebuilt.isna().sum()), int(X[flag].isna().sum()))
            same = (rebuilt == X[flag]) | (rebuilt.isna() & X[flag].isna())
            with self.subTest(flag=flag):
                # the blood-pressure list in notebook 02 was hand-picked and omits one
                # drug class that the pattern catches, affecting one participant
                self.assertLessEqual(int((~same).sum()), 1 if flag == "DERIVED_HTN_MED" else 0)


if __name__ == "__main__":
    unittest.main()
