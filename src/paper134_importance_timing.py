"""Time feature-importance calculations under identical conditions.

Models are trained once on the training part of outer fold 1 (7,965 people; same folds as the
main analysis). Each importance calculation is then timed 3 times and the median reported.

  XGBoost        built-in total gain | TreeSHAP on 2,000 people (paper 2.6) | TreeSHAP on all
                 training people | permutation (134 predictors, held-out fold, 1 and 5 repeats)
                 | KernelSHAP (model-agnostic) on 20 people with a 100-person background
  Random Forest  built-in impurity | TreeSHAP on 2,000 people | permutation (1 repeat)

Random Forest uses median-imputed inputs here (timing does not depend on how values were filled).
Run on an otherwise idle machine; other jobs distort timings.

Usage: .venv/bin/python src/paper134_importance_timing.py
Output: results/parsimony_v0.1/importance_timing.csv
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper134_models as M  # noqa: E402
import paper134_analysis as A  # noqa: E402
import paper134_parsimony as P  # noqa: E402
import shap  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

OUT = M.ROOT / "results" / "parsimony_v0.1" / "importance_timing.csv"


def timed(fn, repeats=3):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main():
    X, y, _, _ = M.load()
    feats = list(X.columns)
    tr, te = next(StratifiedKFold(5, shuffle=True, random_state=M.SEED).split(X, y))
    prep = M.Prep(feats).fit(X.iloc[tr], impute=False)
    (a, g), (b, _) = prep.native(X.iloc[tr]), prep.native(X.iloc[te])
    ytr, yte = y.iloc[tr], y.iloc[te]
    rows = []

    def add(model, method, people, fn, repeats=3):
        s = timed(fn, repeats)
        rows.append({"Model": model, "Method": method, "People explained / scored": people,
                     "Seconds": s, "Timed runs (median)": repeats})
        print(f"{model:13s} {method:48s} {s:9.3f} s", flush=True)

    xgb = M.fit_xgb(a, ytr)
    samp = a.sample(2000, random_state=M.SEED)
    add("XGBoost", "Built-in total gain", "-", lambda: P.gain(xgb, list(a.columns), g, feats))
    add("XGBoost", "TreeSHAP", 2000, lambda: A.xgb_contribs(xgb, samp))
    add("XGBoost", "TreeSHAP", len(a), lambda: A.xgb_contribs(xgb, a))
    rng = np.random.default_rng(0)
    add("XGBoost", "Permutation, 1 shuffle per predictor", len(b),
        lambda: P.permutation_drops(xgb, b, yte, g, feats, {}, rng))
    add("XGBoost", "Permutation, 5 shuffles per predictor", len(b),
        lambda: A.permutation_importance(xgb, b, yte, g, feats, rng, 5), repeats=1)
    background = shap.sample(a.fillna(a.median()), 100, random_state=0)
    kern = shap.KernelExplainer(lambda z: xgb.predict_proba(pd.DataFrame(z, columns=a.columns))[:, 1],
                                background)
    people = b.fillna(a.median()).iloc[:20]
    add("XGBoost", "KernelSHAP (model-agnostic), 100 background", 20,
        lambda: kern.shap_values(people, nsamples=500, silent=True), repeats=1)

    imp = SimpleImputer(strategy="median").fit(a)
    ai = pd.DataFrame(imp.transform(a), columns=a.columns, index=a.index)
    bi = pd.DataFrame(imp.transform(b), columns=b.columns, index=b.index)
    rf = M.fit_rf(ai, ytr)
    add("Random Forest", "Built-in impurity", "-", lambda: rf.feature_importances_)
    add("Random Forest", "TreeSHAP", 2000,
        lambda: shap.TreeExplainer(rf).shap_values(ai.loc[samp.index], check_additivity=False),
        repeats=1)
    add("Random Forest", "Permutation, 1 shuffle per predictor", len(bi),
        lambda: P.permutation_drops(rf, bi, yte, g, feats, {}, rng), repeats=1)

    t = pd.DataFrame(rows)
    for model in t["Model"].unique():
        ref = t.loc[(t["Model"] == model) & (t["Method"] == "TreeSHAP")
                    & (t["People explained / scored"] == 2000), "Seconds"].iloc[0]
        t.loc[t["Model"] == model, "Relative to TreeSHAP on 2,000 people"] = t["Seconds"] / ref
    t.to_csv(OUT, index=False)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
