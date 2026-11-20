"""Sensitivity of the systematic parsimonious-model procedure to its thresholds.

src/paper134_parsimony.py uses fixed thresholds. Some come from the paper (top 10, 2,000
people for SHAP) or standard practice (one-standard-error rule); others are our choices.
This script changes one threshold at a time, keeping the others at their original values,
and records which predictors are chosen and how well they predict.

Efficiency: step 1 (SHAP rankings) does not depend on the thresholds, so it is computed once
per dataset (all participants, and the training part of each of the 5 outer folds). The inner
cross-validated AUC of any predictor set is cached, so each set is fitted only once.

Usage: .venv/bin/python src/paper134_sensitivity.py
Output: results/parsimony_v0.1/sensitivity.csv, sensitivity_summary.csv
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper134_models as M  # noqa: E402
import paper134_parsimony as P  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

OUT = M.ROOT / "results" / "parsimony_v0.1"
BASE = {"top_k": 10, "min_folds": 4, "subgroup_max_rank": 15, "min_observed": 0.50, "stop": "1se"}
VARIANTS = [("top_k", v) for v in (5, 8, 12, 15)] + \
           [("min_folds", v) for v in (3, 5)] + \
           [("subgroup_max_rank", v) for v in (10, 20, 30, np.inf)] + \
           [("min_observed", v) for v in (0.30, 0.70, 0.80)] + \
           [("stop", v) for v in ("best", "2se", "gain0.005")]


def screen(X, y, feats):
    """Step 1: per-inner-fold SHAP ranks overall and per subgroup (as in paper134_parsimony)."""
    inner = list(StratifiedKFold(5, shuffle=True, random_state=P.INNER_SEED).split(X, y))
    groups = P.subgroups(X)
    ranks, sub = [], {g: [] for g in groups}
    for tr, _ in inner:
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        prep = M.Prep(feats).fit(Xtr, impute=False)
        mat, g = prep.native(Xtr)
        model = M.fit_xgb(mat, ytr)
        samp = mat.sample(min(P.SHAP_SAMPLE, len(mat)), random_state=M.SEED)
        v = np.abs(M.aggregate(P.A.xgb_contribs(model, samp), g, feats))
        ranks.append(pd.Series(v.mean(0), index=feats).rank(ascending=False))
        for name, mask in groups.items():
            k = mask.loc[samp.index].to_numpy()
            sub[name].append(pd.Series(v[k].mean(0), index=feats).rank(ascending=False))
    R = pd.concat(ranks, axis=1)
    worst = pd.concat([pd.concat(sub[n], axis=1).mean(axis=1) for n in groups], axis=1).max(axis=1)
    return R, worst, X[feats].notna().mean(), inner


class Chooser:
    def __init__(self, X, y, feats):
        self.X, self.y = X, y
        self.R, self.worst, self.observed, self.inner = screen(X, y, feats)
        self.cache = {}

    def auc(self, cols):
        key = tuple(cols)
        if key not in self.cache:
            a = [P.xgb_auc(self.X.iloc[tr], self.y.iloc[tr], self.X.iloc[te], self.y.iloc[te], list(cols))
                 for tr, te in self.inner]
            self.cache[key] = (np.mean(a), np.std(a, ddof=1) / np.sqrt(len(a)))
        return self.cache[key]

    def choose(self, s):
        stable = ((self.R <= s["top_k"]).sum(axis=1) >= s["min_folds"]) & \
                 (self.worst <= s["subgroup_max_rank"])
        ok = stable & (self.observed >= s["min_observed"])
        cands = list(self.R.mean(axis=1)[ok].sort_values().index)[:P.MAX_CANDIDATES]
        if not cands:
            return []
        curve = [self.auc(cands[:k]) for k in range(1, len(cands) + 1)]
        m = np.array([c[0] for c in curve])
        se = np.array([c[1] for c in curve])
        b = int(m.argmax())
        if s["stop"] == "best":
            k = b + 1
        elif s["stop"] in ("1se", "2se"):
            mult = 1 if s["stop"] == "1se" else 2
            k = int(np.where(m >= m[b] - mult * se[b])[0].min()) + 1
        else:  # add while each new predictor raises AUC by >= 0.005
            k = 1
            while k < len(m) and m[k] - m[k - 1] >= 0.005:
                k += 1
        return cands[:k]


def main():
    X, y, _, names = M.load()
    feats = list(X.columns)
    print("Screening all participants", flush=True)
    full = Chooser(X, y, feats)
    outer = list(StratifiedKFold(5, shuffle=True, random_state=P.OUTER_SEED).split(X, y))
    folds = []
    for f, (tr, te) in enumerate(outer, start=1):
        print(f"Screening outer fold {f}/5", flush=True)
        folds.append((Chooser(X.iloc[tr], y.iloc[tr], feats), tr, te))
    test_cache = {}

    def outer_auc(f, cols, tr, te):
        key = (f, tuple(cols))
        if key not in test_cache:
            test_cache[key] = P.xgb_auc(X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te], list(cols)) \
                if cols else float("nan")
        return test_cache[key]

    rows = []
    for setting, value in [("original", None)] + VARIANTS:
        s = dict(BASE)
        if setting != "original":
            s[setting] = value
        chosen = full.choose(s)
        fold_sets = [c.choose(s) for c, _, _ in folds]
        aucs = [outer_auc(f, cs, tr, te) for f, (cs, (_, tr, te)) in enumerate(zip(fold_sets, folds))]
        freq = pd.Series([c for cs in fold_sets for c in cs]).value_counts()
        rows.append({"Setting changed": setting,
                     "Value": "original" if setting == "original" else
                              ("no subgroup rule" if value == np.inf else value),
                     "Chosen (all participants)": ", ".join(names[c] for c in chosen),
                     "Size": len(chosen),
                     "Same as original": None,
                     "Sizes in outer folds": [len(c) for c in fold_sets],
                     "Nested outer AUC": float(np.nanmean(aucs)),
                     **{f"folds: {names[c]}": int(freq.get(c, 0))
                        for c in ["RIDAGEYR", "SSTNT", "PAQ180", "SSBNP", "LBXRDW", "RIAGENDR",
                                  "BMXCALF", "LBXBCD"]},
                     "_set": frozenset(chosen)})
        print(f"{setting:18s} {str(rows[-1]['Value']):18s} -> {rows[-1]['Chosen (all participants)']}"
              f" | nested AUC {rows[-1]['Nested outer AUC']:.4f}", flush=True)
    t = pd.DataFrame(rows)
    base = t.loc[0, "_set"]
    t["Same as original"] = t["_set"] == base
    t.drop(columns="_set").to_csv(OUT / "sensitivity.csv", index=False)

    counts = pd.Series([c for s in t["_set"] for c in s]).value_counts()
    summ = pd.DataFrame({"Predictor": [names[c] for c in counts.index],
                         "Chosen in N of %d settings" % len(t): counts.values,
                         "In paper's five": [c in M.FIVE for c in counts.index]})
    summ.to_csv(OUT / "sensitivity_summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 100)
    print("\n", t.drop(columns="_set").round(4).to_string(index=False))
    print("\n", summ.to_string(index=False))
    print(f"\nSettings giving exactly the original set: {int(t['Same as original'].sum())} of {len(t)}")


if __name__ == "__main__":
    main()
