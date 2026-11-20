"""A systematic, pre-specified procedure for choosing a parsimonious predictor set.

Lin et al. (2026, Methods 2.8) chose five predictors by "jointly considering (i) consistent
importance signals across subgroups, (ii) real-world data availability and feasibility for
implementation, and (iii) parsimony", with SHAP used "to inform ... rather than as the sole
automated selection rule". They give no algorithm. This script turns those three criteria into
fixed rules, decided before looking at the results, and runs them so that the data used to test
a predictor set never influence its choice (nested cross-validation).

Procedure, run inside the training part of each outer fold:
  1. Screen      XGBoost in 5 inner folds; rank all 134 predictors by mean |SHAP| computed on
                 inner-training data only (up to 2,000 people per fold, as in paper 2.6).
  2. Stability   keep a predictor if it is in the top TOP_K in at least MIN_FOLDS inner folds
                 AND its mean rank is <= SUBGROUP_MAX_RANK in every subgroup
                 (men, women, age < 65, age >= 65, prior CVD, no prior CVD).
  3. Availability keep a predictor if it is measured in >= MIN_OBSERVED of the training data.
  4. Parsimony   order the survivors by mean rank and add them one at a time; inner-CV ROC-AUC
                 for each size; choose the smallest set whose AUC is within one standard error
                 of the best (the "one-standard-error rule").
The chosen set is then fitted on the whole outer-training part and scored on the outer test
part. The outer folds are the same as in the main analysis.

Comparator: L1-penalised (LASSO) logistic regression, penalty chosen by inner CV with the same
one-standard-error rule, on median-imputed standardised inputs. It answers whether a classical
selection method, not based on SHAP, reaches a similar set.

The final recommended set is obtained by running steps 1-4 once on all participants; the
outer-fold selections show how stable that choice is.

Usage: .venv/bin/python src/paper134_parsimony.py [--dataset full|replication]
Output: results/parsimony_v0.1[_replication]/
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import paper134_models as M  # noqa: E402
import paper134_analysis as A  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Pre-specified rules (fixed before running; see the docstring)
TOP_K = 10               # "top ten" as in paper 2.6
MIN_FOLDS = 4            # in the top TOP_K in at least 4 of 5 inner folds
SUBGROUP_MAX_RANK = 15   # mean rank at most 15 in every subgroup
MIN_OBSERVED = 0.50      # measured in at least half of the training data
MAX_CANDIDATES = 15      # forward selection considers at most this many predictors
SHAP_SAMPLE = 2000       # paper 2.6
INNER_SEED, OUTER_SEED = 43, M.SEED


def subgroups(X):
    cvd = (X[["MCQ160B", "MCQ160E", "MCQ160F"]] == 1).any(axis=1)
    return {"Men": X["RIAGENDR"] == 1, "Women": X["RIAGENDR"] == 2,
            "Age < 65": X["RIDAGEYR"] < 65, "Age >= 65": X["RIDAGEYR"] >= 65,
            "Prior CVD": cvd, "No prior CVD": ~cvd}


def xgb_auc(Xtr, ytr, Xte, yte, cols):
    prep = M.Prep(cols).fit(Xtr[cols], impute=False)
    model = M.fit_xgb(prep.native(Xtr[cols])[0], ytr)
    return roc_auc_score(yte, model.predict_proba(prep.native(Xte[cols])[0])[:, 1])


def gain(model, cols, groups, feats):
    sc = model.get_booster().get_score(importance_type="total_gain")
    return pd.Series([sc.get(c, 0.0) for c in cols]).groupby(np.asarray(groups)).sum()[feats].to_numpy()


def permutation_drops(model, Xte, yte, groups, feats, masks, rng):
    """AUC drop when one original predictor is shuffled (one-hot columns together), overall
    and inside each subgroup, from the same shuffled predictions."""
    arr, g = Xte.to_numpy(copy=True), np.asarray(groups)
    yv = np.asarray(yte)
    base = model.predict_proba(Xte)[:, 1]
    keys = ["All"] + list(masks)
    rows = {k: np.asarray(masks[k]) if k != "All" else np.ones(len(yv), bool) for k in keys}
    base_auc = {k: roc_auc_score(yv[rows[k]], base[rows[k]]) for k in keys}
    out = {k: [] for k in keys}
    for f in feats:
        idx = np.where(g == f)[0]
        saved = arr[:, idx].copy()
        arr[:, idx] = saved[rng.permutation(len(arr))]
        p = model.predict_proba(pd.DataFrame(arr, columns=Xte.columns))[:, 1]
        arr[:, idx] = saved
        for k in keys:
            out[k].append(base_auc[k] - roc_auc_score(yv[rows[k]], p[rows[k]]))
    return {k: np.asarray(v) for k, v in out.items()}


def select(X, y, feats, log=None, method="shap", subgroup_rule=True):
    """Steps 1-4 on the data given (which must be training data only).

    method: importance measure for step 1.
      shap         mean |SHAP| (TreeSHAP) on up to 2,000 inner-training people; subgroups from
                   the same SHAP values.
      gain         XGBoost total gain (recorded during training); subgroups from a separate
                   model fitted on each subgroup's inner-training data.
      permutation  drop in ROC-AUC on the inner held-out part when a predictor is shuffled;
                   subgroups from the same shuffled predictions.
    """
    import time
    inner = list(StratifiedKFold(5, shuffle=True, random_state=INNER_SEED).split(X, y))
    groups = subgroups(X)
    ranks, sub_ranks = [], {g: [] for g in groups}
    seconds = 0.0
    for fi, (tr, te) in enumerate(inner):                 # step 1
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        prep = M.Prep(feats).fit(Xtr, impute=False)
        mat, g = prep.native(Xtr)
        model = M.fit_xgb(mat, ytr)
        t0 = time.perf_counter()
        if method == "shap":
            samp = mat.sample(min(SHAP_SAMPLE, len(mat)), random_state=M.SEED)
            v = np.abs(M.aggregate(A.xgb_contribs(model, samp), g, feats))
            overall = v.mean(0)
            sub = {name: v[mask.loc[samp.index].to_numpy()].mean(0) for name, mask in groups.items()}
        elif method == "gain":
            overall = gain(model, list(mat.columns), g, feats)
            sub = {}
            for name, mask in groups.items():
                k = mask.loc[Xtr.index].to_numpy()
                sub[name] = gain(M.fit_xgb(mat[k], ytr[k]), list(mat.columns), g, feats)
        elif method == "permutation":
            Xte, yte = X.iloc[te], y.iloc[te]
            mte = prep.native(Xte)[0]
            masks = {name: mask.loc[Xte.index].to_numpy() for name, mask in groups.items()}
            d = permutation_drops(model, mte, yte, g, feats, masks, np.random.default_rng(M.SEED + fi))
            overall, sub = d["All"], {k: d[k] for k in groups}
        else:
            raise ValueError(method)
        seconds += time.perf_counter() - t0
        ranks.append(pd.Series(overall, index=feats).rank(ascending=False))
        for name in groups:
            sub_ranks[name].append(pd.Series(sub[name], index=feats).rank(ascending=False))
    R = pd.concat(ranks, axis=1)
    table = pd.DataFrame({"mean rank": R.mean(axis=1),
                          "folds in top %d" % TOP_K: (R <= TOP_K).sum(axis=1)})
    for name in groups:
        table[f"rank: {name}"] = pd.concat(sub_ranks[name], axis=1).mean(axis=1)
    table["worst subgroup rank"] = table[[f"rank: {n}" for n in groups]].max(axis=1)
    table["observed share"] = X[feats].notna().mean()
    table["stable"] = (table[f"folds in top {TOP_K}"] >= MIN_FOLDS) & \
                      ((table["worst subgroup rank"] <= SUBGROUP_MAX_RANK) | (not subgroup_rule))  # step 2
    table["available"] = table["observed share"] >= MIN_OBSERVED                 # step 3
    table["candidate"] = table["stable"] & table["available"]
    table = table.sort_values("mean rank")
    cands = list(table.index[table["candidate"]])[:MAX_CANDIDATES]

    curve = []                                            # step 4
    for k in range(1, len(cands) + 1):
        aucs = [xgb_auc(X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te], cands[:k])
                for tr, te in inner]
        curve.append({"size": k, "added": cands[k - 1], "inner AUC": np.mean(aucs),
                      "inner AUC SE": np.std(aucs, ddof=1) / np.sqrt(len(aucs))})
        if log:
            log(f"      size {k:2d} +{cands[k - 1]:<10s} inner AUC {np.mean(aucs):.4f}")
    curve = pd.DataFrame(curve)
    table.attrs["importance_seconds"] = seconds
    if not cands:                                         # nothing passed steps 2-3
        return [], table, curve, float("nan")
    best = curve["inner AUC"].idxmax()
    threshold = curve.loc[best, "inner AUC"] - curve.loc[best, "inner AUC SE"]
    size = int(curve.loc[curve["inner AUC"] >= threshold, "size"].min())
    table.attrs["importance_seconds"] = seconds
    return cands[:size], table, curve, threshold


def lasso_select(X, y, feats):
    """Comparator: L1 logistic regression, penalty by inner CV with the 1-SE rule."""
    inner = list(StratifiedKFold(5, shuffle=True, random_state=INNER_SEED).split(X, y))
    Cs = np.logspace(-3.5, 0, 15)
    scores = np.zeros((len(inner), len(Cs)))

    def design(Xa, Xb):
        prep = M.Prep(feats).fit(Xa, impute=False)
        a, g = prep.native(Xa)
        b, _ = prep.native(Xb)
        imp = SimpleImputer(strategy="median").fit(a)
        sc = StandardScaler().fit(imp.transform(a))
        return sc.transform(imp.transform(a)), sc.transform(imp.transform(b)), np.asarray(g)

    for i, (tr, te) in enumerate(inner):
        a, b, _ = design(X.iloc[tr], X.iloc[te])
        for j, C in enumerate(Cs):
            m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000)
            scores[i, j] = roc_auc_score(y.iloc[te], m.fit(a, y.iloc[tr]).predict_proba(b)[:, 1])
    mean, se = scores.mean(0), scores.std(0, ddof=1) / np.sqrt(len(inner))
    best = mean.argmax()
    C = Cs[np.where(mean >= mean[best] - se[best])[0].min()]   # strongest penalty within 1 SE
    a, _, g = design(X, X)
    coef = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000
                              ).fit(a, y).coef_[0]
    chosen = sorted({g[j] for j in np.where(coef != 0)[0]},
                    key=lambda f: -np.abs(coef[g == f]).max())
    return chosen, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["full", "replication"], default="full")
    ap.add_argument("--importance", choices=["shap", "gain", "permutation"], default="shap",
                    help="importance measure for step 1 (default shap)")
    ap.add_argument("--skip-lasso", action="store_true", help="skip the LASSO comparator")
    ap.add_argument("--no-subgroup-rule", action="store_true",
                    help="step 2 uses fold stability only (for comparing importance measures)")
    args = ap.parse_args()
    out = M.ROOT / "results" / ("parsimony_v0.1" + ("_replication" if args.dataset == "replication" else "")
                                + ("" if args.importance == "shap" else f"_{args.importance}")
                                + ("_nosubgroup" if args.no_subgroup_rule else ""))
    if args.dataset == "replication":
        M.CAND = M.REPLICATION
    out.mkdir(parents=True, exist_ok=True)
    X, y, cyc, names = M.load()
    feats = list(X.columns)
    log = lambda s: print(s, flush=True)  # noqa: E731

    outer = list(StratifiedKFold(5, shuffle=True, random_state=OUTER_SEED).split(X, y))
    rows, curves, lasso_rows = [], [], []
    for f, (tr, te) in enumerate(outer, start=1):
        log(f"Outer fold {f}/5")
        Xtr, ytr, Xte, yte = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]
        chosen, table, curve, thr = select(Xtr, ytr, feats, log, method=args.importance,
                                           subgroup_rule=not args.no_subgroup_rule)
        curves.append(curve.assign(fold=f, threshold=thr))
        lasso, C = ([], float("nan")) if args.skip_lasso else lasso_select(Xtr, ytr, feats)
        rows.append({"fold": f, "selected": ", ".join(names[c] for c in chosen),
                     "size": len(chosen),
                     "outer AUC (selected)": (xgb_auc(Xtr, ytr, Xte, yte, chosen) if chosen
                                              else float("nan")),
                     "outer AUC (paper's five)": xgb_auc(Xtr, ytr, Xte, yte, M.FIVE),
                     "outer AUC (all 134)": xgb_auc(Xtr, ytr, Xte, yte, feats),
                     "outer AUC (age only)": roc_auc_score(yte, Xte["RIDAGEYR"]),
                     "candidates after steps 1-3": int(table["candidate"].sum()),
                     "importance seconds (5 inner folds)": table.attrs["importance_seconds"],
                     "_cols": chosen})
        lasso_rows.append({"fold": f, "LASSO size": len(lasso), "LASSO C": C,
                           "LASSO selected": ", ".join(names[c] for c in lasso),
                           "outer AUC (LASSO set, XGBoost)": (xgb_auc(Xtr, ytr, Xte, yte, lasso)
                                                              if lasso else float("nan")),
                           "_cols": lasso})
        log(f"   chose {len(chosen)}: {[names[c] for c in chosen]} | LASSO {len(lasso)}")

    folds = pd.DataFrame(rows)
    lasso = pd.DataFrame(lasso_rows)
    freq = pd.Series([c for r in rows for c in r["_cols"]]).value_counts()
    freq_l = pd.Series([c for r in lasso_rows for c in r["_cols"]]).value_counts()
    freq_tab = pd.DataFrame({"Paper variable": [names[c] for c in freq.index],
                             "outer folds selected (of 5)": freq.values,
                             "in paper's five": [c in M.FIVE for c in freq.index]},
                            index=freq.index)
    freq_tab["outer folds selected by LASSO"] = freq_l.reindex(freq.index).fillna(0).astype(int)

    log("Final selection on all participants")
    final, table, curve, thr = select(X, y, feats, log, method=args.importance,
                                      subgroup_rule=not args.no_subgroup_rule)
    final_lasso, C_final = ([], float("nan")) if args.skip_lasso else lasso_select(X, y, feats)
    table.insert(0, "Paper variable", [names[c] for c in table.index])
    table.to_csv(out / "screening_table_all_participants.csv")
    curve.assign(threshold=thr).to_csv(out / "forward_curve_all_participants.csv", index=False)
    pd.concat(curves).to_csv(out / "forward_curves_outer_folds.csv", index=False)
    folds.drop(columns="_cols").to_csv(out / "outer_folds.csv", index=False)
    lasso.drop(columns="_cols").to_csv(out / "lasso_outer_folds.csv", index=False)
    freq_tab.to_csv(out / "selection_frequency.csv")

    folds = folds.drop(columns=[])
    summary = {k: float(folds[k].mean()) for k in folds if k.startswith("outer AUC")}
    summary.update({k + " SD": float(folds[k].std()) for k in folds if k.startswith("outer AUC")})
    summary["outer AUC (LASSO set, XGBoost)"] = float(lasso["outer AUC (LASSO set, XGBoost)"].mean())
    json.dump({"dataset": M.CAND.name, "importance": args.importance,
               "subgroup_rule": not args.no_subgroup_rule,
               "importance_seconds_outer_folds": folds["importance seconds (5 inner folds)"].tolist(),
               "rules": {"TOP_K": TOP_K, "MIN_FOLDS": MIN_FOLDS,
                         "SUBGROUP_MAX_RANK": SUBGROUP_MAX_RANK, "MIN_OBSERVED": MIN_OBSERVED,
                         "MAX_CANDIDATES": MAX_CANDIDATES, "stopping": "one-standard-error rule"},
               "final_selected": final, "final_selected_names": [names[c] for c in final],
               "final_lasso": final_lasso, "final_lasso_names": [names[c] for c in final_lasso],
               "final_lasso_C": C_final, "papers_five": M.FIVE, "outer_cv": summary},
              open(out / "summary.json", "w"), indent=2)

    if len(curve):
        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.errorbar(curve["size"], curve["inner AUC"], yerr=curve["inner AUC SE"], color=M.COL["XGBoost"],
                    marker="o", markersize=5, capsize=3, lw=2)
        ax.axhline(thr, color=M.MUTED, lw=1, ls="--")
        ax.axvline(len(final), color=M.MUTED, lw=1)
        ax.text(len(final) + 0.15, curve["inner AUC"].min(), f"chosen: {len(final)} predictors",
                color=M.INK2, fontsize=9)
        ax.set_xticks(curve["size"], [names[c] for c in curve["added"]], rotation=40, ha="right")
        ax.set(ylabel="ROC-AUC (inner 5-fold CV)",
               title=f"Adding predictors in order of {dict(shap='SHAP', gain='built-in gain', permutation='permutation')[args.importance]} importance (all participants)")
        ax.grid(axis="x", visible=False)
        fig.text(0.01, 0.01, "Dashed line: best AUC minus one standard error. The smallest set above it "
                 "is chosen.", color=M.MUTED, fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(out / "fig_forward_selection.png", dpi=160)
        plt.close(fig)

    pd.set_option("display.width", 220, "display.max_colwidth", 120)
    print("\nOUTER FOLDS\n", folds.drop(columns="_cols").round(4).to_string(index=False))
    print("\nLASSO\n", lasso.drop(columns="_cols").round(4).to_string(index=False))
    print("\nSELECTION FREQUENCY\n", freq_tab.to_string())
    print("\nFINAL:", [names[c] for c in final], "| LASSO final:", [names[c] for c in final_lasso])
    print("\nFORWARD CURVE (all participants)\n", curve.round(4).to_string(index=False), "\nthreshold", round(thr, 4))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
