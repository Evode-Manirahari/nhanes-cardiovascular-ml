"""Full analysis (v0.2) on the 134-predictor candidate dataset, following Lin et al. 2026.

What this adds to src/paper134_models.py (v0.1):
  2.3  preprocessing as in the paper: full iterative imputation inside each fold;
       sensitivity checks for the imputer shortcut, LR scaling, provisional definitions
  2.5  survey-weighted calibration next to the unweighted metrics
  2.6  SHAP inside the training folds (2,000 people per fold), subgroup rankings,
       top-ten correlation clustering, TNT x NT-proBNP interaction, dependence plot
  2.7  cardiovascular / non-cardiovascular death with competing-risk cumulative incidence
  2.8  simpler model chosen from feature importance; final model with a 70/30 split,
       early stopping, class weighting and isotonic recalibration
  2.9  NRI and IDI against the benchmark models
  +    importance beyond SHAP: permutation importance, built-in model importance,
       partial dependence
  +    explanation agreement across models on all five folds, with noise floors

Input : results/paper134_candidate_v0.1.pkl, data/processed/cohort_9957.csv (weights),
        data/raw/mortality/*.dat (follow-up time, cause of death) - all read only
Output: results/models_v0.2/

Usage: .venv/bin/python src/paper134_analysis.py          (about 25 minutes)
       .venv/bin/python src/paper134_analysis.py --fast   (quick check, approximate)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import paper134_models as M  # noqa: E402
from paper134_models import (COL, FIVE, INK2, MUTED, SEED, LR, Prep, aggregate,  # noqa: E402
                             fit_lgb, fit_rf, fit_xgb, kaplan_meier, logrank, metrics)
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, brier_score_loss,  # noqa: E402
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402
import shap  # noqa: E402
import xgboost as xgb  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

ROOT = M.ROOT
OUT = ROOT / "results" / "models_v0.2"
MAIN = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
# Paper 2.3: only XGBoost receives missing values as NaN; "the remaining algorithms"
# (GLM, Random Forest, LightGBM) receive the imputed table.
KIND = {"Logistic Regression": "imp", "Random Forest": "imp", "XGBoost": "nat",
        "LightGBM": "imp", "XGBoost (seed 2)": "nat", "Random Forest (seed 2)": "imp",
        "XGBoost (imputed inputs)": "imp"}
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
DIVERGING = LinearSegmentedColormap.from_list("div", [BLUE, "#f0efec", ORANGE])
SEQ_BLUE = LinearSegmentedColormap.from_list("seq", ["#86b6ef", "#0d366b"])


# ----------------------------------------------------------------------------
# Small statistics
# ----------------------------------------------------------------------------
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_w(y, p, w=None):
    """Calibration slope and calibration-in-the-large, optionally survey-weighted."""
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    k = w > 0
    y, lp, w = y[k], logit(np.asarray(p)[k]), w[k] / w[k].mean()
    slope = LogisticRegression(C=1e9, max_iter=2000).fit(lp.reshape(-1, 1), y,
                                                         sample_weight=w).coef_[0, 0]
    a = 0.0
    for _ in range(100):
        q = 1 / (1 + np.exp(-(lp + a)))
        a += (w * (y - q)).sum() / (w * q * (1 - q)).sum()
    return slope, a


def brier_w(y, p, w):
    k = np.asarray(w) > 0
    return brier_score_loss(np.asarray(y)[k], np.asarray(p)[k], sample_weight=np.asarray(w)[k])


def ppv40_point(y, p, w=None):
    """Most sensitive threshold whose (weighted) PPV is at least 40%."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    best = None
    for t in np.unique(p):
        pred = p >= t
        if w[pred].sum() == 0:
            continue
        ppv = (w * y)[pred].sum() / w[pred].sum()
        sens = (w * y)[pred].sum() / (w * y).sum()
        if ppv >= 0.40 and (best is None or sens > best["Sensitivity"]):
            best = {"Threshold": t, "Sensitivity": sens,
                    "Specificity": (w * (1 - y))[~pred].sum() / (w * (1 - y)).sum(),
                    "PPV": ppv, "NPV": (w * (1 - y))[~pred].sum() / max(w[~pred].sum(), 1e-12),
                    "Predicted positive rate": w[pred].sum() / w.sum()}
    return best or {}


def nri_idi(y, p_new, p_old):
    """Continuous (category-free) NRI and IDI of p_new over p_old."""
    y = np.asarray(y).astype(bool)
    up, down = p_new > p_old, p_new < p_old
    nri = (up[y].mean() - down[y].mean()) + (down[~y].mean() - up[~y].mean())
    idi = (p_new[y].mean() - p_new[~y].mean()) - (p_old[y].mean() - p_old[~y].mean())
    return nri, idi


def aalen_johansen(t, cause, k):
    """Cumulative incidence of cause k with other causes as competing events."""
    t, cause = np.asarray(t, float), np.asarray(cause, int)
    times = np.unique(t[cause > 0])
    s, f, out = 1.0, 0.0, [0.0]
    for u in times:
        n = (t >= u).sum()
        f += s * ((t == u) & (cause == k)).sum() / n
        s *= 1 - ((t == u) & (cause > 0)).sum() / n
        out.append(f)
    return np.r_[0, times], np.asarray(out)


def gray_test(t, cause, g, k, horizon=60):
    """Gray's test for equality of cumulative incidence across groups.

    Nobody in this cohort is censored before the 60-month horizon, and in that case
    Gray's test is exactly a log-rank test in which people who die of a competing
    cause stay in the risk set to the end of follow-up."""
    t, cause = np.asarray(t, float), np.asarray(cause, int)
    return logrank(np.where(cause == k, t, horizon), (cause == k).astype(int), g)


def survey_weights(index):
    c = pd.read_csv(ROOT / "data/processed/cohort_9957.csv",
                    usecols=["SEQN", "SDDSRVYR", "WTMEC4YR", "WTMEC2YR"])
    w = np.where(c["SDDSRVYR"] < 3, c["WTMEC4YR"] * 2 / 3, c["WTMEC2YR"] / 3)
    return pd.Series(w, index=c["SEQN"].astype(float)).reindex(index).fillna(0.0)


# ----------------------------------------------------------------------------
# Explanations
# ----------------------------------------------------------------------------
def xgb_contribs(model, data, interactions=False):
    """Exact TreeSHAP from XGBoost itself, on the early-stopped model (log-odds)."""
    rng = (0, model.best_iteration + 1)
    dm = xgb.DMatrix(data)
    if interactions:
        return model.get_booster().predict(dm, pred_interactions=True, iteration_range=rng)
    return model.get_booster().predict(dm, pred_contribs=True, iteration_range=rng)[:, :-1]


def explain(name, model, data, groups, feats):
    base = name.split(" (")[0]
    if base == "XGBoost":
        v = xgb_contribs(model, data)
    elif base == "LightGBM":
        v = model.predict(data, pred_contrib=True)[:, :-1]
    elif base == "Random Forest":
        v = np.asarray(shap.TreeExplainer(model).shap_values(data, check_additivity=False))
        v = v[:, :, 1] if v.ndim == 3 else v
    else:
        v = model.shap_values(data)
    return aggregate(v, groups, feats)


def builtin_importance(name, model, cols, groups, feats):
    if name == "XGBoost":
        sc = model.get_booster().get_score(importance_type="total_gain")
        v = [sc.get(c, 0.0) for c in cols]
    elif name == "LightGBM":
        v = model.booster_.feature_importance("gain")
    elif name == "Random Forest":
        v = model.feature_importances_
    else:
        v = np.abs(model.m.coef_[0])  # standardized coefficients
    s = pd.Series(np.asarray(v, float)).groupby(np.asarray(groups)).sum()[feats]
    return (s / s.sum()).to_numpy()


def permutation_importance(model, Xte, yte, groups, feats, rng, repeats):
    """Drop in held-out ROC-AUC when one original predictor is shuffled
    (all one-hot columns of a categorical predictor move together)."""
    arr, g = Xte.to_numpy(copy=True), np.asarray(groups)
    base = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
    out = []
    for f in feats:
        idx = np.where(g == f)[0]
        saved, drops = arr[:, idx].copy(), []
        for _ in range(repeats):
            arr[:, idx] = saved[rng.permutation(len(arr))]
            p = model.predict_proba(pd.DataFrame(arr, columns=Xte.columns))[:, 1]
            drops.append(base - roc_auc_score(yte, p))
        arr[:, idx] = saved
        out.append(np.mean(drops))
    return np.asarray(out)


def agreement_rows(S, clusters):
    SC = {k: pd.DataFrame(np.abs(v)).T.groupby(clusters).sum().T.to_numpy() for k, v in S.items()}
    pairs = [(a, b, "Different models") for i, a in enumerate(MAIN) for b in MAIN[i + 1:]]
    pairs += [("XGBoost", "XGBoost (seed 2)", "Same model, new random seed"),
              ("Random Forest", "Random Forest (seed 2)", "Same model, new random seed"),
              ("XGBoost", "XGBoost (imputed inputs)", "Same model, imputed vs raw inputs"),
              ("XGBoost (imputed inputs)", "Random Forest", "Different models, same imputed inputs")]
    rows = []
    for a, b, kind in pairs:
        A, B = np.abs(S[a]), np.abs(S[b])
        rows.append({"Model A": a, "Model B": b, "Type": kind,
                     "Top-1 feature agreement": (A.argmax(1) == B.argmax(1)).mean(),
                     "Top-5 overlap": np.mean([len(set(np.argsort(-A[i])[:5])
                                                   & set(np.argsort(-B[i])[:5])) / 5
                                               for i in range(len(A))]),
                     "Mean Spearman |SHAP|": np.mean([spearmanr(A[i], B[i]).statistic
                                                      for i in range(len(A))]),
                     "Top-1 agreement (correlated groups)":
                         (SC[a].argmax(1) == SC[b].argmax(1)).mean()})
    return rows


def subgroup_masks(X):
    cvd = (X[["MCQ160B", "MCQ160E", "MCQ160F"]] == 1).any(axis=1)
    return {"All": pd.Series(True, index=X.index), "Men": X["RIAGENDR"] == 1,
            "Women": X["RIAGENDR"] == 2, "Age under 65": X["RIDAGEYR"] < 65,
            "Age 65+": X["RIDAGEYR"] >= 65, "Prior CVD": cvd, "No prior CVD": ~cvd}


# ----------------------------------------------------------------------------
# Cross-validation loop: everything that needs the fold matrices
# ----------------------------------------------------------------------------
def fold_matrices(X, y, tr, te, feats, f, fast):
    cache = OUT / "cache" / f"fold{f}_{'fast' if fast else 'full'}.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    prep = Prep(feats).fit(X.iloc[tr], impute=True, full=not fast)
    (ntr, g_nat), (nte, _) = prep.native(X.iloc[tr]), prep.native(X.iloc[te])
    (itr, g_imp), (ite, _) = prep.imputed(X.iloc[tr]), prep.imputed(X.iloc[te])
    d = {"nat": (ntr, nte, g_nat), "imp": (itr, ite, g_imp)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(d, open(cache, "wb"))
    return d


def cv_everything(X, y, feats, fast):
    n_explain, n_train_shap, repeats = (60, 400, 1) if fast else (300, 2000, 3)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = pd.DataFrame(index=X.index, columns=MAIN + ["Logistic Regression (unscaled)"], dtype=float)
    fold_id = pd.Series(-1, index=X.index)
    R = {"agree": [], "builtin": {m: [] for m in MAIN}, "perm": {m: [] for m in MAIN},
         "S": {k: [] for k in KIND}, "people": [], "train_shap": [], "train_sub": [],
         "interaction": [], "dependence": None, "pdp_model": None}
    masks = subgroup_masks(X)
    for f, (tr, te) in enumerate(skf.split(X, y)):
        mats = fold_matrices(X, y, tr, te, feats, f, fast)
        ytr, yte = y.iloc[tr], y.iloc[te]
        fold_id.iloc[te] = f
        fitted = {"Logistic Regression": LR().fit(mats["imp"][0], ytr),
                  "Random Forest": fit_rf(mats["imp"][0], ytr),
                  "XGBoost": fit_xgb(mats["nat"][0], ytr),
                  "LightGBM": fit_lgb(mats["imp"][0], ytr),
                  "XGBoost (seed 2)": fit_xgb(mats["nat"][0], ytr, seed=7),
                  "Random Forest (seed 2)": fit_rf(mats["imp"][0], ytr, seed=7),
                  "XGBoost (imputed inputs)": fit_xgb(mats["imp"][0], ytr)}
        for m in MAIN:
            oof.iloc[te, oof.columns.get_loc(m)] = fitted[m].predict_proba(mats[KIND[m]][1])[:, 1]
        # Paper 2.3 says no feature scaling: plain logistic regression on unscaled inputs
        raw_lr = LogisticRegression(max_iter=3000).fit(mats["imp"][0], ytr)
        oof.iloc[te, -1] = raw_lr.predict_proba(mats["imp"][1])[:, 1]

        rng = np.random.default_rng(SEED + f)
        for m in MAIN:
            a, b, g = mats[KIND[m]]
            R["builtin"][m].append(builtin_importance(m, fitted[m], list(a.columns), g, feats))
            R["perm"][m].append(permutation_importance(fitted[m], b, yte, g, feats, rng, repeats))

        people = X.iloc[te].sample(n_explain, random_state=SEED + f).index
        S = {k: explain(k, fitted[k], mats[KIND[k]][1].loc[people], mats[KIND[k]][2], feats)
             for k in KIND}
        corr = X.iloc[tr][feats].corr(method="spearman", min_periods=200).abs().fillna(0).to_numpy()
        corr = np.clip((corr + corr.T) / 2, 0, 1)
        np.fill_diagonal(corr, 1)
        clusters = fcluster(linkage(1 - corr[np.triu_indices(len(feats), 1)], "average"),
                            0.3, "distance")
        R["agree"] += [{"fold": f + 1, **r} for r in agreement_rows(S, clusters)]
        for k in KIND:
            R["S"][k].append(S[k])
        R["people"] += list(people)

        # Paper 2.6: SHAP inside the training data only, up to 2,000 people per fold
        ntr, _, g_nat = mats["nat"]
        samp = ntr.sample(min(n_train_shap, len(ntr)), random_state=SEED + f)
        v = np.abs(aggregate(xgb_contribs(fitted["XGBoost"], samp), g_nat, feats))
        R["train_shap"].append(v.mean(0))
        R["train_sub"].append({name: v[mk.loc[samp.index].to_numpy()].mean(0)
                               for name, mk in masks.items()})
        inter = xgb_contribs(fitted["XGBoost"], samp, interactions=True)
        i, j = list(samp.columns).index("SSTNT"), list(samp.columns).index("SSBNP")
        val = inter[:, i, j] + inter[:, j, i]
        main_tnt = np.abs(inter[:, i, i])
        R["interaction"].append({"fold": f + 1, "mean(|interaction|)": np.abs(val).mean(),
                                 "mean(interaction)": val.mean(),
                                 "median(interaction)": np.median(val),
                                 "95% quantile(|interaction|)": np.quantile(np.abs(val), 0.95),
                                 "mean(|TNT main effect|)": main_tnt.mean()})
        if f == 0:
            R["dependence"] = pd.DataFrame({"TNT": samp["SSTNT"], "NT-proBNP": samp["SSBNP"],
                                            "SHAP TNT": xgb_contribs(fitted["XGBoost"], samp)[:, i]})
            R["pdp_model"] = (fitted["XGBoost"], mats["nat"][1].sample(
                min(1000, len(te)), random_state=SEED))
        print(f"  fold {f + 1}/5 done", flush=True)
    assert oof.notna().all().all()
    R["S"] = {k: np.vstack(v) for k, v in R["S"].items()}
    return oof, fold_id, R


def cv_xgb(X, y, cols):
    """Out-of-fold XGBoost predictions with native missing values (same folds)."""
    p = pd.Series(np.nan, index=X.index)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        prep = Prep(cols).fit(X.iloc[tr][cols], impute=False)
        m = fit_xgb(prep.native(X.iloc[tr][cols])[0], y.iloc[tr])
        p.iloc[te] = m.predict_proba(prep.native(X.iloc[te][cols])[0])[:, 1]
    return p


# ----------------------------------------------------------------------------
# Paper 2.8: final parsimonious model
# ----------------------------------------------------------------------------
def fit_xgb_weighted(a, ya, b, yb):
    m = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.03, max_depth=4, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss",
                          early_stopping_rounds=50, n_jobs=8, random_state=SEED,
                          tree_method="hist", scale_pos_weight=(ya == 0).sum() / (ya == 1).sum())
    return m.fit(a, ya, eval_set=[(b, yb)], verbose=False)


def final_models(X, y, w, feats, names, fast):
    idx = np.arange(len(X))
    dev, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    tr, va = train_test_split(dev, test_size=0.15, stratify=y.iloc[dev], random_state=SEED)

    def run(cols):
        prep = Prep(cols).fit(X.iloc[tr][cols], impute=False)
        a, b, c = (prep.native(X.iloc[i][cols])[0] for i in (tr, va, te))
        m = fit_xgb_weighted(a, y.iloc[tr], b, y.iloc[va])
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
            m.predict_proba(b)[:, 1], y.iloc[va])
        raw = m.predict_proba(c)[:, 1]
        return m, prep, a, raw, iso.predict(raw)

    # Feature ranking from the 134-predictor model, training portion only
    m, prep, a, _, _ = run(feats)
    samp = a.sample(min(400 if fast else 2000, len(a)), random_state=SEED)
    v = np.abs(aggregate(xgb_contribs(m, samp), prep.native(X.iloc[tr][feats])[1], feats))
    masks = subgroup_masks(X)
    rank = pd.DataFrame({name: pd.Series(v[mk.loc[samp.index].to_numpy()].mean(0), index=feats)
                         .rank(ascending=False).astype(int) for name, mk in masks.items()})
    rank.insert(0, "Mean |SHAP|", v.mean(0))
    rank.insert(0, "Paper variable", pd.Series(names))
    rank["Worst subgroup rank"] = rank[list(masks)[1:]].max(axis=1)
    rank["In paper's five"] = rank.index.isin(FIVE)
    rank = rank.sort_values("All")
    ours = list(rank.index[:5])
    same = set(ours) == set(FIVE)  # identical set: one model, not two

    yt, wt = y.iloc[te].to_numpy(), w.iloc[te].to_numpy()
    rows, ops = [], []
    candidates = [("Paper's five predictors (also our top five)" if same
                   else "Paper's five predictors", FIVE)]
    if not same:
        candidates.append(("Top five by our SHAP ranking", ours))
    for label, cols in candidates + [("All 134 predictors", feats)]:
        _, _, _, raw, cal = run(cols)
        for stage, p in [("Raw", raw), ("Isotonic", cal)]:
            s_u, c_u = calibration_w(yt, p)
            s_w, c_w = calibration_w(yt, p, wt)
            rows.append({"Model": label, "Predictions": stage, "Test n": len(te),
                         "ROC-AUC": roc_auc_score(yt, p), "PR-AUC": average_precision_score(yt, p),
                         "Brier": brier_score_loss(yt, p), "Brier (weighted)": brier_w(yt, p, wt),
                         "Slope": s_u, "CITL": c_u, "Slope (weighted)": s_w, "CITL (weighted)": c_w})
        ops.append({"Model": label, "Setting": "Test, weighted, isotonic", **ppv40_point(yt, cal, wt)})
        ops.append({"Model": label, "Setting": "Test, unweighted, isotonic", **ppv40_point(yt, cal)})
    return rank, ours, same, pd.DataFrame(rows), pd.DataFrame(ops)


# ----------------------------------------------------------------------------
# Paper 2.7 and 2.9
# ----------------------------------------------------------------------------
def mortality(index):
    m = pd.concat([pd.read_fwf(ROOT / f"data/raw/mortality/NHANES_{c}_MORT_2019_PUBLIC.dat",
                               colspecs=[(0, 6), (15, 16), (16, 19), (42, 45)],
                               names=["SEQN", "MORTSTAT", "UCOD", "PERMTH_INT"],
                               dtype={"UCOD": str}, na_values=["."])
                   for c in ["1999_2000", "2001_2002", "2003_2004"]])
    return m.set_index(m["SEQN"].astype(float)).reindex(index)


def youden(y, x):
    fpr, tpr, thr = roc_curve(y, x)
    return thr[np.argmax(tpr - fpr)]


def survival(X, y, fold_id, w):
    mort = mortality(X.index)
    assert (mort.loc[mort["MORTSTAT"] == 0, "PERMTH_INT"] >= 60).all()  # no early censoring
    died5 = (mort["MORTSTAT"] == 1) & (mort["PERMTH_INT"] <= 60)
    assert died5.astype(int).equals(y)
    # UCOD_LEADING: 001 = diseases of heart, 005 = cerebrovascular diseases
    cvd = died5 & mort["UCOD"].isin(["001", "005"])
    d = pd.DataFrame({"tnt": X["SSTNT"], "bnp": X["SSBNP"], "y": y, "fold": fold_id,
                      "t": mort["PERMTH_INT"].clip(upper=60),
                      "cause": np.where(cvd, 1, np.where(died5, 2, 0)),
                      "any_death": mort["MORTSTAT"], "t_full": mort["PERMTH_INT"],
                      "w": w}).dropna()
    cuts, order = [], ["Both low", "NT-proBNP high only", "TNT high only", "Both high"]
    d["group"] = ""
    for f in range(5):
        trn, tst = d[d.fold != f], d.index[d.fold == f]
        ct, cb = youden(trn["y"], trn["tnt"]), youden(trn["y"], trn["bnp"])
        cuts.append({"fold": f + 1, "n_train": len(trn), "n_test": len(tst),
                     "TNT_cutoff": ct, "NT-proBNP_cutoff": cb})
        ht, hb = d.loc[tst, "tnt"] >= ct, d.loc[tst, "bnp"] >= cb
        d.loc[tst, "group"] = np.select([~ht & ~hb, ~ht & hb, ht & ~hb], order[:3], order[3])

    summ = d.groupby("group").agg(n=("y", "size"), deaths=("y", "sum"),
                                  cvd_deaths=("cause", lambda s: (s == 1).sum()),
                                  other_deaths=("cause", lambda s: (s == 2).sum())).loc[order]
    for c in ["deaths", "cvd_deaths", "other_deaths"]:
        summ[c.replace("deaths", "mortality %")] = 100 * summ[c] / summ["n"]
    tests = pd.DataFrame([
        {"Outcome": "All-cause death", "Test": "Log-rank",
         "p": logrank(d["t"], d["y"], d["group"])},
        {"Outcome": "Cardiovascular death", "Test": "Gray (competing risk)",
         "p": gray_test(d["t"], d["cause"], d["group"], 1)},
        {"Outcome": "Non-cardiovascular death", "Test": "Gray (competing risk)",
         "p": gray_test(d["t"], d["cause"], d["group"], 2)}])

    # Why do our cut-offs differ from the paper's (TNT about 7.5, NT-proBNP about 75-85)?
    alt = []
    for label, target, sub in [
            ("5-year all-cause death (used here)", d["y"], d),
            ("5-year cardiovascular death", (d["cause"] == 1).astype(int), d),
            ("Death at any time to 2019", d["any_death"].astype(int), d)]:
        alt.append({"Outcome used for Youden index": label,
                    "TNT_cutoff": youden(target, sub["tnt"]),
                    "NT-proBNP_cutoff": youden(target, sub["bnp"])})
    alt.append({"Outcome used for Youden index": "Median of the biomarker",
                "TNT_cutoff": d["tnt"].median(), "NT-proBNP_cutoff": d["bnp"].median()})
    alt.append({"Outcome used for Youden index": "Paper, Table S3 (range over folds)",
                "TNT_cutoff": "7.41-7.59", "NT-proBNP_cutoff": "75.31-85.41"})

    colors = [BLUE, AQUA, YELLOW, ORANGE]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for g, col in zip(order, colors):
        k = d.group == g
        tt, ss = kaplan_meier(d.loc[k, "t"], d.loc[k, "y"])
        ax.step(np.r_[tt, 60], np.r_[ss, ss[-1]], where="post", color=col,
                label=f"{g} (n={k.sum():,})")
        ax.annotate(f"{g}  {100 * ss[-1]:.0f}%", (60, ss[-1]), xytext=(6, 0), fontsize=9,
                    textcoords="offset points", va="center", color=INK2, annotation_clip=False)
    ax.set(xlim=(0, 60), xlabel="Months since interview", ylabel="Survival probability",
           title="Five-year survival by Troponin-T and NT-proBNP group")
    ax.set_ylim(top=1.0)
    ax.legend(loc="lower left")
    fig.text(0.01, 0.01, f"Cut-offs: Youden index in the other four folds. "
             f"Log-rank p = {tests.p[0]:.1e}. n = {len(d):,}.", color=MUTED, fontsize=8)
    fig.subplots_adjust(right=0.74, bottom=0.17)
    fig.savefig(OUT / "fig_km_tnt_ntprobnp.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, k, title, pv in [(axes[0], 1, "Cardiovascular death", tests.p[1]),
                             (axes[1], 2, "Non-cardiovascular death", tests.p[2])]:
        ends = []
        for g, col in zip(order, colors):
            sel = d.group == g
            tt, ff = aalen_johansen(d.loc[sel, "t"], d.loc[sel, "cause"], k)
            ax.step(np.r_[tt, 60], 100 * np.r_[ff, ff[-1]], where="post", color=col, label=g)
            ends.append(100 * ff[-1])
        pos, last = {}, -np.inf  # nudge end labels apart so close values stay readable
        for i in np.argsort(ends):
            pos[i] = max(ends[i], last + 1.1)
            last = pos[i]
        for i, v in enumerate(ends):
            ax.annotate(f"{v:.1f}%", (60, pos[i]), xytext=(5, 0), fontsize=9,
                        textcoords="offset points", va="center", color=INK2,
                        annotation_clip=False)
        ax.set(xlim=(0, 60), xlabel="Months since interview",
               title=f"{title} (Gray's test p = {pv:.1e})")
    axes[0].set_ylabel("Cumulative incidence (%)")
    axes[0].legend(loc="upper left")
    fig.subplots_adjust(wspace=0.16, right=0.95, bottom=0.13)
    fig.savefig(OUT / "fig_cumulative_incidence.png", dpi=160)
    plt.close(fig)
    # Paper Figure 3 follows people for about 250 months, so repeat the analysis on the
    # full follow-up: cut-offs by Youden index for death at any time, other four folds.
    d["group_full"], cuts_full = "", []
    for f in range(5):
        trn, tst = d[d.fold != f], d.index[d.fold == f]
        ct, cb = youden(trn["any_death"], trn["tnt"]), youden(trn["any_death"], trn["bnp"])
        cuts_full.append({"fold": f + 1, "n_train": len(trn), "n_test": len(tst),
                          "TNT_cutoff": ct, "NT-proBNP_cutoff": cb})
        ht, hb = d.loc[tst, "tnt"] >= ct, d.loc[tst, "bnp"] >= cb
        d.loc[tst, "group_full"] = np.select([~ht & ~hb, ~ht & hb, ht & ~hb], order[:3], order[3])
    p_full = logrank(d["t_full"], d["any_death"], d["group_full"])
    full = d.groupby("group_full").agg(n=("y", "size"), deaths=("any_death", "sum")).loc[order]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    surv_end = []
    for g, col in zip(order, colors):
        k = d.group_full == g
        tt, ss = kaplan_meier(d.loc[k, "t_full"], d.loc[k, "any_death"])
        ax.step(np.r_[tt[tt <= 240], 240], np.r_[ss[tt <= 240], ss[tt <= 240][-1]],
                where="post", color=col, label=f"{g} (n={k.sum():,})")
        surv_end.append(100 * ss[tt <= 240][-1])  # few people are followed beyond 240 months
    pos, last = {}, -np.inf
    for i in np.argsort(surv_end):
        pos[i] = max(surv_end[i], last + 5)
        last = pos[i]
    for i, v in enumerate(surv_end):
        ax.annotate(f"{v:.0f}%", (240, pos[i] / 100), xytext=(6, 0), fontsize=9,
                    textcoords="offset points", va="center", color=INK2, annotation_clip=False)
    full["survival at 240 months %"] = surv_end
    ax.set(xlim=(0, 240), ylim=(0, 1), xlabel="Months since interview",
           ylabel="Survival probability",
           title="Survival over the full follow-up, by Troponin-T and NT-proBNP group")
    ax.legend(loc="lower left")
    fig.text(0.01, 0.01, f"Cut-offs: Youden index for death at any time, other four folds. "
             f"Log-rank p {'= %.1e' % p_full if p_full > 0 else '< 1e-300'}. n = {len(d):,}.",
             color=MUTED, fontsize=8)
    fig.subplots_adjust(right=0.92, bottom=0.17)
    fig.savefig(OUT / "fig_km_full_followup.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(cuts_full).to_csv(OUT / "km_cutoffs_full_followup.csv", index=False)
    full.reset_index().to_csv(OUT / "km_groups_full_followup.csv", index=False)
    tests.loc[len(tests)] = ["All-cause death, full follow-up", "Log-rank", p_full]

    return pd.DataFrame(cuts), summ.reset_index(), tests, pd.DataFrame(alt)


def temporal(X, y, cyc, feats):
    tr, te = (cyc.isin([1, 2])).to_numpy(), (cyc == 3).to_numpy()
    yt = y[te].to_numpy()
    preds = {}
    for name, cols in [("Main model: XGBoost, five predictors", FIVE),
                       ("XGBoost, 134 predictors", feats)]:
        prep = Prep(cols).fit(X[tr][cols], impute=False)
        preds[name] = fit_xgb(prep.native(X[tr][cols])[0], y[tr]).predict_proba(
            prep.native(X[te][cols])[0])[:, 1]
    for name, cols in [("Model A: age", ["RIDAGEYR"]),
                       ("Model B: age + gender", ["RIDAGEYR", "RIAGENDR"]),
                       ("Model C: age + physical activity", ["RIDAGEYR", "PAQ180"])]:
        prep = Prep(cols).fit(X[tr][cols], impute=False)
        a, b = prep.native(X[tr][cols])[0], prep.native(X[te][cols])[0]
        preds[name] = LR().fit(a, y[tr]).predict_proba(b)[:, 1]
    main = preds["Main model: XGBoost, five predictors"]
    rows = []
    for name, p in preds.items():
        s, c = calibration_w(yt, p)
        r = {"Model": name, "Train n": int(tr.sum()), "Test n": int(te.sum()),
             "AUC": roc_auc_score(yt, p), "CITL": c, "Slope": s, "Brier": brier_score_loss(yt, p)}
        if name.startswith("Model "):
            r["NRI (main vs this)"], r["IDI (main vs this)"] = nri_idi(yt, main, p)
        rows.append(r)
    return pd.DataFrame(rows)


def definition_sensitivity(X, y, cand, feats):
    """Does the unconfirmed MET / heart-rate definition change model performance?"""
    full = pd.concat([X, cand[[c for c in cand.columns if c not in X.columns]]], axis=1)
    swap = lambda m: [m.get(c, c) for c in feats]  # noqa: E731
    variants = {
        "Base (as used in all results)": feats,
        "Without MET score and heart-rate increment (132)":
            [c for c in feats if c not in ("MET_MIN_30D_CAND", "HR_INCREMENT_CAND_REST")],
        "MET = summed MET values": swap({"MET_MIN_30D_CAND": "MET_SUM_CAND"}),
        "HR increment = peak minus predicted maximum":
            swap({"HR_INCREMENT_CAND_REST": "HR_INCREMENT_CAND_PRED"}),
        "HR recovery = peak minus recovery heart rate":
            swap({f"CVDR{k}HR": f"HRR_{k}MIN_CAND" for k in (1, 2, 3)}),
        "eGFR = CKD-EPI 2009": swap({"eGFR": "eGFR_CKDEPI2009_CAND"}),
        "Glycated albumin = percent (SSGALBP)": swap({"SSGALB": "SSGALBP"}),
        "Coronary heart disease (MCQ160C) instead of heart attack":
            swap({"MCQ160E": "MCQ160C"}),
    }
    rows = []
    for name, cols in variants.items():
        p = cv_xgb(full, y, cols)
        rows.append({"Variant": name, "Predictors": len(cols), "XGBoost ROC-AUC": roc_auc_score(y, p),
                     "Brier": brier_score_loss(y, p)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def short(name):
    return name.replace("Logistic Regression", "LR").replace("Random Forest", "RF")


def fig_agreement(agg):
    d = agg.sort_values("Top-1 feature agreement")
    kinds = {"Different models": ORANGE, "Different models, same imputed inputs": ORANGE,
             "Same model, imputed vs raw inputs": AQUA, "Same model, new random seed": BLUE}
    fig, ax = plt.subplots(figsize=(8.6, 5))
    labels = [f"{short(a)} vs {short(b)}" for a, b in zip(d["Model A"], d["Model B"])]
    ax.barh(labels, 100 * d["Top-1 feature agreement"], color=d["Type"].map(kinds), height=0.62,
            xerr=100 * d["Top-1 SD across folds"], error_kw={"ecolor": MUTED, "lw": 1})
    for i, (v, s) in enumerate(zip(100 * d["Top-1 feature agreement"],
                                   100 * d["Top-1 SD across folds"])):
        ax.text(v + s + 1.5, i, f"{v:.0f}%", va="center", color=INK2, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set(xlim=(0, 112), xlabel="Participants with the same top SHAP feature (%), mean of 5 folds",
           title="Do models agree on each person's most important predictor?")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ORANGE, label="Different models"),
                       Patch(color=AQUA, label="Same model, imputed vs raw inputs"),
                       Patch(color=BLUE, label="Same model, new random seed")], loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_shap_agreement.png", dpi=160)
    plt.close(fig)


def fig_shap_global(train_rank, names):
    top = train_rank.head(15)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.barh(top["Paper variable"][::-1], top["Mean |SHAP|"][::-1], color=AQUA, height=0.62,
            xerr=top["SD across folds"][::-1], error_kw={"ecolor": MUTED, "lw": 1})
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Mean |SHAP value| (log-odds), training folds",
           title="XGBoost: 15 most important predictors (global SHAP)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_shap_global_xgboost.png", dpi=160)
    plt.close(fig)


def fig_top10(X, top10, names):
    c = X[top10].corr(method="spearman", min_periods=200)
    order = leaves_list(linkage(X[top10].corr(method="spearman", min_periods=200).fillna(0)
                                .to_numpy(), "average", metric="correlation"))
    c = c.iloc[order, order]
    lab = [names[f] for f in c.index]
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    im = ax.imshow(c, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(10), lab, rotation=45, ha="right")
    ax.set_yticks(range(10), lab)
    ax.grid(False)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{c.iat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if abs(c.iat[i, j]) > 0.6 else M.INK)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman correlation")
    ax.set_title("Top ten SHAP predictors: Spearman correlation")
    fig.tight_layout()
    fig.savefig(OUT / "fig_top10_correlation.png", dpi=160)
    plt.close(fig)
    c.index, c.columns = lab, lab
    return c


def fig_dependence(dep):
    d = dep.dropna(subset=["TNT"]).copy()
    d = d[d["TNT"] > 0]
    col = np.log10(d["NT-proBNP"].clip(lower=1)).fillna(-1)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    known = col >= 0
    if (~known).any():
        ax.scatter(d.loc[~known, "TNT"], d.loc[~known, "SHAP TNT"], s=14, color="#c3c2b7",
                   label="NT-proBNP not measured")
        ax.legend(loc="upper left")
    sc = ax.scatter(d.loc[known, "TNT"], d.loc[known, "SHAP TNT"], s=14, c=col[known],
                    cmap=SEQ_BLUE, vmin=1, vmax=3.5)
    cb = fig.colorbar(sc, ax=ax, ticks=[1, 2, 3])
    cb.ax.set_yticklabels(["10", "100", "1,000"])
    cb.set_label("NT-proBNP (pg/mL)")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set(xscale="log", xlabel="Troponin-T (ng/L, log scale)",
           ylabel="SHAP value for Troponin-T (log-odds)",
           title="Troponin-T effect on predicted risk, coloured by NT-proBNP")
    fig.tight_layout()
    fig.savefig(OUT / "fig_shap_dependence_tnt.png", dpi=160)
    plt.close(fig)


def fig_importance(imp, names):
    """XGBoost: three importance measures side by side for the top ten by SHAP."""
    cols = [("XGBoost", "SHAP"), ("XGBoost", "Permutation"), ("XGBoost", "Built-in")]
    titles = ["SHAP (mean |value|)", "Permutation (AUC drop)", "Built-in (total gain)"]
    top = imp[("XGBoost", "SHAP")].sort_values(ascending=False).index[:10]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    for ax, c, t in zip(axes, cols, titles):
        share = 100 * imp[c].clip(lower=0) / imp[c].clip(lower=0).sum()
        rank = imp[c].rank(ascending=False).astype(int)
        ax.barh([names[f] for f in top][::-1], share[top][::-1], color=AQUA, height=0.62)
        for i, f in enumerate(top[::-1]):
            ax.text(share[f] + 0.6, i, f"#{rank[f]}", va="center", color=INK2, fontsize=9)
        ax.grid(axis="y", visible=False)
        ax.set(title=t, xlabel="Share of total importance (%)",
               xlim=(0, share[top].max() * 1.18))
    fig.suptitle("XGBoost: does the importance measure change the ranking? (# = rank among 134)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig_importance_measures_xgboost.png", dpi=160)
    plt.close(fig)


def fig_pdp(model, sample, X):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    out = {}
    for ax, (col, label, logx) in zip(axes, [("RIDAGEYR", "Age (years)", None),
                                             ("SSTNT", "Troponin-T (ng/L)", [3, 10, 30]),
                                             ("SSBNP", "NT-proBNP (pg/mL)", [10, 100, 1000])]):
        grid = np.unique(np.nanquantile(X[col], np.linspace(0.02, 0.98, 25)))
        ys = []
        for g in grid:
            s = sample.copy()
            s[col] = g
            ys.append(model.predict_proba(s)[:, 1].mean())
        ax.plot(grid, 100 * np.asarray(ys), color=AQUA)
        ax.set(xlabel=label, title=label.split(" (")[0])
        if logx:
            ax.set_xscale("log")
            ax.set_xticks(logx, [f"{v:,}" for v in logx])
            ax.xaxis.set_minor_formatter(NullFormatter())
        out[col] = pd.Series(ys, index=grid)
    levels = [c for c in sample.columns if c.startswith("PAQ180=") and c[-1] in "1234"]
    ys = []
    for lv in levels:
        s = sample.copy()
        s[[c for c in sample.columns if c.startswith("PAQ180=")]] = 0.0
        s[lv] = 1.0
        ys.append(100 * model.predict_proba(s)[:, 1].mean())
    lab = ["Mostly sits", "Stands / walks", "Light loads", "Heavy work"][:len(levels)]
    axes[3].bar([x.replace(" ", "\n", 1).replace("\n/ ", " /\n") for x in lab], ys,
                color=AQUA, width=0.6)
    for i, v in enumerate(ys):
        axes[3].text(i, v + 0.2, f"{v:.1f}", ha="center", color=INK2, fontsize=9)
    axes[3].grid(axis="x", visible=False)
    axes[3].tick_params(axis="x", labelsize=9)
    axes[3].set(title="Physical activity", xlabel="Usual daily activity")
    axes[0].set_ylabel("Average predicted 5-year risk (%)")
    fig.suptitle("XGBoost partial dependence: predicted risk as one predictor changes",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig_partial_dependence_xgboost.png", dpi=160)
    plt.close(fig)
    out["PAQ180"] = pd.Series(np.asarray(ys) / 100, index=lab)
    return pd.concat(out).rename("Mean predicted risk").rename_axis(["Predictor", "Value"])


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="approximate quick run")
    fast = ap.parse_args().fast
    OUT.mkdir(parents=True, exist_ok=True)
    M.OUT = OUT
    X, y, cyc, names = M.load()
    feats = list(X.columns)
    cand = pd.read_pickle(M.CAND).set_index("SEQN")
    w = survey_weights(X.index)
    print(f"Data {X.shape}, deaths {int(y.sum())}, weighted n {w.sum():.3g}", flush=True)

    # ---- which predictors are one-hot encoded -----------------------------------
    enc = pd.DataFrame([{
        "Paper variable": names[c], "Column": c, "Categories": X[c].nunique(),
        "Category codes": ", ".join(f"{v:g}" for v in sorted(X[c].dropna().unique())),
        "Missing (n)": int(X[c].isna().sum()),
        "Missing (%)": round(100 * X[c].isna().mean(), 1),
        "Filled with (most frequent)": f"{X[c].mode()[0]:g}"} for c in feats if c in M.CATEGORICAL])
    assert len(enc) == len(M.CATEGORICAL)
    enc.to_csv(OUT / "categorical_encoding.csv", index=False)

    # ---- cross-validation: models, importance, SHAP -----------------------------
    oof, fold_id, R = cv_everything(X, y, feats, fast)
    oof["XGBoost (paper's five predictors)"] = cv_xgb(X, y, FIVE)

    # ---- 2.8 final model and the data-driven simple model ---------------------
    rank, ours, same, final, ops = final_models(X, y, w, feats, names, fast)
    if not same:
        oof["XGBoost (top five by our SHAP)"] = cv_xgb(X, y, ours)
    rank.to_csv(OUT / "simple_model_feature_ranking.csv")
    final.to_csv(OUT / "final_model_test_metrics.csv", index=False)
    ops.to_csv(OUT / "final_model_operating_points.csv", index=False)

    rows = []
    for m in oof.columns:
        p = oof[m].to_numpy()
        s_w, c_w = calibration_w(y, p, w)
        n = 134 if "(" not in m or "unscaled" in m else 5
        rows.append({"Model": m, "Predictors": n, **metrics(y.to_numpy(), p),
                     "Brier (weighted)": brier_w(y, p, w), "CalSlope (weighted)": s_w,
                     "CITL (weighted)": c_w})
    cv = pd.DataFrame(rows)
    cv.to_csv(OUT / "cv_metrics.csv", index=False)
    pd.concat([y.rename("death_5yr"), fold_id.rename("fold"), oof], axis=1).to_csv(
        OUT / "oof_predictions.csv")
    M.fig_roc_cal(y, oof[MAIN], cv)

    # ---- preprocessing sensitivity ---------------------------------------------
    old = ROOT / "results/models_v0.1/cv_metrics.csv"
    prep_rows = []
    if old.exists():
        o = pd.read_csv(old).set_index("Model")
        for m in ["Logistic Regression", "Random Forest"]:
            prep_rows.append({"Model": m, "Setting": "Shortcut imputer (5 rounds, 20 neighbours)",
                              "ROC-AUC": o.loc[m, "ROC-AUC"], "Brier": o.loc[m, "Brier"]})
    c = cv.set_index("Model")
    for m, label in [("Logistic Regression", "Full imputer, scaled inputs"),
                     ("Random Forest", "Full imputer"),
                     ("Logistic Regression (unscaled)", "Full imputer, unscaled inputs (paper 2.3)")]:
        prep_rows.append({"Model": m.split(" (")[0], "Setting": label,
                          "ROC-AUC": c.loc[m, "ROC-AUC"], "Brier": c.loc[m, "Brier"]})
    pd.DataFrame(prep_rows).to_csv(OUT / "preprocessing_sensitivity.csv", index=False)
    defs = definition_sensitivity(X, y, cand, feats)
    defs.to_csv(OUT / "definition_sensitivity.csv", index=False)

    # ---- research question: explanation agreement ------------------------------
    ag = pd.DataFrame(R["agree"])
    ag.to_csv(OUT / "shap_agreement_by_fold.csv", index=False)
    keys = ["Model A", "Model B", "Type"]
    agg = ag.groupby(keys, sort=False).mean(numeric_only=True).drop(columns="fold")
    agg["Top-1 SD across folds"] = ag.groupby(keys, sort=False)["Top-1 feature agreement"].std()
    agg = agg.reset_index()
    agg.to_csv(OUT / "shap_agreement.csv", index=False)
    fig_agreement(agg)
    S, people = R["S"], R["people"]
    top1 = pd.DataFrame({m: pd.Series(np.asarray(feats)[np.abs(S[m]).argmax(1)]).map(names)
                         .value_counts(normalize=True) for m in MAIN}).fillna(0)
    top1.sort_values("XGBoost", ascending=False).to_csv(OUT / "shap_top1_feature_share.csv")
    local = pd.DataFrame({"SEQN": people})
    for m in MAIN:
        o = np.argsort(-np.abs(S[m]), axis=1)[:, :3]
        for k in range(3):
            local[f"{m} #{k + 1}"] = [names[feats[j]] for j in o[:, k]]
    local.to_csv(OUT / "shap_local_top3.csv", index=False)

    # ---- 2.6 SHAP replication ---------------------------------------------------
    ts = np.vstack(R["train_shap"])
    train_rank = pd.DataFrame({"Paper variable": pd.Series(names), "Mean |SHAP|": ts.mean(0),
                               "SD across folds": ts.std(0)}, index=feats)
    train_rank["Rank"] = train_rank["Mean |SHAP|"].rank(ascending=False).astype(int)
    for name in subgroup_masks(X):
        if name != "All":
            v = np.mean([d[name] for d in R["train_sub"]], axis=0)
            train_rank[f"Rank: {name}"] = pd.Series(v, index=feats).rank(ascending=False).astype(int)
    train_rank = train_rank.sort_values("Rank")
    train_rank.to_csv(OUT / "shap_training_folds_ranking.csv")
    fig_shap_global(train_rank, names)
    top10 = list(train_rank.index[:10])
    fig_top10(X, top10, names).to_csv(OUT / "shap_top10_correlation.csv")
    inter = pd.DataFrame(R["interaction"])
    inter.to_csv(OUT / "shap_interaction_tnt_ntprobnp.csv", index=False)
    fig_dependence(R["dependence"])
    plt.figure()
    shap.summary_plot(S["XGBoost"], X.loc[people, feats].to_numpy(),
                      feature_names=[names[f] for f in feats], max_display=15, show=False,
                      plot_size=(8, 5.5))
    plt.title("XGBoost: SHAP values per participant (top 15)")
    plt.tight_layout()
    plt.savefig(OUT / "fig_shap_beeswarm_xgboost.png", dpi=160)
    plt.close("all")

    # ---- importance beyond SHAP ------------------------------------------------
    imp = {}
    for m in MAIN:
        imp[(m, "SHAP")] = np.abs(S[m]).mean(0)
        imp[(m, "Permutation")] = np.mean(R["perm"][m], axis=0)
        imp[(m, "Built-in")] = np.mean(R["builtin"][m], axis=0)
    imp = pd.DataFrame(imp, index=feats)
    out_imp = imp.copy()
    out_imp.columns = [f"{m} | {k}" for m, k in imp.columns]
    ranks = imp.rank(ascending=False).astype(int)
    ranks.columns = [f"{m} | {k} rank" for m, k in imp.columns]
    pd.concat([pd.Series(names, name="Paper variable"), out_imp, ranks], axis=1).sort_values(
        "XGBoost | SHAP rank").to_csv(OUT / "importance_all_measures.csv")

    def top10_overlap(a, b):
        return len(set(a.nlargest(10).index) & set(b.nlargest(10).index)) / 10

    within, between = [], []
    kinds = ["SHAP", "Permutation", "Built-in"]
    for m in MAIN:
        for i, a in enumerate(kinds):
            for b in kinds[i + 1:]:
                within.append({"Model": m, "Measure A": a, "Measure B": b,
                               "Spearman of rankings": spearmanr(imp[(m, a)], imp[(m, b)]).statistic,
                               "Top-10 overlap": top10_overlap(imp[(m, a)], imp[(m, b)])})
    for k in kinds:
        for i, a in enumerate(MAIN):
            for b in MAIN[i + 1:]:
                between.append({"Measure": k, "Model A": a, "Model B": b,
                                "Spearman of rankings": spearmanr(imp[(a, k)], imp[(b, k)]).statistic,
                                "Top-10 overlap": top10_overlap(imp[(a, k)], imp[(b, k)])})
    within, between = pd.DataFrame(within), pd.DataFrame(between)
    within.to_csv(OUT / "importance_agreement_between_measures.csv", index=False)
    between.to_csv(OUT / "importance_agreement_between_models.csv", index=False)
    top_tbl = pd.DataFrame({f"{short(m)} | {k}": [names[f] for f in imp[(m, k)].nlargest(5).index]
                            for m in MAIN for k in kinds})
    top_tbl.to_csv(OUT / "importance_top5_by_measure.csv", index=False)
    fig_importance(imp, names)
    pdp = fig_pdp(*R["pdp_model"], X)
    pdp.to_csv(OUT / "partial_dependence_xgboost.csv")

    # ---- 2.7 survival, 2.9 temporal --------------------------------------------
    cuts, groups, tests, alt = survival(X, y, fold_id, w)
    cuts.to_csv(OUT / "km_cutoffs.csv", index=False)
    groups.to_csv(OUT / "km_groups.csv", index=False)
    tests.to_csv(OUT / "survival_tests.csv", index=False)
    alt.to_csv(OUT / "km_cutoff_diagnostics.csv", index=False)
    temp = temporal(X, y, cyc, feats)
    temp.to_csv(OUT / "temporal_validation.csv", index=False)

    json.dump({"fast_mode": fast, "n": len(X), "deaths": int(y.sum()), "seed": SEED,
               "provisional_definitions": M.PROVISIONAL, "categorical_predictors": M.CATEGORICAL,
               "top_five_by_our_shap": ours,
               "our_top_five_equals_papers_five": same,
               "people_explained": len(people)}, open(OUT / "summary.json", "w"), indent=2)

    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60)
    show = ["Model", "Predictors", "ROC-AUC", "PR-AUC", "Brier", "CalSlope", "CITL",
            "Brier (weighted)", "CalSlope (weighted)", "CITL (weighted)", "Sens@PPV40", "Spec@Sens85"]
    for title, df in [("CV METRICS", cv[show]), ("PREPROCESSING", pd.DataFrame(prep_rows)),
                      ("DEFINITIONS", defs), ("AGREEMENT", agg), ("TOP-1 SHARE", top1.sort_values(
                          "XGBoost", ascending=False).head(8)),
                      ("TRAIN-FOLD SHAP", train_rank.head(12)), ("INTERACTION", inter),
                      ("SIMPLE MODEL RANKING", rank.head(10)), ("FINAL MODEL", final),
                      ("OPERATING POINTS", ops), ("MEASURES WITHIN MODEL", within),
                      ("MODELS WITHIN MEASURE", between), ("TOP5", top_tbl.T),
                      ("KM CUTS", cuts), ("KM GROUPS", groups), ("TESTS", tests),
                      ("CUTOFF DIAGNOSTICS", alt), ("TEMPORAL", temp)]:
        print(f"\n===== {title}\n{df.round(3).to_string()}", flush=True)


if __name__ == "__main__":
    main()
