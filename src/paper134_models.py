"""Model pipeline on the 134-predictor candidate dataset (Lin et al. 2026 replication).

Covers the roadmap items after data construction:
  * Logistic Regression, Random Forest, XGBoost, LightGBM
  * stratified 5-fold CV with out-of-fold (OOF) predictions and ROC curves
  * precision / recall / F1 / ROC-AUC / PR-AUC, Brier score, calibration
  * clinically relevant operating points (PPV fixed at 40%, sensitivity fixed at 85%)
  * global and local SHAP, and explanation agreement across models
  * 134-predictor vs five-predictor model
  * temporal validation across survey cycles (train 1999-2002, test 2003-2004)
  * Kaplan-Meier curves for Troponin-T x NT-proBNP groups

Input : results/paper134_candidate_v0.1.pkl (read only)
Output: results/models_v0.1/  (tables as CSV, figures as PNG, summary.json)

PROVISIONAL DEFINITIONS (paper does not define them; see evidence report):
  heart rate increment = MAX_HR - resting pulse  (HR_INCREMENT_CAND_REST)
  MET score            = sum(MET x times x minutes) over 30 days (MET_MIN_30D_CAND)
  heart rate recovery, eGFR, medication flags = current master definitions

Usage: .venv/bin/python src/paper134_models.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.experimental import enable_iterative_imputer  # noqa: F401,E402
from sklearn.impute import IterativeImputer, SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (average_precision_score, brier_score_loss,  # noqa: E402
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
import lightgbm as lgb  # noqa: E402
import xgboost as xgb  # noqa: E402
import shap  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CAND = ROOT / "results" / "paper134_candidate_v0.1.pkl"
CROSS = ROOT / "results" / "paper_134_feature_audit.csv"
OUT = ROOT / "results" / "models_v0.1"
SEED = 42
N_EXPLAIN = 500

PROVISIONAL = {"DERIVED_HR_INCREMENT": "HR_INCREMENT_CAND_REST",
               "DERIVED_MET": "MET_MIN_30D_CAND"}
CATEGORICAL = ["RIAGENDR", "RIDRETH1", "INDHHINC", "PAQ180", "BPQ020", "MCQ160B",
               "MCQ160C", "MCQ160F", "DIQ010", "XY_Estimation"]
FIVE = ["RIDAGEYR", "RIAGENDR", "PAQ180", "SSTNT", "SSBNP"]
FORBIDDEN = {"SEQN", "death_5yr", "SDDSRVYR", "MORTSTAT", "PERMTH_INT", "ELIGSTAT"}

# Chart style (validated palette: blue, orange, aqua, yellow on light surface)
COL = {"Logistic Regression": "#2a78d6", "Random Forest": "#eb6834",
       "XGBoost": "#1baf7a", "LightGBM": "#eda100"}
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
                     "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "legend.frameon": False, "lines.linewidth": 2})


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load():
    c = pd.read_pickle(CAND).set_index("SEQN")
    cross = pd.read_csv(CROSS)
    feats = [r["Dataset column"] if isinstance(r["Dataset column"], str)
             else PROVISIONAL[r["Paper code"]] for _, r in cross.iterrows()]
    assert len(feats) == 134 and len(set(feats)) == 134
    assert not FORBIDDEN & set(feats)
    names = dict(zip(feats, cross["Paper feature"]))
    X = c[feats].astype(float)
    y = c["death_5yr"].astype(int)
    return X, y, c["SDDSRVYR"].astype(int), names


class Prep:
    """Fold-safe preprocessing. Categorical: most-frequent impute then one-hot
    (unknown levels ignored). Numeric: left as NaN ('native', for boosted trees) or
    iterative multivariable imputation ('imputed', for LR and RF)."""

    def __init__(self, cols):
        self.cat = [c for c in cols if c in CATEGORICAL]
        self.num = [c for c in cols if c not in CATEGORICAL]

    def fit(self, X, impute=True, full=False):
        self.mode = {c: X[c].mode().iloc[0] for c in self.cat}
        self.levels = {c: sorted(X[c].fillna(self.mode[c]).unique()) for c in self.cat}
        self.imp = None
        if impute and self.num:
            # n_nearest_features keeps the chained-equation imputer tractable.
            # full=True: all predictors, 10 rounds (paper 2.3). full=False: fast approximation.
            kw = {} if full else {"max_iter": 5, "n_nearest_features": 20}
            # Imputed values are bounded to the range observed in the training fold:
            # the 30 DNA-methylation variables are jointly missing and collinear, and
            # the unbounded chained equations diverge (values near 1e48).
            self.imp = IterativeImputer(random_state=SEED, initial_strategy="median",
                                        skip_complete=True,
                                        min_value=np.nanmin(X[self.num].to_numpy(), axis=0),
                                        max_value=np.nanmax(X[self.num].to_numpy(), axis=0), **kw)
            self.imp.fit(X[self.num])
        return self

    def _cat(self, X):
        parts, groups = [], []
        for c in self.cat:
            s = X[c].fillna(self.mode[c])
            for lv in self.levels[c]:
                parts.append((s == lv).astype(float).rename(f"{c}={lv:g}"))
                groups.append(c)
        return (pd.concat(parts, axis=1) if parts else pd.DataFrame(index=X.index)), groups

    def native(self, X):
        cat, g = self._cat(X)
        return pd.concat([X[self.num], cat], axis=1), self.num + g

    def imputed(self, X):
        cat, g = self._cat(X)
        num = pd.DataFrame(self.imp.transform(X[self.num]), index=X.index, columns=self.num)
        return pd.concat([num, cat], axis=1), self.num + g


def inner_split(Xtr, ytr):
    return train_test_split(Xtr, ytr, test_size=0.15, stratify=ytr, random_state=SEED)


def fit_xgb(Xtr, ytr, seed=SEED):
    a, b, ya, yb = inner_split(Xtr, ytr)
    m = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.03, max_depth=4, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                          eval_metric="logloss", early_stopping_rounds=50, n_jobs=8,
                          random_state=seed, tree_method="hist")
    m.fit(a, ya, eval_set=[(b, yb)], verbose=False)
    return m


def fit_lgb(Xtr, ytr, seed=SEED):
    a, b, ya, yb = inner_split(Xtr, ytr)
    m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=15, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, min_child_samples=20,
                           n_jobs=8, random_state=seed, verbose=-1)
    m.fit(a, ya, eval_set=[(b, yb)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def fit_rf(Xtr, ytr, seed=SEED):
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=8,
                                  random_state=seed).fit(Xtr, ytr)


class LR:
    def fit(self, Xtr, ytr):
        self.sc = StandardScaler().fit(Xtr)
        self.m = LogisticRegression(max_iter=5000, C=1.0).fit(self.sc.transform(Xtr), ytr)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(self.sc.transform(X))

    def shap_values(self, X):
        """Exact SHAP for a linear model with independent features (log-odds)."""
        return self.sc.transform(X) * self.m.coef_[0]


MODELS = {"Logistic Regression": ("imputed", lambda X, y: LR().fit(X, y)),
          "Random Forest": ("imputed", fit_rf),
          "XGBoost": ("native", fit_xgb),
          "LightGBM": ("native", fit_lgb)}


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def calibration(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    slope = LogisticRegression(C=1e9, max_iter=1000).fit(lp.reshape(-1, 1), y).coef_[0, 0]
    # calibration-in-the-large: intercept with slope fixed at 1 (Newton steps)
    a = 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-(lp + a)))
        a += (y - q).sum() / (q * (1 - q)).sum()
    return slope, a


def operating_points(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    ok = prec[:-1] >= 0.40
    i = np.argmax(np.where(ok, rec[:-1], -1)) if ok.any() else None
    fpr, tpr, t2 = roc_curve(y, p)
    j = np.argmax(tpr >= 0.85)
    pred = p >= t2[j]
    out = {"Sens@PPV40": rec[i] if i is not None else np.nan,
           "Thr@PPV40": thr[i] if i is not None else np.nan,
           "FlaggedFrac@PPV40": (p >= thr[i]).mean() if i is not None else np.nan,
           "Thr@Sens85": t2[j], "Spec@Sens85": 1 - fpr[j],
           "PPV@Sens85": y[pred].mean(), "NPV@Sens85": 1 - y[~pred].mean(),
           "FlaggedFrac@Sens85": pred.mean()}
    return out


def metrics(y, p):
    y = np.asarray(y)
    pred = p >= 0.5
    tp = (pred & (y == 1)).sum()
    precision = tp / max(pred.sum(), 1)
    recall = tp / y.sum()
    slope, citl = calibration(y, p)
    return {"ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p),
            "Brier": brier_score_loss(y, p), "CalSlope": slope, "CITL": citl,
            "Precision@0.5": precision, "Recall@0.5": recall,
            "F1@0.5": 2 * precision * recall / max(precision + recall, 1e-12),
            **operating_points(y, p)}


# ----------------------------------------------------------------------------
# Cross-validation
# ----------------------------------------------------------------------------
def run_cv(X, y, cols, which, tag):
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = pd.DataFrame(index=X.index, columns=which, dtype=float)
    fold_id = pd.Series(-1, index=X.index)
    fold_auc = {m: [] for m in which}
    need_imp = any(MODELS[m][0] == "imputed" for m in which)
    for f, (tr, te) in enumerate(skf.split(X, y)):
        Xtr, Xte, ytr = X.iloc[tr][cols], X.iloc[te][cols], y.iloc[tr]
        prep = Prep(cols).fit(Xtr, impute=need_imp)
        mats = {"native": (prep.native(Xtr)[0], prep.native(Xte)[0])}
        if need_imp:
            mats["imputed"] = (prep.imputed(Xtr)[0], prep.imputed(Xte)[0])
        for m in which:
            kind, fit = MODELS[m]
            a, b = mats[kind]
            p = fit(a, ytr).predict_proba(b)[:, 1]
            oof.iloc[te, oof.columns.get_loc(m)] = p
            fold_auc[m].append(roc_auc_score(y.iloc[te], p))
        fold_id.iloc[te] = f
        print(f"  [{tag}] fold {f + 1}/5 done", flush=True)
    assert oof.notna().all().all() and (fold_id >= 0).all()
    rows = []
    for m in which:
        r = {"Model": m, "Predictors": len(cols), **metrics(y.values, oof[m].values)}
        r["AUC fold mean"], r["AUC fold SD"] = np.mean(fold_auc[m]), np.std(fold_auc[m])
        rows.append(r)
    return oof, fold_id, pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# SHAP
# ----------------------------------------------------------------------------
def aggregate(values, groups, feats):
    df = pd.DataFrame(values).T.groupby(np.asarray(groups)).sum().T
    return df[feats].to_numpy()


def shap_study(X, y, feats, names):
    tr, te = next(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    Xtr, Xte, ytr = X.iloc[tr], X.iloc[te], y.iloc[tr]
    prep = Prep(feats).fit(Xtr)
    (ntr, g_nat), (nte, _) = prep.native(Xtr), prep.native(Xte)
    (itr, g_imp), (ite, _) = prep.imputed(Xtr), prep.imputed(Xte)
    people = Xte.sample(N_EXPLAIN, random_state=SEED).index
    S = {}

    def tree(model, data, groups):
        v = shap.TreeExplainer(model).shap_values(data.loc[people], check_additivity=False)
        v = np.asarray(v)
        if v.ndim == 3:  # (n, features, classes) for sklearn forests
            v = v[:, :, 1]
        return aggregate(v, groups, feats)

    S["XGBoost"] = tree(fit_xgb(ntr, ytr), nte, g_nat)
    S["LightGBM"] = tree(fit_lgb(ntr, ytr), nte, g_nat)
    S["Random Forest"] = tree(fit_rf(itr, ytr), ite, g_imp)
    S["Logistic Regression"] = aggregate(LR().fit(itr, ytr).shap_values(ite.loc[people]),
                                         g_imp, feats)
    # Noise floor: same algorithm, same data, different random seed
    S["XGBoost (seed 2)"] = tree(fit_xgb(ntr, ytr, seed=7), nte, g_nat)
    S["Random Forest (seed 2)"] = tree(fit_rf(itr, ytr, seed=7), ite, g_imp)
    print("  [shap] values computed", flush=True)

    # Correlation clusters (|Spearman| >= 0.7) so near-duplicate features count as one concept
    corr = Xtr[feats].corr(method="spearman", min_periods=200).abs().fillna(0).to_numpy()
    corr = np.clip((corr + corr.T) / 2, 0, 1)
    np.fill_diagonal(corr, 1)
    Z = linkage(1 - corr[np.triu_indices(len(feats), 1)], "average")
    cl = fcluster(Z, 0.3, "distance")
    SC = {k: pd.DataFrame(np.abs(v)).T.groupby(cl).sum().T.to_numpy() for k, v in S.items()}

    def agree(a, b):
        A, B = np.abs(a), np.abs(b)
        top1 = (A.argmax(1) == B.argmax(1)).mean()
        t5 = np.mean([len(set(np.argsort(-A[i])[:5]) & set(np.argsort(-B[i])[:5])) / 5
                      for i in range(len(A))])
        rho = np.mean([spearmanr(A[i], B[i]).statistic for i in range(len(A))])
        return top1, t5, rho

    rows = []
    keys = list(S)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            base_a, base_b = a.split(" (")[0], b.split(" (")[0]
            if "(seed 2)" in a or ("(seed 2)" in b and base_a != base_b):
                continue
            t1, t5, rho = agree(S[a], S[b])
            c1 = (SC[a].argmax(1) == SC[b].argmax(1)).mean()
            rows.append({"Model A": a, "Model B": b,
                         "Type": "same model, new seed" if base_a == base_b else "different models",
                         "Top-1 feature agreement": t1, "Top-5 overlap": t5,
                         "Mean Spearman |SHAP|": rho, "Top-1 agreement (correlated groups)": c1})
    agreement = pd.DataFrame(rows)

    main = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
    glob = pd.DataFrame({m: np.abs(S[m]).mean(0) for m in main}, index=feats)
    glob_rank = glob.rank(ascending=False).astype(int).add_suffix(" rank")
    glob = pd.concat([pd.Series(names, name="Paper variable"), glob, glob_rank], axis=1)
    glob = glob.sort_values("XGBoost rank")
    top1 = pd.DataFrame({m: pd.Series(np.asarray(feats)[np.abs(S[m]).argmax(1)])
                         .map(names).value_counts(normalize=True) for m in main}).fillna(0)
    local = pd.DataFrame({"SEQN": people})
    for m in main:
        order = np.argsort(-np.abs(S[m]), axis=1)[:, :3]
        for k in range(3):
            local[f"{m} #{k + 1}"] = [names[feats[j]] for j in order[:, k]]
    return S, people, Xte, glob, agreement, top1.sort_values("XGBoost", ascending=False), local, len(set(cl))


# ----------------------------------------------------------------------------
# Temporal validation and survival
# ----------------------------------------------------------------------------
def temporal(X, y, cyc, feats):
    tr, te = cyc.isin([1, 2]), cyc == 3
    rows = []

    def add(name, p):
        m = metrics(y[te].values, p)
        rows.append({"Model": name, "Train n": int(tr.sum()), "Test n": int(te.sum()),
                     **{k: m[k] for k in ["ROC-AUC", "PR-AUC", "Brier", "CalSlope", "CITL"]}})

    for name, cols in [("XGBoost, five predictors (paper's final set)", FIVE),
                       ("XGBoost, 134 predictors", feats)]:
        prep = Prep(cols).fit(X[tr][cols], impute=False)
        add(name, fit_xgb(prep.native(X[tr][cols])[0], y[tr])
            .predict_proba(prep.native(X[te][cols])[0])[:, 1])
    for name, cols in [("Benchmark A: age", ["RIDAGEYR"]),
                       ("Benchmark B: age + gender", ["RIDAGEYR", "RIAGENDR"]),
                       ("Benchmark C: age + physical activity", ["RIDAGEYR", "PAQ180"])]:
        prep = Prep(cols).fit(X[tr][cols], impute=False)
        a, b = prep.native(X[tr][cols])[0], prep.native(X[te][cols])[0]
        imp = SimpleImputer(strategy="median").fit(a)
        add(name, LR().fit(imp.transform(a), y[tr]).predict_proba(imp.transform(b))[:, 1])
    return pd.DataFrame(rows)


def kaplan_meier(t, e):
    """Kaplan-Meier survival estimate: returns (times, S(t)) starting at (0, 1)."""
    t, e = np.asarray(t, float), np.asarray(e, int)
    times = np.unique(t[e == 1])
    s, out = 1.0, [1.0]
    for u in times:
        s *= 1 - ((t == u) & (e == 1)).sum() / (t >= u).sum()
        out.append(s)
    return np.r_[0, times], np.asarray(out)


def logrank(t, e, g):
    """k-group log-rank test; returns the chi-square p-value."""
    from scipy.stats import chi2
    t, e, g = np.asarray(t, float), np.asarray(e, int), np.asarray(g)
    groups = np.unique(g)
    k = len(groups)
    OE, V = np.zeros(k), np.zeros((k, k))
    for u in np.unique(t[e == 1]):
        at = t >= u
        n, d = at.sum(), ((t == u) & (e == 1)).sum()
        nj = np.array([(at & (g == x)).sum() for x in groups], float)
        dj = np.array([((t == u) & (e == 1) & (g == x)).sum() for x in groups], float)
        OE += dj - d * nj / n
        if n > 1:
            V += d * (n - d) / (n - 1) * (np.diag(nj / n) - np.outer(nj, nj) / n ** 2)
    stat = OE[:-1] @ np.linalg.solve(V[:-1, :-1], OE[:-1])
    return float(chi2.sf(stat, k - 1))


def survival(X, y, fold_id):
    mort = pd.concat([pd.read_fwf(ROOT / f"data/raw/mortality/NHANES_{c}_MORT_2019_PUBLIC.dat",
                                  colspecs=[(0, 6), (15, 16), (42, 45)],
                                  names=["SEQN", "MORTSTAT", "PERMTH_INT"], na_values=["."])
                      for c in ["1999_2000", "2001_2002", "2003_2004"]])
    mort = mort.set_index(mort["SEQN"].astype(float)).reindex(X.index)
    d = pd.DataFrame({"tnt": X["SSTNT"], "bnp": X["SSBNP"], "y": y, "fold": fold_id,
                      "t": mort["PERMTH_INT"].clip(upper=60)}).dropna()
    d["event"] = d["y"]
    assert ((mort.loc[d.index, "MORTSTAT"] == 1) & (mort.loc[d.index, "PERMTH_INT"] <= 60)
            ).astype(int).equals(d["y"])
    cuts = []
    d["group"] = ""
    for f in range(5):  # Youden cut-offs from the other four folds only
        trn, tst = d[d.fold != f], d.index[d.fold == f]
        c = {}
        for v in ("tnt", "bnp"):
            fpr, tpr, thr = roc_curve(trn["y"], trn[v])
            c[v] = thr[np.argmax(tpr - fpr)]
        cuts.append({"fold": f + 1, "n_train": len(trn), "n_test": len(tst),
                     "TNT_cutoff": c["tnt"], "NT-proBNP_cutoff": c["bnp"]})
        hi_t, hi_b = d.loc[tst, "tnt"] >= c["tnt"], d.loc[tst, "bnp"] >= c["bnp"]
        d.loc[tst, "group"] = np.select([~hi_t & ~hi_b, ~hi_t & hi_b, hi_t & ~hi_b],
                                        ["Both low", "NT-proBNP high only", "TNT high only"],
                                        "Both high")
    p_value = logrank(d["t"], d["event"], d["group"])
    order = ["Both low", "NT-proBNP high only", "TNT high only", "Both high"]
    summ = d.groupby("group").agg(n=("y", "size"), deaths=("y", "sum")).loc[order]
    summ["5-year mortality %"] = 100 * summ["deaths"] / summ["n"]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = ["#2a78d6", "#1baf7a", "#eda100", "#eb6834"]
    for g, col in zip(order, colors):
        tt, ss = kaplan_meier(d.loc[d.group == g, "t"], d.loc[d.group == g, "event"])
        ax.step(np.r_[tt, 60], np.r_[ss, ss[-1]], where="post", color=col,
                label=f"{g} (n={summ.loc[g, 'n']:,})")
        ax.annotate(f"{g}  {100 * ss[-1]:.0f}%", (60, ss[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK2, fontsize=9,
                    annotation_clip=False)
    ax.set(xlim=(0, 60), xlabel="Months since interview", ylabel="Survival probability",
           title="Five-year survival by Troponin-T and NT-proBNP group")
    ax.set_ylim(top=1.0)
    ax.legend(loc="lower left")
    fig.text(0.01, 0.01, f"Cut-offs chosen by Youden index in the other four folds. "
             f"Log-rank p = {p_value:.1e}. n = {len(d):,}.", color=MUTED, fontsize=8)
    fig.subplots_adjust(right=0.74, bottom=0.17)
    fig.savefig(OUT / "fig_km_tnt_ntprobnp.png", dpi=160)
    plt.close(fig)
    return pd.DataFrame(cuts), summ.reset_index(), p_value


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_roc_cal(y, oof, table):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.8))
    for m in oof.columns:
        fpr, tpr, _ = roc_curve(y, oof[m])
        auc = table.set_index("Model").loc[m, "ROC-AUC"]
        a.plot(fpr, tpr, color=COL[m], label=f"{m} (AUC {auc:.3f})")
        q = pd.qcut(oof[m], 10, duplicates="drop")
        g = pd.DataFrame({"p": oof[m], "y": y}).groupby(q, observed=True).mean()
        b.plot(g["p"], g["y"], color=COL[m], marker="o", markersize=5, label=m)
    a.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    a.set(xlabel="False positive rate", ylabel="True positive rate",
          title="ROC curves, out-of-fold predictions")
    a.legend(loc="lower right")
    lim = max(0.6, float(b.get_xlim()[1]))
    b.plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--")
    b.set(xlabel="Predicted 5-year mortality risk (decile mean)", ylabel="Observed mortality",
          title="Calibration, out-of-fold predictions")
    b.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_roc_calibration.png", dpi=160)
    plt.close(fig)


def fig_shap(S, people, Xte, feats, names, glob):
    top = glob.head(15)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.barh(top["Paper variable"][::-1], top["XGBoost"][::-1], color=COL["XGBoost"], height=0.62)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Mean |SHAP value| (log-odds)",
           title="XGBoost: 15 most important predictors (global SHAP)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_shap_global_xgboost.png", dpi=160)
    plt.close(fig)

    plt.figure()
    shap.summary_plot(S["XGBoost"], Xte.loc[people, feats].to_numpy(),
                      feature_names=[names[f] for f in feats], max_display=15, show=False,
                      plot_size=(8, 5.5))
    plt.title("XGBoost: SHAP values per participant (top 15)")
    plt.tight_layout()
    plt.savefig(OUT / "fig_shap_beeswarm_xgboost.png", dpi=160)
    plt.close("all")


def fig_agreement(agreement):
    d = agreement.copy()
    d["pair"] = d["Model A"].str.replace("Logistic Regression", "LR").str.replace(
        "Random Forest", "RF") + " vs " + d["Model B"].str.replace(
        "Logistic Regression", "LR").str.replace("Random Forest", "RF")
    d = d.sort_values("Top-1 feature agreement")
    fig, ax = plt.subplots(figsize=(8, 4.4))
    colors = np.where(d["Type"] == "same model, new seed", "#2a78d6", "#eb6834")
    ax.barh(d["pair"], 100 * d["Top-1 feature agreement"], color=colors, height=0.62)
    for i, v in enumerate(100 * d["Top-1 feature agreement"]):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", color=INK2, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set(xlim=(0, 108), xlabel="Participants with the same top SHAP feature (%)",
           title="Do models agree on each person's most important predictor?")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#eb6834", label="Different models"),
                       Patch(color="#2a78d6", label="Same model, different random seed")],
              loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_shap_agreement.png", dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X, y, cyc, names = load()
    feats = list(X.columns)
    print(f"Data: {X.shape}, deaths {int(y.sum())}", flush=True)

    oof, fold_id, table = run_cv(X, y, feats, list(MODELS), "134 predictors")
    oof5, _, table5 = run_cv(X, y, FIVE, ["XGBoost"], "5 predictors")
    table5["Model"] = "XGBoost (five predictors)"
    allm = pd.concat([table, table5], ignore_index=True)
    allm.to_csv(OUT / "cv_metrics.csv", index=False)
    pd.concat([y.rename("death_5yr"), fold_id.rename("fold"), oof,
               oof5["XGBoost"].rename("XGBoost (five predictors)")], axis=1
              ).to_csv(OUT / "oof_predictions.csv")
    fig_roc_cal(y, oof, table)

    S, people, Xte, glob, agreement, top1, local, n_clusters = shap_study(X, y, feats, names)
    glob.to_csv(OUT / "shap_global_importance.csv")
    agreement.to_csv(OUT / "shap_agreement.csv", index=False)
    top1.to_csv(OUT / "shap_top1_feature_share.csv")
    local.to_csv(OUT / "shap_local_top3.csv", index=False)
    fig_shap(S, people, Xte, feats, names, glob)
    fig_agreement(agreement)

    temp = temporal(X, y, cyc, feats)
    temp.to_csv(OUT / "temporal_validation.csv", index=False)
    cuts, km, p = survival(X, y, fold_id)
    cuts.to_csv(OUT / "km_cutoffs.csv", index=False)
    km.to_csv(OUT / "km_groups.csv", index=False)

    json.dump({"n": len(X), "deaths": int(y.sum()), "predictors": 134, "seed": SEED,
               "provisional_definitions": PROVISIONAL, "categorical": CATEGORICAL,
               "shap_people": N_EXPLAIN, "correlation_clusters": n_clusters,
               "km_logrank_p": p}, open(OUT / "summary.json", "w"), indent=2)
    pd.set_option("display.width", 220)
    print(allm.round(3).to_string(index=False))
    print(agreement.round(3).to_string(index=False))
    print(temp.round(3).to_string(index=False))
    print(km.round(1).to_string(index=False), "\nlog-rank p", p)
    print(cuts.round(2).to_string(index=False))
    print(glob.head(12).round(3).to_string())


if __name__ == "__main__":
    main()
