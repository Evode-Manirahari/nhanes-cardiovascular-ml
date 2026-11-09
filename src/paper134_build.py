"""Build the proposed Lin et al. (2026) 134-variable dataset and its evidence report.

Outputs (new, versioned; nothing existing is modified):
  results/paper134_candidate_v0.1.pkl          one row per cohort participant
  results/paper134_candidate_v0.1_columns.csv  column dictionary
  results/paper134_evidence_report_v0.1.xlsx   consolidated evidence report

Inputs (read only):
  data/processed/cohort_9957.csv     cohort SEQNs and survey cycle
  data/raw/*.xpt, dnmepi.sas7bdat    CDC source files
  data/docs/cdc/*                    CDC codebooks (cached by audit_paper_variables.py)
  data/docs/paper/lin2026_mmc1.docx  paper supplement (Table S1 counts)
  results/paper_134_feature_audit.csv  working crosswalk
  results/nhanes_master_v1.pkl       only to carry the unchanged medication flags
                                     and to report master counts

Principle: every value comes from a raw CDC file or an explicit formula. Where the
paper's definition is not established, the current column is kept and alternatives
are added as separately named *_CAND columns. Nothing is silently replaced.

Usage: .venv/bin/python src/paper134_build.py
"""

from __future__ import annotations

import hashlib
import html
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_paper_variables as A  # noqa: E402

VERSION = "0.1"
ROOT = A.ROOT
RAW = A.RAW
RESULTS = ROOT / "results"
COHORT = ROOT / "data" / "processed" / "cohort_9957.csv"
SUPP = ROOT / "data" / "docs" / "paper" / "lin2026_mmc1.docx"
OUT_PKL = RESULTS / f"paper134_candidate_v{VERSION}.pkl"
OUT_COLS = RESULTS / f"paper134_candidate_v{VERSION}_columns.csv"
OUT_REPORT = RESULTS / f"paper134_evidence_report_v{VERSION}.xlsx"

PAPER_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC12969032/"
PAPER_DOI = "https://doi.org/10.1016/j.ijcrp.2026.200595"
SUPP_URL = "https://ars.els-cdn.com/content/image/1-s2.0-S2772487526000255-mmc1.docx"
SUPP_SHA256 = "32940aeef0175b4031161fc4bd07564d5e53b75e8ca108c8a4f4745764a51971"
DNAM_URL = "https://wwwn.cdc.gov/nchs/data/nhanes/dnam/dnmepi.sas7bdat"
S1 = "Supplement Table S1 (valid n)"
ZERO_ARTIFACT = 1e-70  # pandas reads an exact SAS zero as 5.4e-79

# ----------------------------------------------------------------------------
# Raw access
# ----------------------------------------------------------------------------
_cache: dict[str, pd.DataFrame] = {}


def read_raw(stem: str) -> pd.DataFrame:
    if stem not in _cache:
        if stem == "dnmepi":
            df = pd.read_sas(RAW / "dnmepi.sas7bdat")
        else:
            df = pd.read_sas(RAW / f"{stem}.xpt", format="xport")
        df.columns = [c.upper() for c in df.columns]
        num = df.select_dtypes("number").columns
        df[num] = df[num].mask(df[num].abs() < ZERO_ARTIFACT, 0.0)
        _cache[stem] = df.drop_duplicates("SEQN").set_index("SEQN")
    return _cache[stem]


def doc_url(stem: str) -> str:
    return A.CDC.format(year=A.doc_year(stem), name=stem)


def load_cohort() -> pd.DataFrame:
    c = pd.read_csv(COHORT, usecols=["SEQN", "SDDSRVYR", "death_5yr"])
    c["SEQN"] = c["SEQN"].astype(float)
    return c.set_index("SEQN").sort_index()


def source_map(columns: list[str]) -> dict[str, dict[int, tuple[str, str]]]:
    """column -> {cycle: (file stem, CDC variable)} for direct raw variables."""
    labels = {p.stem: A.file_labels(p) for p in sorted(RAW.glob("*.xpt"))
              if p.stem != "RXQ_DRUG"}
    dnam = set(read_raw("dnmepi").columns)
    out: dict[str, dict[int, tuple[str, str]]] = {}
    for col in columns:
        key = col.upper()
        m: dict[int, tuple[str, str]] = {}
        if key in dnam:
            m = {1: ("dnmepi", key), 2: ("dnmepi", key)}
        for stem, labs in labels.items():
            if key in labs:
                for c in A.cycles_of(stem):
                    if c in m and m[c] != (stem, key):
                        raise ValueError(f"{col}: two sources for cycle {c}")
                    m[c] = (stem, key)
        for c, src in A.ALTERNATE_SOURCES.get(col, {}).items():
            m[c] = src
        if m:
            out[col] = m
    return out


def from_raw(cohort: pd.DataFrame, srcs: dict[int, tuple[str, str]]) -> pd.Series:
    parts = []
    for c, (stem, var) in srcs.items():
        idx = cohort.index[cohort["SDDSRVYR"] == c]
        parts.append(pd.to_numeric(read_raw(stem)[var], errors="coerce").reindex(idx))
    s = pd.concat(parts) if parts else pd.Series(dtype=float)
    return s.reindex(cohort.index)


def stacked(cohort: pd.DataFrame, stems: list[str], var: str) -> pd.Series:
    return from_raw(cohort, {A.cycle_of(s): (s, var) for s in stems})


# ----------------------------------------------------------------------------
# Derived variables (formulas are the only place definitions live)
# ----------------------------------------------------------------------------
def derive(cohort: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    cvx = ["CVX", "CVX_B", "CVX_C"]
    s1, s2 = stacked(cohort, cvx, "CVDS1HR"), stacked(cohort, cvx, "CVDS2HR")
    pmhr = stacked(cohort, cvx, "CVDPMHR")
    out = pd.DataFrame(index=cohort.index)
    out["MAX_HR"] = pd.concat([s1, s2], axis=1).max(axis=1)
    out["CVDPMHR"] = pmhr
    for k in (1, 2, 3):
        out[f"HRR_{k}MIN_CAND"] = out["MAX_HR"] - d[f"CVDR{k}HR"]
    out["HR_INCREMENT_CAND_REST"] = out["MAX_HR"] - d["BPXPLS"]
    out["HR_INCREMENT_CAND_PRED"] = out["MAX_HR"] - pmhr

    pa = pd.concat([read_raw(s).reset_index() for s in ["PAQIAF", "PAQIAF_B", "PAQIAF_C"]],
                   ignore_index=True)
    # PAQIAF has several rows per person; read_raw de-duplicated, so reread fully.
    pa = pd.concat([pd.read_sas(RAW / f"{s}.xpt", format="xport")
                    for s in ["PAQIAF", "PAQIAF_B", "PAQIAF_C"]], ignore_index=True)
    pa = pa[pa["SEQN"].isin(cohort.index)]
    pa["MET_MIN"] = pa["PADMETS"] * pa["PADTIMES"] * pa["PADDURAT"]
    g = pa.groupby("SEQN")
    out["MET_MIN_30D_CAND"] = g["MET_MIN"].sum(min_count=0).reindex(cohort.index)
    out["MET_SUM_CAND"] = g["PADMETS"].sum().reindex(cohort.index)
    out["PAQIAF_N_ACTIVITIES"] = g.size().reindex(cohort.index)

    # Creatinine standardized per CDC lab documentation (LAB18, L40_B, L40_C)
    scr = from_raw(cohort, {1: ("LAB18", "LBXSCR"), 2: ("L40_B", "LBDSCR"),
                            3: ("L40_C", "LBXSCR")})
    scr = scr.where(cohort["SDDSRVYR"] != 1, 1.013 * scr + 0.147)
    out["SCR_STANDARDIZED"] = scr
    fem = d["RIAGENDR"] == 2
    age = d["RIDAGEYR"]
    k = np.where(fem, 0.7, 0.9)
    out["eGFR"] = (142 * np.minimum(scr / k, 1) ** np.where(fem, -0.241, -0.302)
                   * np.maximum(scr / k, 1) ** -1.200 * 0.9938 ** age
                   * np.where(fem, 1.012, 1.0))
    out["eGFR_CKDEPI2009_CAND"] = (141 * np.minimum(scr / k, 1) ** np.where(fem, -0.329, -0.411)
                                   * np.maximum(scr / k, 1) ** -1.209 * 0.993 ** age
                                   * np.where(fem, 1.018, 1.0)
                                   * np.where(d["RIDRETH1"] == 4, 1.159, 1.0))
    return out


# Documented below-detection fill values, by cycle, with the CDC source.
BLOD = {
    "LBXBCD": {3: ([0.1], "L06BMT_C codebook: 0.1 = Below Limit of Detection")},
    "LBXBPB": {3: ([0.2], "L06BMT_C codebook: 0.2 = Below Limit of Detection")},
    "LBXTHG": {2: ([0.07], "L06BMT_C text: total mercury fill value .07 in 2001-2002"),
               3: ([0.1, 0.14], "L06BMT_C codebook: 0.1 / 0.14 = Below First/Second LOD")},
    "LBXCOT": {1: ([0.035], "L06COT_C text: 1999-2000 below-LOD value .035"),
               3: ([0.011], "L06COT_C codebook: 0.011 = Below Limit of Detection")},
    "LBXCPSI": {3: ([0.021], "L10AM_C codebook: 0.021 = Below Limit of Detection")},
    "LBXIN": {3: ([0.71], "L10AM_C codebook: 0.71 = Below Limit of Detection")},
    "LBXCRP": {3: ([0.01], "L11_C codebook: 0.01 = Below Limit of Detection")},
}


def qc_columns(cohort: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    q = pd.DataFrame(index=cohort.index)
    allc = {1: None, 2: None, 3: None}
    for var, stem in [("SSTNTLC", "SSTROP_A"), ("SSPRISTP", "SSTROP_A"),
                      ("SSBNPL", "SSBNP_A"), ("SSB2ML", "SSCARD_A"),
                      ("SSCYSTL", "SSCARD_A"), ("SSGALBL", "SSCARD_A"),
                      ("SSALB", "SSCARD_A"), ("SSGALBP", "SSCARD_A")]:
        q[var] = from_raw(cohort, {c: (stem, var) for c in allc})
    q["SSPRIS_BNP"] = from_raw(cohort, {c: ("SSBNP_A", "SSPRIS") for c in allc})
    q["SSPRIS_CARD"] = from_raw(cohort, {c: ("SSCARD_A", "SSPRIS") for c in allc})
    q["LBDCOTLC"] = from_raw(cohort, {2: ("L06_B", "LBDCOTLC")})
    for var, by_cycle in BLOD.items():
        flag = pd.Series(np.nan, index=cohort.index)
        for c, (fills, _) in by_cycle.items():
            k = (cohort["SDDSRVYR"] == c) & d[var].notna()
            flag[k] = np.isin(d.loc[k, var].round(4), fills).astype(float)
        if var == "LBXCOT":  # 2001-02 documented via comment code
            k = (cohort["SDDSRVYR"] == 2) & d[var].notna()
            flag[k] = q.loc[k, "LBDCOTLC"]
        q[f"{var}_BLOD"] = flag
    for var, stems in [("BPQ040A", ["BPQ", "BPQ_B", "BPQ_C"]),
                       ("BPQ050A", ["BPQ", "BPQ_B", "BPQ_C"]),
                       ("BPQ080", ["BPQ", "BPQ_B", "BPQ_C"]),
                       ("BPQ090D", ["BPQ", "BPQ_B", "BPQ_C"]),
                       ("BPQ100D", ["BPQ", "BPQ_B", "BPQ_C"]),
                       ("DIQ050", ["DIQ", "DIQ_B", "DIQ_C"]),
                       ("DIQ070", ["DIQ", "DIQ_B", "DIQ_C"])]:
        q[var] = stacked(cohort, stems, var)
    return q


# ----------------------------------------------------------------------------
# Supplement Table S1
# ----------------------------------------------------------------------------
def read_table_s1() -> pd.DataFrame:
    data = SUPP.read_bytes()
    if hashlib.sha256(data).hexdigest() != SUPP_SHA256:
        raise ValueError("Supplement checksum differs from the recorded original")
    x = zipfile.ZipFile(SUPP).read("word/document.xml").decode()
    tbl = re.search(r"<w:tbl>.*?</w:tbl>", x, re.S).group(0)

    def text(s):
        return html.unescape("".join(re.findall(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", s, re.S)))

    rows = []
    for tr in re.findall(r"<w:tr[ >].*?</w:tr>", tbl, re.S):
        cells = [text(c).strip() for c in re.findall(r"<w:tc>.*?</w:tc>", tr, re.S)]
        for i in (0, 3):
            if len(cells) > i + 2 and cells[i + 2].isdigit():
                rows.append((cells[i], int(cells[i + 1]), int(cells[i + 2])))
    s1 = pd.DataFrame(rows, columns=["S1 variable", "S1 missing", "S1 valid"])
    assert len(s1) == 105 and (s1["S1 missing"] + s1["S1 valid"] == 9957).all()
    return s1


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def s1_lookup(cross: pd.DataFrame, s1: pd.DataFrame) -> dict[str, int]:
    """Paper feature -> Table S1 valid n."""
    by = {}
    tc = s1[s1["S1 variable"] == "Total cholesterol"]["S1 valid"].tolist()
    for _, r in s1.iterrows():
        by[norm(r["S1 variable"])] = r["S1 valid"]
    out = {}
    for f in cross["Paper feature"]:
        if f.startswith("DNA methylation"):
            out[f] = by[norm("30 variables of DNA methylation")]
        elif f == "Total cholesterol [reference assay]":
            out[f] = tc[0]
        elif f == "Total cholesterol [biochemistry profile]":
            out[f] = tc[1]
        else:
            out[f] = by.get(norm(f))
    return out


# ----------------------------------------------------------------------------
# Decisions (evidence-backed); default applies to verified direct variables
# ----------------------------------------------------------------------------
def decisions(n: dict) -> dict[str, dict]:
    S = f"{S1}; {PAPER_DOI}"
    return {
        "RIDAGEYR": dict(transform="none; 85 = CDC top code '>= 85 years'",
                         decision="Keep. Top-coding documented.",
                         evidence=f"DEMO/DEMO_B/DEMO_C codebooks (85 = '>= 85 years of age'); {S}",
                         unresolved=""),
        "INDHHINC": dict(transform="none (CDC codes kept)",
                         decision="Keep raw codes in the candidate; 12/13 are coarse "
                                  "'Over/Under $20,000' categories, 77/99 nonresponse.",
                         evidence="DEMO codebooks; S1 valid 8818 = raw non-missing 8842 "
                                  f"minus code 99 (n=24) exactly; {S}",
                         unresolved="Paper count implies 99 set missing but 77 (n=21) "
                                    "kept - inconsistent; confirm with authors."),
        "PAQ180": dict(transform="none (CDC codes 7/9 kept; master recodes them)",
                       decision="Paper kept 7/9 as observed values; candidate follows raw.",
                       evidence="PAQ codebooks (7=Refused, 9=Don't know); S1 valid 9952 = "
                                f"raw incl. 7/9 (n=24); master has 9928; {S}",
                       unresolved="Whether 7/9 should be a category is a modelling choice."),
        **{v: dict(transform="none (CDC codes kept)",
                   decision="Keep raw; paper reports 0 missing so it assigned a value to "
                            "the CDC-missing records - rule unknown.",
                   evidence=f"{f} codebooks; S1 valid 9957 vs raw non-missing {n.get(v)}; {S}",
                   unresolved="How paper filled CDC-missing and coded 7/9"
                              + (" and 3=Borderline" if v == "DIQ010" else "") + ".")
           for v, f in [("BPQ020", "BPQ"), ("MCQ160B", "MCQ"), ("MCQ160C", "MCQ"),
                        ("MCQ160F", "MCQ"), ("DIQ010", "DIQ")]},
        "LBXSTR": dict(transform="none",
                       decision="RESOLVED: paper used biochemistry-profile LBXSTR.",
                       evidence=f"S1 valid 8542 = LBXSTR n 8542; reference LBXTR n would be "
                                f"4204 (fasting subsample); {S}",
                       unresolved="CDC advises LBXTR for most analyses (L40 docs)."),
        "LBXTC": dict(transform="none", decision="RESOLVED: first 'Total cholesterol' in S1.",
                      evidence=f"S1 valid 8551 = LBXTC n; {S}", unresolved=""),
        "LBXSCH": dict(transform="none", decision="RESOLVED: second 'Total cholesterol' in S1.",
                       evidence=f"S1 valid 8543 = LBXSCH n; {S}", unresolved=""),
        "SSGALB": dict(transform="none (g/dL)",
                       decision="RESOLVED: paper used SSGALB (g/dL), not SSGALBP (%).",
                       evidence=f"S1 valid 8012 = SSGALB n; SSGALBP n 8001; SSCARD_A codebook "
                                f"({doc_url('SSCARD_A')}); {S}",
                       unresolved="CDC recommends % (SSGALBP) and pristine-only analysis "
                                  "(SSPRIS); scientific choice, QC columns provided."),
        "SSTESTO": dict(transform="none (ng/mL)",
                        decision="RESOLVED: units ng/mL in every cycle; 'ng/dL' in "
                                 "SSCHL_C.xpt embedded label is a CDC labelling error.",
                        evidence=f"SSCHL_A/B/C codebooks list 'SSTESTO - Testosterone "
                                 f"(ng/mL)'; male medians 4.0-4.6 in all cycles; {S}",
                        unresolved=""),
        "MAX_HR": dict(transform="max(CVDS1HR, CVDS2HR)",
                       decision="RESOLVED: observed peak exercise HR; CDC CVDPMHR "
                                "(predicted max = 220 - age) excluded.",
                       evidence=f"S1 valid 1023 = max(S1,S2) n 1023; CVDPMHR n 2365; CVX "
                                f"codebook ({doc_url('CVX')}); {S}",
                       unresolved=""),
        "DERIVED_HR_INCREMENT": dict(
            proposed="HR_INCREMENT_CAND_REST | HR_INCREMENT_CAND_PRED",
            transform="MAX_HR - BPXPLS  or  MAX_HR - CVDPMHR",
            decision="NARROWED, not resolved. Both candidates give n=1023 = S1; "
                     "MAX_HR - CVDWHR (n=861) and MAX_HR - CVDS1HR (n=1022) are excluded.",
            evidence=f"S1 valid 1023; co-presence counts computed from CVX/BPX; {S}",
            unresolved="Formula not stated in paper or supplement; ask authors."),
        **{f"CVDR{k}HR": dict(
            proposed=f"HRR_{k}MIN_CAND",
            transform=f"current: CVDR{k}HR as published; proposed: MAX_HR - CVDR{k}HR",
            decision="PROPOSAL awaiting approval: S1 count equals people with BOTH a "
                     "recovery HR and MAX_HR, matching the clinical definition "
                     "HRR = peak - recovery HR.",
            evidence=f"S1 valid {s} = n(CVDR{k}HR & MAX_HR) {s}; raw CVDR{k}HR n {r}; {S}",
            unresolved="Counts alone cannot distinguish 'peak - recovery' from raw "
                       "recovery HR restricted to MAX_HR-present; confirm with authors.")
            for k, s, r in [(1, 885, 950), (2, 923, 995), (3, 424, 471)]},
        "DERIVED_MET": dict(
            proposed="MET_MIN_30D_CAND | MET_SUM_CAND",
            transform="sum over PAQIAF activities of PADMETS*PADTIMES*PADDURAT (or sum "
                      "PADMETS); missing if no PAQIAF record",
            decision="NARROWED: source and population resolved (participants with >= 1 "
                     "PAQIAF record, n=4741 = S1); aggregation formula not resolved.",
            evidence=f"S1 valid 4741 = cohort members in PAQIAF; PAQIAF codebook "
                     f"({doc_url('PAQIAF')}) PADMETS from Ainsworth 2000 compendium; {S}",
            unresolved="Exact MET formula/time window unstated; non-reporters are missing "
                       "(not 0) in the paper."),
        "eGFR": dict(
            proposed="eGFR | eGFR_CKDEPI2009_CAND",
            transform="CKD-EPI 2021 on CDC-standardized creatinine (1999-2000: "
                      "1.013*Scr + 0.147; 2001-04 none)",
            decision="Current (CKD-EPI 2021) reproduced from raw; 2009 equation provided "
                     "as alternative.",
            evidence=f"LAB18 ({doc_url('LAB18')}), L40_B, L40_C creatinine notes; S1 valid "
                     f"8541 matches; paper Table 1 weighted means not reproducible (paper "
                     f"weights unknown) so inconclusive; {S}",
            unresolved="Equation used by paper not stated."),
        **{v: dict(transform="current notebook-02 RXQ drug-class flag, unchanged",
                   decision="Not replaced. Questionnaire items supplied as evidence columns.",
                   evidence=f"S1 reports 0 missing (9957); current flag has {9957 - n.get(v, 0)} "
                            f"missing; disagreement with questionnaire ({q}); {S}",
                   unresolved="Paper definition unknown (prescription file vs questionnaire; "
                              "how missing became 0).")
           for v, q in [("DERIVED_HTN_MED", "BPQ050A"), ("DERIVED_HL_MED", "BPQ100D/BPQ090D"),
                        ("DERIVED_DM_MED", "DIQ050/DIQ070")]},
        **{v: dict(transform="none (all three cycles)",
                   decision="Keep all cycles; paper count differs.",
                   evidence=f"S1 valid {s}; our n {o}; all-cycle and 1999-2002 counts "
                            f"tested; {S}",
                   unresolved=f"Paper count ≈ 1999-2002 only ({c}); residual unexplained.")
           for v, s, o, c in [("LBXGLU", 2829, 4258, 2856), ("LBXCPSI", 2790, 4187, 2791),
                              ("LBXIN", 2791, 4188, 2793)]},
        "LBXTHG": dict(transform="none (all three cycles)",
                       decision="Keep all cycles; paper count ≈ 1999-2000 only (353).",
                       evidence=f"S1 valid 354; our n 3718; L06BMT_C/L06_B/LAB06 docs; {S}",
                       unresolved="Why paper restricted mercury; 1-person residual."),
        **{v: dict(transform="none (all three cycles)",
                   decision="Keep all cycles; paper count not reproducible.",
                   evidence=f"S1 valid {s}; our n {o}; no cycle or subsample combination "
                            f"within ±2 of S1; {S}",
                   unresolved="Paper's source/merge for this laboratory variable unknown.")
           for v, s, o in [("LBXBCD", 2994, 8721), ("LBXBPB", 2994, 8721),
                           ("LBXRBF", 2987, 8683), ("LBXB12", 2946, 8609),
                           ("LBXCOT", 5785, 8499)]},
        "SSTNT": dict(transform="none (ng/L); SSTNTLC carried as QC column",
                      decision="Keep; 25 CDC-imputed (SSTNTLC=2) and 427 'unblinded' "
                               "(SSTNTLC=1) values flagged.",
                      evidence=f"SSTROP_A codebook ({doc_url('SSTROP_A')}): 0=Measurable, "
                               f"1=Unblinded, 2=Imputed; LoD 3 ng/L; S1 7965 vs ours 7966; {S}",
                      unresolved="One participant difference vs S1 unexplained."),
        "SSBNP": dict(transform="none (pg/mL); SSBNPL carried as QC column",
                      decision="Keep; below-LLOD fill 3.54 (=5/sqrt2) and ULOD 35000 flagged.",
                      evidence=f"SSBNP_A codebook ({doc_url('SSBNP_A')}); S1 match; {S}",
                      unresolved=""),
        "LBDHDL": dict(transform="1999-2002 LBDHDL; 2003-04 LBXHDD (direct method)",
                       decision="Keep; method change documented.",
                       evidence=f"L13_C codebook ({doc_url('L13_C')}): 2003-04 HDL not "
                                f"corrected; S1 match; {S}",
                       unresolved=""),
        "LBXBAP": dict(transform="2001-02 from LBDBAP", decision="Keep; two assays in 2001-02.",
                       evidence=f"L11_B codebook ({doc_url('L11_B')}); S1 match; {S}",
                       unresolved=""),
        **{v: dict(transform=f"2001-02 from {a} (CDC cross-lab regression adjusted)",
                   decision="Keep; documented.",
                   evidence=f"L40_B codebook ({doc_url('L40_B')}) 'Adjustment of Data "
                            f"Between the Laboratories'; S1 match; {S}",
                   unresolved="")
           for v, a in [("LBXSPH", "LBDSPH"), ("LBXSTB", "LBDSTB"),
                        ("LBXSAPSI", "LBDSAPSI"), ("LBXSLDSI", "LBDSLDSI")]},
        "XY_ESTIMATION": dict(
            transform="none (categorical)",
            decision="Keep; treat as categorical.",
            evidence=f"CDC DNAm documentation Table 1 and QC code; file {DNAM_URL} "
                     f"(SHA-256 583e0660...); S1 match 2532; {S}",
            unresolved="CDC Table 1 says 1=XX but data/QC code imply 1=male; 60 cohort "
                       "members discordant with RIAGENDR."),
    }


CATEGORICAL = ["RIAGENDR", "RIDRETH1", "INDHHINC", "PAQ180", "CVDFITLV", "BPQ020",
               "MCQ160B", "MCQ160C", "MCQ160F", "DIQ010", "XY_Estimation",
               "DERIVED_HL_MED", "DERIVED_HTN_MED", "DERIVED_DM_MED"]
# Units only where a CDC source states them (source in comment).
UNIT_OVERRIDE = {
    **{c: "categorical code (see Questionnaire codes sheet)" for c in CATEGORICAL},
    "SSTESTO": "ng/mL",              # SSCHL_A/B/C codebooks
    "RIDAGEYR": "years (85 = >=85)",  # DEMO codebooks
    "TELOMEAN": "T/S ratio",          # TELO_A documentation
    "TELOSTD": "T/S ratio",           # TELO_A documentation
    "LBXRBCSI": "million cells/uL",   # L25_C label
    "MAX_HR": "per min",              # CVX labels (CVDS1HR, CVDS2HR)
    "DERIVED_HR_INCREMENT": "per min (candidates)",
    "eGFR": "mL/min/1.73 m2",         # CKD-EPI equation output
    "DERIVED_MET": "candidate: MET-min per 30 days, or summed MET",
}


def units(label: str) -> str:
    m = re.search(r"\(([^()]*)\)\s*:?\s*$", label) or re.search(
        r"[, ]\s*((?:ng|pg|mg|ug|g|nmol|mmol|umol|pmol|U|uU|fL|ml)/?\w*/?\w*|%)\s*$", label)
    return m.group(1) if m else ""


# ----------------------------------------------------------------------------
# Main build
# ----------------------------------------------------------------------------
def main() -> None:
    cohort = load_cohort()
    cross = pd.read_csv(A.CROSSWALK)
    master = pd.read_pickle(A.MASTER).set_index("SEQN").reindex(cohort.index)
    cyc = cohort["SDDSRVYR"]

    direct_cols = [c for c in cross["Dataset column"].dropna() if not c.startswith("DERIVED")
                   and c not in ("MAX_HR", "eGFR")]
    srcs = source_map(direct_cols + ["BPXPLS"])
    d = pd.DataFrame({c: from_raw(cohort, srcs[c]) for c in srcs}, index=cohort.index)
    der = derive(cohort, d)
    qc = qc_columns(cohort, d)
    meds = master[["DERIVED_HL_MED", "DERIVED_HTN_MED", "DERIVED_DM_MED"]]

    cand = pd.concat([cohort[["SDDSRVYR", "death_5yr"]], d, der, meds, qc], axis=1)
    cand = cand.loc[:, ~cand.columns.duplicated()]
    cand.attrs = {"version": VERSION, "built_by": "src/paper134_build.py"}
    cand.reset_index().to_pickle(OUT_PKL)

    # ---- Table S1 comparison -------------------------------------------------
    s1 = read_table_s1()
    s1n = s1_lookup(cross, s1)
    n_cand = {c: int(cand[c].notna().sum()) for c in cand.columns}
    n_master = {c: int(master[c].notna().sum()) for c in master.columns}
    dec = decisions(n_cand)

    # ---- Variables sheet ------------------------------------------------------
    labels = {p.stem: A.file_labels(p) for p in sorted(RAW.glob("*.xpt"))
              if p.stem != "RXQ_DRUG"}
    codebooks = {s: A.parse_codebook(A.fetch_doc(s)) for s in labels}
    dnam_labels = A.parse_dnam_doc()
    rows = []
    for i, r in cross.iterrows():
        col = r["Dataset column"] if isinstance(r["Dataset column"], str) else None
        key = col or r["Paper code"]
        dd = dec.get(key, dec.get(str(key).upper(), {}))
        proposed = dd.get("proposed", col if col in cand else "")
        first = proposed.split(" | ")[0] if proposed else ""
        src = srcs.get(col, {})
        if src:
            src_txt = "; ".join(f"{A.CYCLE[c]}: {s}.{v}" for c, (s, v) in sorted(src.items()))
        elif key in ("MAX_HR", "DERIVED_HR_INCREMENT"):
            src_txt = "CVX/CVX_B/CVX_C (CVDS1HR, CVDS2HR, CVDPMHR); BPX (BPXPLS)"
        elif key == "DERIVED_MET":
            src_txt = "PAQIAF/PAQIAF_B/PAQIAF_C (PADMETS, PADTIMES, PADDURAT)"
        elif key == "eGFR":
            src_txt = "LAB18.LBXSCR; L40_B.LBDSCR; L40_C.LBXSCR; DEMO age/sex(/race)"
        elif str(key).startswith("DERIVED_"):
            src_txt = "RXQ_RX/RXQ_RX_B/RXQ_RX_C + RXQ_DRUG (notebook 02)"
        else:
            src_txt = ""
        cdc_lab, cdc_target = set(), set()
        for c, (s, v) in src.items():
            if s == "dnmepi":
                cdc_lab.add(dnam_labels.get(v, ""))
                cdc_target.add(A.DNAM_TARGET)
            elif v in codebooks.get(s, {}):
                cdc_lab.add(codebooks[s][v]["label"])
                cdc_target.add(codebooks[s][v]["target"])
        label = " | ".join(sorted(x for x in cdc_lab if x))
        valid_c = n_cand.get(first)
        paper_v = s1n.get(r["Paper feature"])
        if first in cand and col and col in d:
            raw_ok = "matches raw (by construction; see tests)"
        elif first in der or first in meds:
            raw_ok = "derived; recomputed in tests"
        else:
            raw_ok = "not built"
        rec = {
            "#": i + 1,
            "Paper variable": r["Paper feature"],
            "Current master column": col or "",
            "Proposed candidate column(s)": proposed,
            "NHANES source by cycle": src_txt,
            "CDC definition (codebook label)": label,
            "CDC target population": " | ".join(sorted(x for x in cdc_target if x)),
            "Units": UNIT_OVERRIDE.get(key) or (units(label) if label else "")
                     or ("not stated by CDC" if label or src else ""),
            "Transformation": dd.get("transform", "none - CDC value as published"),
            "Paper S1 valid n": paper_v,
            "Candidate valid n": valid_c,
            "Master valid n": n_master.get(col) if col else None,
            "S1 match": ("yes" if paper_v == valid_c else
                         f"no ({(valid_c or 0) - paper_v:+d})") if paper_v is not None else "",
        }
        for c, name in A.CYCLE.items():
            rec[f"% observed {name}"] = (round(100 * cand.loc[cyc == c, first].notna().mean(), 1)
                                         if first in cand else None)
        rec["Raw-source validation"] = raw_ok
        rec["Decision"] = dd.get("decision", "Keep; verified against raw, CDC codebook and S1.")
        rec["Evidence / source"] = dd.get(
            "evidence", "; ".join(sorted({doc_url(s) for s, _ in src.values() if s != "dnmepi"}
                                         | ({DNAM_URL} if any(s == "dnmepi" for s, _ in src.values())
                                            else set()))) + f"; {S1}")
        rec["Unresolved issues"] = dd.get("unresolved", "")
        rows.append(rec)
    var = pd.DataFrame(rows)

    # ---- Questionnaire special codes by variable and cycle ---------------------
    qrows = []
    qvars = {"RIAGENDR": "DEMO", "RIDAGEYR": "DEMO", "RIDRETH1": "DEMO", "INDHHINC": "DEMO",
             "PAQ180": "PAQ", "BPQ020": "BPQ", "BPQ040A": "BPQ", "BPQ050A": "BPQ",
             "BPQ080": "BPQ", "BPQ090D": "BPQ", "BPQ100D": "BPQ", "MCQ160B": "MCQ",
             "MCQ160C": "MCQ", "MCQ160F": "MCQ", "DIQ010": "DIQ", "DIQ050": "DIQ",
             "DIQ070": "DIQ", "CVDFITLV": "CVX"}
    for v, base in qvars.items():
        series = cand[v] if v in cand else d[v]
        for c in (1, 2, 3):
            stem = base + {1: "", 2: "_B", 3: "_C"}[c]
            vals = codebooks[stem].get(v, {}).get("values", [])
            cs = series[cyc == c]
            for code, desc, cnt in vals:
                if re.match(r"range of values", desc, re.I):
                    continue
                if code == ".":
                    n = int(cs.isna().sum())
                else:
                    try:
                        n = int((cs == float(code)).sum())
                    except ValueError:
                        n = None
                if re.search(r"refused", desc, re.I):
                    kind = "nonresponse: refused"
                elif re.search(r"don'?t know", desc, re.I):
                    kind = "nonresponse: don't know"
                elif code == ".":
                    kind = "missing / not asked (skip pattern or not examined)"
                elif v == "DIQ010" and code == "3":
                    kind = "VALID category (Borderline) - not missing"
                elif v == "INDHHINC" and code in ("12", "13"):
                    kind = "VALID but coarse (Over/Under $20,000) - not ordinal with 1-11"
                elif v == "RIDAGEYR" and code == "85":
                    kind = "top code (>= 85 years)"
                else:
                    kind = "valid response"
                qrows.append({"Variable": v, "Cycle": A.CYCLE[c], "CDC file": stem,
                              "Code": code, "CDC description": desc[:120],
                              "CDC count (all ages)": cnt, "Cohort count": n,
                              "Classification": kind, "Source": doc_url(stem)})
    qcodes = pd.DataFrame(qrows)

    # ---- Laboratory detection limits and CDC flags ---------------------------
    lrows = []
    for v, by in BLOD.items():
        for c in (1, 2, 3):
            fills, src = by.get(c, ([], "Only general rule documented (LOD/sqrt(2)); "
                                       "fill value not stated - not inferred"))
            k = (cyc == c) & cand[v].notna()
            n_flag = int(cand.loc[k, f"{v}_BLOD"].sum()) if c in by or (v == "LBXCOT" and c == 2) else None
            if v == "LBXCOT" and c == 2:
                fills, src = ["0.035 or 0.011"], "L06_B LBDCOTLC: 1 = below detectable limit"
            lrows.append({"Variable": v, "Cycle": A.CYCLE[c], "Documented fill value(s)":
                          ", ".join(map(str, fills)) or "not documented for this cycle",
                          "Documentation": src,
                          "Cohort n measured": int(k.sum()), "Cohort n flagged below LOD": n_flag,
                          "Flag column": f"{v}_BLOD"})
    for v, codes in [("SSTNTLC", "0=Measurable, 1=Unblinded, 2=Imputed"),
                     ("SSBNPL", "0=within limits, 1=below LLOD (fill 3.54), 2=above ULOD (35000)"),
                     ("SSB2ML", "0/1/2 detection status"), ("SSCYSTL", "0/1/2 detection status"),
                     ("SSGALBL", "0/1/2 detection status"),
                     ("SSPRIS_CARD", "0=non-pristine, 1=pristine (CDC: restrict GA to pristine)")]:
        vc = cand[v].value_counts(dropna=False).sort_index()
        lrows.append({"Variable": v, "Cycle": "1999-2004", "Documented fill value(s)": codes,
                      "Documentation": "SSTROP_A / SSBNP_A / SSCARD_A codebooks",
                      "Cohort n measured": int(cand[v].notna().sum()),
                      "Cohort n flagged below LOD": "; ".join(
                          f"{'NaN' if pd.isna(k) else int(k)}={n}" for k, n in vc.items()),
                      "Flag column": v})
    labs = pd.DataFrame(lrows)

    # ---- Column dictionary ------------------------------------------------------
    roles = []
    for c in cand.columns:
        if c in ("SDDSRVYR", "death_5yr"):
            role = "cohort"
        elif c.endswith("_CAND"):
            role = "PROPOSED alternative (not approved)"
        elif c in meds:
            role = "paper variable - current definition, unchanged from master"
        elif c in der:
            role = "paper variable - derived" if c in ("MAX_HR", "eGFR") else "derivation input"
        elif c in d and c in set(cross["Dataset column"].dropna()):
            role = "paper variable - direct CDC value"
        else:
            role = "QC / evidence column"
        roles.append({"Column": c, "Role": role, "Non-missing": n_cand[c]})
    coldict = pd.DataFrame(roles)
    coldict.to_csv(OUT_COLS, index=False)

    s1cmp = s1.copy()
    unresolved = var[var["Unresolved issues"] != ""][
        ["#", "Paper variable", "Current master column", "Proposed candidate column(s)",
         "Decision", "Unresolved issues", "Evidence / source"]]

    readme = pd.DataFrame({"Item": [
        "Report version", "Paper", "Supplement source", "Supplement SHA-256",
        "Supplement content", "Paper Table 1", "CDC sources", "DNAm source",
        "Candidate dataset", "Master dataset", "Expanded datasets", "Tests",
        "Rule"], "Value": [
        VERSION, f"Lin, Lin & Chuang 2026, {PAPER_DOI} ({PAPER_URL})",
        SUPP_URL, SUPP_SHA256,
        "Tables S1-S7 and figure legends only; NO variable definitions, formulas, units, "
        "or NHANES codes. Table S1 (missing/valid n per variable) used as count evidence.",
        "Survey-weighted means consulted; paper's weighted N (1.09e8 / 7.30e6) not "
        "reproducible with standard 6-year MEC weights (1.12e8 / 9.16e6), so not used as "
        "decisive evidence.",
        "CDC NHANES codebooks cached in data/docs/cdc/ (URLs per row)",
        f"{DNAM_URL} + CDC DNAm documentation PDF",
        f"{OUT_PKL.name} (built from raw; master untouched)",
        "results/nhanes_master_v1.pkl - unchanged (checksum verified by tests)",
        "results/expanded_candidates_v1/v2.pkl - unchanged (checksum verified by tests)",
        "tests/test_paper134_candidate.py (python -m unittest)",
        "No current variable replaced; alternatives are *_CAND columns awaiting approval."]})

    with pd.ExcelWriter(OUT_REPORT) as xl:
        readme.to_excel(xl, sheet_name="README", index=False)
        var.to_excel(xl, sheet_name="Variables (134)", index=False)
        unresolved.to_excel(xl, sheet_name="Unresolved", index=False)
        s1cmp.to_excel(xl, sheet_name="Supplement Table S1", index=False)
        qcodes.to_excel(xl, sheet_name="Questionnaire codes x cycle", index=False)
        labs.to_excel(xl, sheet_name="Lab LOD & CDC flags", index=False)
        coldict.to_excel(xl, sheet_name="Candidate columns", index=False)

    print(f"Wrote {OUT_PKL.relative_to(ROOT)} {cand.shape}")
    print(f"Wrote {OUT_COLS.relative_to(ROOT)}")
    print(f"Wrote {OUT_REPORT.relative_to(ROOT)}")
    print("S1 match:", var["S1 match"].value_counts().to_dict())
    print("Unresolved rows:", len(unresolved))


if __name__ == "__main__":
    main()
