"""Evidence audit for the 134 predictors in Lin et al. (2026).

For every row of results/paper_134_feature_audit.csv this script records:
  * which raw NHANES file and variable supplies each survey cycle,
  * the label CDC embedded in the file and the label on CDC's codebook page,
  * whether the master dataset value equals the raw value for every participant,
  * coverage by cycle and CDC's stated target population,
  * special codes (refused / don't know / detection-limit fills) still present,
  * documented issues and anything still unresolved.

Read-only with respect to the cohort, master dataset, notebooks and models.
CDC codebook pages are cached in data/docs/cdc/ (downloaded if missing).

Usage:  .venv/bin/python src/audit_paper_variables.py
Output: results/paper_134_variable_evidence.csv
"""

from __future__ import annotations

import html
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.io.sas.sas_xport import XportReader

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DOCS = ROOT / "data" / "docs" / "cdc"
CROSSWALK = ROOT / "results" / "paper_134_feature_audit.csv"
MASTER = ROOT / "results" / "nhanes_master_v1.pkl"
OUT = ROOT / "results" / "paper_134_variable_evidence.csv"

CDC = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.htm"
CYCLE = {1: "1999-2000", 2: "2001-2002", 3: "2003-2004"}

# Cycle-specific CDC variable names used to fill a master column
# (verified identical to the master values by this script).
ALTERNATE_SOURCES = {
    "LBXSPH": {2: ("L40_B", "LBDSPH")},
    "LBXSTB": {2: ("L40_B", "LBDSTB")},
    "LBXSAPSI": {2: ("L40_B", "LBDSAPSI")},
    "LBXSLDSI": {2: ("L40_B", "LBDSLDSI")},
    "LBXBAP": {2: ("L11_B", "LBDBAP")},
    "LBDHDL": {3: ("L13_C", "LBXHDD")},
}

SPECIAL = re.compile(
    r"refused|don'?t know|fill value|below (the )?(lower )?(limit|detection)|"
    r"above (the )?(upper )?(limit|detection)|or more|or greater|top.?coded",
    re.I,
)

# Findings from CDC documentation and checks run during the audit.
NOTES = {
    "SSTESTO": "CDC codebook gives ng/mL for every cycle; the 'ng/dL' label embedded in "
               "SSCHL_C.xpt is a CDC labelling error (values on the same scale in all "
               "cycles). Men only. No conversion needed.",
    "SSGALB": "UNRESOLVED: CDC's final glycated albumin result is SSGALBP (%), "
              "= ((SSGALB/SSALB)/1.14)*100 + 2.9 (formula verified). The working map uses "
              "SSGALB (g/dL). CDC advises restricting to pristine samples (SSPRIS=1); 2 "
              "pristine samples have SSALB of about 0.12 g/dL, giving SSGALBP > 370%.",
    "LBXSTR": "UNRESOLVED: CDC advises using the reference-method LBXTR (fasting "
              "subsample) rather than biochemistry-profile LBXSTR. Paper does not say "
              "which.",
    "LBXSCH": "CDC advises the reference-method LBXTC for most analyses; the paper lists "
              "both total cholesterols, so both are kept.",
    "LBXSGL": "CDC: LBXSGL should not be used to define diabetes; reference method is "
              "LBXGLU. Paper lists both glucoses.",
    "LBDHDL": "2003-04 values come from LBXHDD (direct immunoassay, NOT corrected); "
              "1999-2002 HDL was corrected. CDC cautions about method effects on HDL "
              "across 1999-2008.",
    "LBXBAP": "2001-02 values come from LBDBAP; CDC reports two assays that cycle "
              "(Hybritech Ostase in 2001, Beckman Access Ostase in 2002).",
    "LBXSPH": "2001-02 values are CDC's cross-lab regression-adjusted LBDSPH.",
    "LBXSTB": "2001-02 values are CDC's cross-lab regression-adjusted LBDSTB.",
    "LBXSAPSI": "2001-02 values are CDC's cross-lab regression-adjusted LBDSAPSI.",
    "LBXSLDSI": "2001-02 values are CDC's cross-lab regression-adjusted LBDSLDSI.",
    "MAX_HR": "UNRESOLVED: master MAX_HR = max(CVDS1HR, CVDS2HR), the observed peak "
              "exercise HR (reproduced from raw). CDC also publishes CVDPMHR 'Predicted "
              "maximal heart rate' (= 220 - age, verified). Paper definition unknown.",
    "eGFR": "Reproduced exactly from raw: CKD-EPI 2021 (race-free) on creatinine "
            "standardized per CDC (1999-2000: 1.013*X + 0.147 [LAB18 doc]; 2001-02 "
            "LBDSCR and 2003-04 LBXSCR need no correction [L40_B, L40_C docs]). "
            "UNRESOLVED: equation used by paper (CKD-EPI 2009 with race vs 2021).",
    "DERIVED_HTN_MED": "UNRESOLVED: built from RXQ prescription classes (incl. all "
                       "diuretics and beta blockers). Disagrees with questionnaire "
                       "BPQ050A 'now taking prescribed medicine' for 1,025 people "
                       "(827 drug-yes/questionnaire-no, 198 the reverse).",
    "DERIVED_HL_MED": "UNRESOLVED: built from RXQ lipid-lowering classes. Disagrees with "
                      "BPQ100D 'now taking prescribed medicine' for 615 people.",
    "DERIVED_DM_MED": "UNRESOLVED: built from RXQ antidiabetic classes. Disagrees with "
                      "DIQ050 (insulin) or DIQ070 (pills) for 189 people.",
    "DERIVED_MET": "NOT BUILT. Source exists: PAQIAF/PAQIAF_B/PAQIAF_C (PADMETS x "
                   "PADTIMES x PADDURAT per activity). Paper formula, time window and "
                   "aggregation unknown.",
    "DERIVED_HR_INCREMENT": "NOT BUILT. Candidate CVX inputs exist (CVDWHR, CVDS1HR, "
                            "CVDS2HR, CVDPMHR) but no CDC variable or paper definition "
                            "specifies the formula.",
    "PAQ180": "Master recodes PAQ180 7/9 (refused/don't know, n=24) to missing; raw "
              "otherwise identical. Other questionnaire items keep their 7/9 codes.",
    "SSTNT": "Surplus serum 1999-2004 (SSTROP_A). LoD 3 ng/L. Comment code SSTNTLC: "
             "0=measurable, 1='unblinded' value below blinding threshold (n=427 in "
             "cohort), 2=imputed by CDC (n=25). Codes not carried into master.",
    "SSBNP": "Surplus serum 1999-2004 (SSBNP_A). Detection limits 5-35000 pg/mL; values "
             "below LLOD are CDC fill values = LLOD/sqrt(2) (3.54, n=121), above ULOD "
             "set to 35000 (n=3); flagged by SSBNPL, which is not carried into master.",
    "LBXTHG": "Coverage differs by design: 1999-2002 mercury measured mainly in "
              "women 16-49 and children; 2003-04 in all ages (see CDC Target).",
}


# Surplus-serum files that each cover all three cycles (1999-2004).
MULTI_CYCLE = {"SSBNP_A", "SSTROP_A", "SSCARD_A"}


def cycles_of(file_stem: str) -> list[int]:
    return [1, 2, 3] if file_stem in MULTI_CYCLE else [cycle_of(file_stem)]


def cycle_of(file_stem: str) -> int:
    if file_stem.endswith("_B"):
        return 2
    if file_stem.endswith("_C"):
        return 3
    return 1


def doc_year(file_stem: str) -> int:
    if file_stem.endswith("_A"):
        return 1999
    return {1: 1999, 2: 2001, 3: 2003}[cycle_of(file_stem)]


def fetch_doc(stem: str) -> str:
    DOCS.mkdir(parents=True, exist_ok=True)
    path = DOCS / f"{stem}.htm"
    if not path.exists() or path.stat().st_size < 5000:
        url = CDC.format(year=doc_year(stem), name=stem)
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                path.write_bytes(r.read())
        except Exception:
            return ""
    return path.read_text(errors="ignore")


def strip(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def parse_codebook(text: str) -> dict:
    """Return {VAR: {'label', 'target', 'values': [(code, desc, count)]}}."""
    out = {}
    blocks = re.split(r'<h3 class="vartitle" id="', text)[1:]
    for b in blocks:
        var = b.split('"', 1)[0].upper()
        lab = re.search(r"SAS Label: </dt>\s*<dd>(.*?)</dd>", b, re.S)
        tgt = re.findall(r"Target: </dt>\s*<dd>(.*?)</dd>", b, re.S)
        rows = re.findall(
            r'<td scope="row" class="values">(.*?)</td>\s*<td class="values">(.*?)</td>'
            r'\s*<td class="values" align="right">(.*?)</td>', b, re.S)
        out[var] = {
            "label": strip(lab.group(1)) if lab else "",
            "target": "; ".join(strip(t) for t in tgt),
            "values": [(strip(c), strip(d), strip(n)) for c, d, n in rows],
        }
    return out


DNAM_PDF_URL = ("https://wwwn.cdc.gov/nchs/data/nhanes/dnam/"
                "NHANES%20DNAm%20Epigenetic%20Biomarkers%20Data%20Documentation.pdf")
DNAM_TXT = DOCS / "DNAm_doc.txt"
DNAM_TARGET = ("Adults 50+ in 1999-2000 and 2001-2002 with blood for DNA; ~half of "
               "eligible non-Hispanic White and all other eligible participants")
DNAM_NOTE = ("Local dnmepi.sas7bdat is byte-identical to CDC's file "
             "(wwwn.cdc.gov/nchs/data/nhanes/dnam/dnmepi.sas7bdat, SHA-256 583e0660...). "
             "Definition from CDC DNAm documentation Table 1. Eligible participants without "
             "a usable sample appear with WTDN4YR=0 and no values. Not measured in "
             "2003-2004 or under age 50.")


def parse_dnam_doc() -> dict:
    """Variable -> label from Table 1 of CDC's DNAm documentation (pdftotext)."""
    if not DNAM_TXT.exists():
        pdf = DOCS / "NHANES_DNAm_Epigenetic_Biomarkers_Documentation.pdf"
        if not pdf.exists():
            urllib.request.urlretrieve(DNAM_PDF_URL, pdf)
        import subprocess
        subprocess.run(["pdftotext", "-layout", str(pdf), str(DNAM_TXT)], check=True)
    labels = {}
    for line in DNAM_TXT.read_text(errors="ignore").splitlines():
        m = re.match(r"^([A-Za-z0-9_]+?)(?:\d+(?:–\d+)?)?\s+(\S*\s?\S*\s?DNA methylation .+)$",
                     line.strip())
        if m:
            labels[m.group(1).upper()] = m.group(2).strip()
    return labels


def file_labels(path: Path) -> dict:
    rd = XportReader(str(path))
    labels = {}
    for f in rd.fields:
        name = f["name"].decode() if isinstance(f["name"], bytes) else f["name"]
        lab = f["label"].decode("latin1") if isinstance(f["label"], bytes) else f["label"]
        labels[name.strip().upper()] = lab.strip()
    rd.close()
    return labels


def main() -> None:
    cross = pd.read_csv(CROSSWALK)
    master = pd.read_pickle(MASTER).set_index("SEQN")
    cyc = master["SDDSRVYR"]

    xpts = sorted(RAW.glob("*.xpt"))
    raw_labels = {p.stem: file_labels(p) for p in xpts if p.stem != "RXQ_DRUG"}
    codebooks = {stem: parse_codebook(fetch_doc(stem)) for stem in raw_labels}
    cache: dict[str, pd.DataFrame] = {}

    def load(stem: str) -> pd.DataFrame:
        if stem not in cache:
            if stem == "dnmepi":
                df = pd.read_sas(RAW / "dnmepi.sas7bdat")
            else:
                df = pd.read_sas(RAW / f"{stem}.xpt", format="xport")
            df.columns = [c.upper() for c in df.columns]
            cache[stem] = df.drop_duplicates("SEQN").set_index("SEQN")
        return cache[stem]

    dnam_cols = {c.upper() for c in load("dnmepi").columns}
    dnam_labels = parse_dnam_doc()
    rows = []
    for _, r in cross.iterrows():
        col = r["Dataset column"] if isinstance(r["Dataset column"], str) else None
        key = col or r["Paper code"]
        rec = {
            "Paper variable": r["Paper feature"],
            "Master column": col or "",
            "Working map status": r["Mapping status"],
        }

        # Locate raw sources per cycle
        sources = {}
        if col:
            for stem, labs in raw_labels.items():
                if col.upper() in labs:
                    for c in cycles_of(stem):
                        sources.setdefault(c, []).append((stem, col.upper()))
            for c, (stem, var) in ALTERNATE_SOURCES.get(col, {}).items():
                sources.setdefault(c, []).append((stem, var))
            if col.upper() in dnam_cols:
                sources = {1: [("dnmepi", col.upper())], 2: [("dnmepi", col.upper())]}

        rec["Raw source by cycle"] = "; ".join(
            f"{CYCLE[c]}: " + ", ".join(f"{s}.{v}" for s, v in sorted(set(lst)))
            for c, lst in sorted(sources.items())) or "none (derived or not built)"

        # Labels from file and CDC codebook
        file_l, doc_l, targets, specials = set(), set(), set(), set()
        for lst in sources.values():
            for stem, var in lst:
                if stem == "dnmepi":
                    continue
                file_l.add(raw_labels[stem].get(var, ""))
                cb = codebooks[stem].get(var)
                if cb:
                    doc_l.add(cb["label"])
                    if cb["target"]:
                        targets.add(cb["target"])
                    for code, desc, _ in cb["values"]:
                        if SPECIAL.search(desc):
                            specials.add((code, desc))
        rec["Label in raw file"] = " | ".join(sorted(l for l in file_l if l))
        rec["Label in CDC codebook"] = " | ".join(sorted(l for l in doc_l if l))
        rec["CDC target population"] = " | ".join(sorted(targets))
        rec["CDC codebook found"] = bool(doc_l) or not sources
        if sources and not doc_l:
            rec["CDC codebook found"] = False

        # Value-level comparison master vs raw
        if col and col in master.columns and sources:
            raw = pd.Series(np.nan, index=master.index)
            for c, lst in sources.items():
                idx = master.index[cyc == c]
                for stem, var in lst:
                    s = pd.to_numeric(load(stem)[var], errors="coerce").reindex(idx)
                    raw.loc[idx] = raw.loc[idx].fillna(s)
            mv = pd.to_numeric(master[col], errors="coerce")
            same = np.isclose(mv, raw, rtol=1e-9, equal_nan=True)
            rec["Master equals raw"] = "yes" if same.all() else f"NO ({int((~same).sum())} differ)"
        elif col and col in master.columns:
            rec["Master equals raw"] = "derived - see notes"
        else:
            rec["Master equals raw"] = "column not built"

        # Coverage
        if col and col in master.columns:
            obs = master[col].notna()
            rec["N observed"] = int(obs.sum())
            for c, name in CYCLE.items():
                rec[f"% observed {name}"] = round(100 * obs[cyc == c].mean(), 1)
        else:
            rec["N observed"] = 0

        # Special codes still present in the master column
        present = []
        if col and col in master.columns:
            vals = pd.to_numeric(master[col], errors="coerce")
            for code, desc in sorted(specials):
                try:
                    v = float(code)
                except ValueError:
                    continue
                n = int((vals == v).sum())
                if n:
                    present.append(f"{code}={desc} (n={n})")
        rec["CDC special codes in codebook"] = "; ".join(f"{c}={d}" for c, d in sorted(specials))
        rec["Special codes still in master"] = "; ".join(present)

        note = NOTES.get(col or "", NOTES.get(key, ""))
        if col and col.upper() in dnam_cols:
            label = dnam_labels.get(col.upper(), "")
            rec["Label in CDC codebook"] = label
            rec["CDC target population"] = DNAM_TARGET
            rec["CDC codebook found"] = bool(label)
            note = DNAM_NOTE
            if col.upper() == "XY_ESTIMATION":
                note += (" UNRESOLVED: CDC Table 1 says 1=XX, 2=XY, but CDC's own QC code "
                         "uses abs(XY_Estimation - RIAGENDR) and the data agree with "
                         "RIAGENDR coding (1=male) for 2,472/2,532 cohort members. 60 "
                         "participants (2.4%) have DNAm-predicted sex discordant with "
                         "recorded sex. Categorical, not continuous.")
        rec["Notes / evidence"] = note

        unresolved = (
            "UNRESOLVED" in note or "NOT BUILT" in note
            or rec["Master equals raw"].startswith("NO") and col != "PAQ180"
            or not rec["CDC codebook found"]
        )
        rec["Status"] = (
            "NOT BUILT" if rec["Master equals raw"] == "column not built"
            else "UNRESOLVED" if unresolved
            else "VERIFIED vs raw + CDC doc (with notes)" if note or present
            else "VERIFIED vs raw + CDC doc"
        )
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT.relative_to(ROOT)} ({len(out)} rows)")
    print(out["Status"].value_counts().to_string())


if __name__ == "__main__":
    main()
