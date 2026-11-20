"""Build the paper-faithful replication dataset (v0.1) from the full candidate dataset.

The full dataset (results/paper134_candidate_v0.1.pkl) uses every public measurement.
This one reproduces the choices the paper's Supplementary Table S1 counts reveal, so that
the paper's results can be replicated. Only these columns differ:

1. Laboratory files (exact Table S1 match for all nine variables).
   For 2001-2002 the paper used CDC's "Second Exam" files instead of the main exam:
     L06_2_B: cadmium, lead, total mercury, RBC folate, vitamin B12, cotinine
     L10_2_B: plasma glucose, C-peptide, insulin
   For 2003-2004 it used L06COT_C (cotinine) and L10AM_C (glucose, C-peptide, insulin),
   but not L06BMT_C (cadmium, lead, mercury) or L06NB_C (folate, B12).
   1999-2000 values are unchanged (LAB06, LAB10AM).
2. Yes/no history items (BPQ020, MCQ160B, MCQ160E, MCQ160F, DIQ010) coded 1 = "yes",
   0 = anything else (no, borderline, refused, don't know, missing). Table S1 reports no
   missing values for them and Table 1 reports them as percentages. Any rule that fills the
   gaps gives the same counts, so this coding is plausible, not proven.
3. Medication flags: missing set to 0 (Table S1 reports none missing).
4. MET score: participants whose activity records have no usable times or minutes are 0,
   as in the paper's count of 4,741 (the full dataset treats them as missing).

Not changed, because the only evidence would be that the count then matches:
income (paper 8,818 vs 8,842) and Troponin-T (paper 7,965 vs 7,966).

Usage: .venv/bin/python src/paper134_replication.py
Output: results/paper134_replication_v0.1.pkl
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FULL = ROOT / "results" / "paper134_candidate_v0.1.pkl"
OUT = ROOT / "results" / "paper134_replication_v0.1.pkl"

SECOND_EXAM = {  # target column -> (file, column in that file)
    "LBXBCD": ("L06_2_B", "LB2BCD"), "LBXBPB": ("L06_2_B", "LB2BPB"),
    "LBXTHG": ("L06_2_B", "LB2THG"), "LBXRBF": ("L06_2_B", "LB2RBF"),
    "LBXB12": ("L06_2_B", "LB2B12"), "LBXCOT": ("L06_2_B", "LB2COT"),
    "LBXGLU": ("L10_2_B", "LB2GLU"), "LBXCPSI": ("L10_2_B", "LB2CPSI"),
    "LBXIN": ("L10_2_B", "LB2IN"),
}
NO_2003_2004 = ["LBXBCD", "LBXBPB", "LBXTHG", "LBXRBF", "LBXB12"]
YES_NO = ["BPQ020", "MCQ160B", "MCQ160E", "MCQ160F", "DIQ010"]
MEDICATION = ["DERIVED_HL_MED", "DERIVED_HTN_MED", "DERIVED_DM_MED"]
ZERO_ARTIFACT = 1e-70


def read(stem):
    df = pd.read_sas(RAW / f"{stem}.xpt", format="xport")
    num = df.select_dtypes("number").columns
    df[num] = df[num].mask(df[num].abs() < ZERO_ARTIFACT, 0.0)
    return df


def main():
    d = pd.read_pickle(FULL).set_index("SEQN")
    cyc = d["SDDSRVYR"]
    changed = {}

    # 1. laboratory files
    for col, (stem, src) in SECOND_EXAM.items():
        s = read(stem).drop_duplicates("SEQN").set_index("SEQN")[src].reindex(d.index)
        before = d[col].copy()
        d.loc[cyc == 2, col] = s[cyc == 2]
        if col in NO_2003_2004:
            d.loc[cyc == 3, col] = np.nan
        changed[col] = int((~((before == d[col]) | (before.isna() & d[col].isna()))).sum())

    # 2. yes/no items: 1 = yes, 0 = everything else
    mcq = pd.concat([read(f)[["SEQN", "MCQ160E"]] for f in ("MCQ", "MCQ_B", "MCQ_C")])
    d["MCQ160E"] = mcq.drop_duplicates("SEQN").set_index("SEQN")["MCQ160E"].reindex(d.index)
    for col in YES_NO:
        before = d[col].copy()
        d[col] = (d[col] == 1).astype(float)
        changed[col] = int((before != d[col]).sum())

    # 3. medication flags: missing -> 0
    for col in MEDICATION:
        changed[col] = int(d[col].isna().sum())
        d[col] = d[col].fillna(0.0)

    # 4. MET score with no usable activity data -> 0 (paper's count)
    pa = pd.concat([read(s) for s in ("PAQIAF", "PAQIAF_B", "PAQIAF_C")], ignore_index=True)
    pa = pa[pa["SEQN"].isin(d.index)]
    met = (pa["PADMETS"] * pa["PADTIMES"] * pa["PADDURAT"]).groupby(pa["SEQN"]).sum(min_count=0)
    before = d["MET_MIN_30D_CAND"].copy()
    d["MET_MIN_30D_CAND"] = met.reindex(d.index)
    changed["MET_MIN_30D_CAND"] = int((~((before == d["MET_MIN_30D_CAND"])
                                         | (before.isna() & d["MET_MIN_30D_CAND"].isna()))).sum())

    d.attrs = {"version": "replication 0.1", "built_by": "src/paper134_replication.py"}
    d.reset_index().to_pickle(OUT)
    print(f"Wrote {OUT.relative_to(ROOT)} {d.shape}")
    print("Values changed from the full dataset:")
    for k, v in changed.items():
        print(f"  {k:18s} {v:5d}")


if __name__ == "__main__":
    main()
