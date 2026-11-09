"""Tests for results/paper134_candidate_v0.1.pkl against raw CDC sources.

Independent of src/paper134_build.py: raw files are re-read here with their own
source mapping and formulas are re-implemented.

Run: .venv/bin/python -m unittest tests/test_paper134_candidate.py -v
"""

import hashlib
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CAND = ROOT / "results" / "paper134_candidate_v0.1.pkl"
MASTER = ROOT / "results" / "nhanes_master_v1.pkl"

# Checksums recorded before any v0.1 file was created (2026-11-08).
PROTECTED = {
    "results/nhanes_master_v1.pkl":
        "fab3b79c9ddc8b26826c1e419360594dc0f9c3a87b313cab87df1edd40a75b78",
    "results/expanded_candidates_v1.pkl":
        "a5fb1517faa1baf6a5f739f66a489f8db891d25b8eaad5ca52d410d8d87c3e2e",
    "results/expanded_candidates_v2.pkl":
        "3454b18f97ed4ed10883cf648d854093ef85cb4a3502bfc67f7888d982c31f47",
    "data/processed/cohort_9957.csv":
        "98336c24a6d0016e682570e869af8e3b36d5ee632bc530be3b8d2ac45ccb2f17",
    "results/paper_134_feature_audit.csv":
        "c78f6000af72fec0e11d912fee60f4348d182395198369e3e2431f038f1f7d99",
    "data/raw/DEMO.xpt":
        "a17bd1ebfd6007b8dc93f16259cb090678eb6c467a39363b7e932bba15febf89",
    "data/raw/dnmepi.sas7bdat":
        "583e0660eafac5b4b2600a18bc4ca6e8a342fd1e87a78c1b03063b14223d956f",
    "data/docs/paper/lin2026_mmc1.docx":
        "32940aeef0175b4031161fc4bd07564d5e53b75e8ca108c8a4f4745764a51971",
}

CYCLE_FILES = {  # explicit per-cycle sources, written by hand for the tests
    "DEMO": ["DEMO", "DEMO_B", "DEMO_C"], "BMX": ["BMX", "BMX_B", "BMX_C"],
    "BPX": ["BPX", "BPX_B", "BPX_C"], "BIX": ["BIX", "BIX_B", "BIX_C"],
    "CVX": ["CVX", "CVX_B", "CVX_C"], "PAQ": ["PAQ", "PAQ_B", "PAQ_C"],
    "BPQ": ["BPQ", "BPQ_B", "BPQ_C"], "MCQ": ["MCQ", "MCQ_B", "MCQ_C"],
    "DIQ": ["DIQ", "DIQ_B", "DIQ_C"], "CBC": ["LAB25", "L25_B", "L25_C"],
    "CHEM": ["LAB18", "L40_B", "L40_C"], "GH": ["LAB10", "L10_B", "L10_C"],
    "AM": ["LAB10AM", "L10AM_B", "L10AM_C"], "CRP": ["LAB11", "L11_B", "L11_C"],
    "CHOL": ["LAB13", "L13_B", "L13_C"], "METAL": ["LAB06", "L06_B", "L06BMT_C"],
    "NUTR": ["LAB06", "L06_B", "L06NB_C"], "COT": ["LAB06", "L06_B", "L06COT_C"],
    "HORM": ["SSCHL_A", "SSCHL_B", "SSCHL_C"], "TELO": ["TELO_A", "TELO_B", None],
}
VARS = {
    "DEMO": ["RIAGENDR", "RIDAGEYR", "RIDRETH1", "INDHHINC"],
    "BMX": ["BMXBMI", "BMXLEG", "BMXCALF", "BMXARML", "BMXARMC", "BMXWAIST",
            "BMXTHICR", "BMXTRI", "BMXSUB"],
    "BPX": ["BPXPLS"], "BIX": ["BIDECF", "BIDTBW", "BIDICF", "BIDFFM", "BIDFAT"],
    "CVX": ["CVDVOMAX", "CVDWHR", "CVDR1HR", "CVDR2HR", "CVDR3HR", "CVDESVO2", "CVDFITLV"],
    "PAQ": ["PAQ180"], "BPQ": ["BPQ020"], "MCQ": ["MCQ160B", "MCQ160C", "MCQ160F"],
    "DIQ": ["DIQ010"],
    "CBC": ["LBXWBCSI", "LBXLYPCT", "LBXMOPCT", "LBXNEPCT", "LBXEOPCT", "LBXBAPCT",
            "LBDLYMNO", "LBDMONO", "LBDNENO", "LBDEONO", "LBDBANO", "LBXRBCSI", "LBXHGB",
            "LBXHCT", "LBXMCVSI", "LBXMCHSI", "LBXMC", "LBXRDW", "LBXPLTSI"],
    "CHEM": ["LBXSTP", "LBXSTR", "LBXSUA", "LBXSNASI", "LBXSKSI", "LBXSCLSI", "LBXSOSSI",
             "LBXSGB", "LBXSAL", "LBXSATSI", "LBXSASSI", "LBXSBU", "LBXSCA", "LBXSCH",
             "LBXSGTSI", "LBXSGL", "LBXSIR"],
    "GH": ["LBXGH"], "AM": ["LBXGLU", "LBXCPSI", "LBXIN"], "CRP": ["LBXCRP"],
    "CHOL": ["LBXTC"], "METAL": ["LBXBCD", "LBXBPB", "LBXTHG"],
    "NUTR": ["LBXRBF", "LBXB12"], "COT": ["LBXCOT"],
    "HORM": ["SSTESTO", "SSSHBG", "SSSE2", "SS3ADLG"], "TELO": ["TELOMEAN", "TELOSTD"],
}
# Variables CDC renamed in one cycle: (target, cycle index, file, CDC name)
RENAMED = [("LBXSPH", ["LAB18", "L40_B", "L40_C"], "LBDSPH"),
           ("LBXSTB", ["LAB18", "L40_B", "L40_C"], "LBDSTB"),
           ("LBXSAPSI", ["LAB18", "L40_B", "L40_C"], "LBDSAPSI"),
           ("LBXSLDSI", ["LAB18", "L40_B", "L40_C"], "LBDSLDSI"),
           ("LBXBAP", ["LAB11", "L11_B", "L11_C"], "LBDBAP"),
           ("LBDHDL", ["LAB13", "L13_B", "L13_C"], "LBXHDD")]
SURPLUS = {"SSTROP_A": ["SSTNT", "SSTNTLC"], "SSBNP_A": ["SSBNP", "SSBNPL"],
           "SSCARD_A": ["SSB2M", "SSCYST", "SSGALB", "SSGALBP", "SSALB"]}
# Table S1 valid counts that the candidate must reproduce exactly.
S1_EXACT = {"RIAGENDR": 9957, "RIDAGEYR": 9957, "BPXPLS": 8778, "BIDECF": 1781,
            "BMXBMI": 8748, "BMXSUB": 6992, "CVDVOMAX": 1140, "CVDWHR": 922,
            "MAX_HR": 1023, "HR_INCREMENT_CAND_REST": 1023, "HR_INCREMENT_CAND_PRED": 1023,
            "HRR_1MIN_CAND": 885, "HRR_2MIN_CAND": 923, "HRR_3MIN_CAND": 424,
            "CVDESVO2": 907, "LBXGH": 8719, "LBXSTR": 8542, "LBXTC": 8551, "LBXSCH": 8543,
            "LBDHDL": 8549, "LBXBAP": 6369, "PAQ180": 9952, "MET_MIN_30D_CAND": 4741,
            "SSBNP": 7967, "SSTESTO": 601, "SSSHBG": 598, "SS3ADLG": 592, "SSB2M": 8013,
            "SSCYST": 8019, "SSGALB": 8012, "eGFR": 8541, "TELOMEAN": 5017,
            "HorvathAge": 2532, "LBXWBCSI": 8731, "LBXLYPCT": 8690}
# Documented, unexplained differences from S1: (column, paper valid, candidate valid)
S1_KNOWN_DIFF = [("INDHHINC", 8818, 8842), ("SSTNT", 7965, 7966), ("BPQ020", 9957, 9902),
                 ("LBXBCD", 2994, 8721), ("LBXTHG", 354, 3718), ("LBXGLU", 2829, 4258),
                 ("LBXCOT", 5785, 8499), ("DERIVED_HTN_MED", 9957, 9806)]


def raw(stem):
    df = pd.read_sas(RAW / f"{stem}.xpt", format="xport")
    df.columns = [c.upper() for c in df.columns]
    return df.drop_duplicates("SEQN").set_index("SEQN")


def same(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.isclose(a, b, rtol=1e-9, atol=1e-60, equal_nan=True)


class TestPaper134Candidate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = pd.read_pickle(CAND).set_index("SEQN")
        cls.cy = cls.c["SDDSRVYR"]

    def by_cycle(self, files, var):
        out = pd.Series(np.nan, index=self.c.index)
        for i, stem in enumerate(files, start=1):
            if stem is None:
                continue
            df = raw(stem)
            if var not in df:
                continue
            idx = self.c.index[self.cy == i]
            out.loc[idx] = pd.to_numeric(df[var], errors="coerce").reindex(idx).values
        return out

    def test_protected_files_unchanged(self):
        for rel, sha in PROTECTED.items():
            with self.subTest(file=rel):
                got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
                self.assertEqual(got, sha)

    def test_cohort_identity(self):
        cohort = pd.read_csv(ROOT / "data/processed/cohort_9957.csv", usecols=["SEQN", "death_5yr"])
        self.assertEqual(len(self.c), 9957)
        self.assertEqual(set(self.c.index), set(cohort.SEQN.astype(float)))
        self.assertEqual(int(self.c["death_5yr"].sum()), 1293)

    def test_direct_variables_equal_raw(self):
        for group, vars_ in VARS.items():
            for v in vars_:
                with self.subTest(var=v):
                    r = self.by_cycle(CYCLE_FILES[group], v)
                    self.assertTrue(same(self.c[v], r).all(), f"{v} differs from raw")

    def test_renamed_cycle_sources(self):
        for target, files, alt in RENAMED:
            with self.subTest(var=target):
                r = self.by_cycle(files, target)
                r2 = self.by_cycle(files, alt)
                self.assertTrue(same(self.c[target], r.fillna(r2)).all())

    def test_surplus_serum_all_cycles(self):
        for stem, vars_ in SURPLUS.items():
            df = raw(stem)
            for v in vars_:
                with self.subTest(var=v):
                    self.assertTrue(same(self.c[v], df[v].reindex(self.c.index)).all())

    def test_dnam_equals_cdc_file(self):
        d = pd.read_sas(RAW / "dnmepi.sas7bdat").set_index("SEQN").reindex(self.c.index)
        d.loc[self.cy == 3] = np.nan
        for v in ["HorvathAge", "GrimAgeMort", "Nkcell", "XY_Estimation", "NeuPP"]:
            with self.subTest(var=v):
                self.assertTrue(same(self.c[v], d[v]).all())

    def test_heart_rate_derivations(self):
        s1 = self.by_cycle(CYCLE_FILES["CVX"], "CVDS1HR")
        s2 = self.by_cycle(CYCLE_FILES["CVX"], "CVDS2HR")
        mx = np.fmax(s1, s2)
        self.assertTrue(same(self.c["MAX_HR"], mx).all())
        for k in (1, 2, 3):
            r = self.by_cycle(CYCLE_FILES["CVX"], f"CVDR{k}HR")
            self.assertTrue(same(self.c[f"HRR_{k}MIN_CAND"], mx - r).all())
        self.assertTrue(same(self.c["HR_INCREMENT_CAND_REST"], mx - self.c["BPXPLS"]).all())
        pm = self.by_cycle(CYCLE_FILES["CVX"], "CVDPMHR")
        self.assertTrue(same(pm.dropna(), (220 - self.c["RIDAGEYR"])[pm.notna()]).all())

    def test_met_candidate(self):
        pa = pd.concat([pd.read_sas(RAW / f"{s}.xpt", format="xport")
                        for s in ["PAQIAF", "PAQIAF_B", "PAQIAF_C"]])
        pa = pa[pa.SEQN.isin(self.c.index)]
        exp = (pa.PADMETS * pa.PADTIMES * pa.PADDURAT).groupby(pa.SEQN).sum()
        got = self.c["MET_MIN_30D_CAND"]
        self.assertTrue(same(got.reindex(exp.index), exp).all())
        self.assertTrue(got[~got.index.isin(exp.index)].isna().all())

    def test_egfr_ckdepi_2021(self):
        scr = self.by_cycle(["LAB18", "L40_B", "L40_C"], "LBXSCR").fillna(
            self.by_cycle(["LAB18", "L40_B", "L40_C"], "LBDSCR"))
        scr = np.where(self.cy == 1, 1.013 * scr + 0.147, scr)
        f = self.c["RIAGENDR"] == 2
        k = np.where(f, 0.7, 0.9)
        e = (142 * np.minimum(scr / k, 1) ** np.where(f, -0.241, -0.302)
             * np.maximum(scr / k, 1) ** -1.2 * 0.9938 ** self.c["RIDAGEYR"]
             * np.where(f, 1.012, 1))
        self.assertTrue(same(self.c["eGFR"], e).all())

    def test_matches_master_where_unchanged(self):
        m = pd.read_pickle(MASTER).set_index("SEQN").reindex(self.c.index)
        allowed = {"PAQ180"}  # master recoded 7/9 to missing; candidate keeps CDC codes
        for v in [x for g in VARS.values() for x in g] + ["MAX_HR", "eGFR", "SSTNT",
                                                          "DERIVED_HL_MED", "DERIVED_HTN_MED",
                                                          "DERIVED_DM_MED"]:
            with self.subTest(var=v):
                ok = same(self.c[v], m[v])
                if v in allowed:
                    self.assertTrue(m.loc[~ok, v].isna().all())
                    self.assertTrue(self.c.loc[~ok, v].isin([7, 9]).all())
                else:
                    self.assertTrue(ok.all())

    def test_table_s1_counts(self):
        for col, n in S1_EXACT.items():
            with self.subTest(col=col):
                self.assertEqual(int(self.c[col].notna().sum()), n)
        for col, paper, ours in S1_KNOWN_DIFF:
            with self.subTest(col=col, known_difference=True):
                self.assertEqual(int(self.c[col].notna().sum()), ours)

    def test_detection_flags(self):
        tnt = self.c["SSTNTLC"].value_counts().to_dict()
        self.assertEqual(tnt.get(2.0), 25)  # CDC-imputed Troponin-T values
        self.assertEqual(int((self.c["SSBNPL"] == 1).sum()), 121)
        self.assertTrue((self.c.loc[self.c["SSBNPL"] == 1, "SSBNP"].round(2) == 3.54).all())
        k = (self.cy == 3) & self.c["LBXBCD"].notna()
        self.assertTrue(((self.c.loc[k, "LBXBCD"].round(4) == 0.1)
                         == (self.c.loc[k, "LBXBCD_BLOD"] == 1)).all())
        self.assertTrue(self.c.loc[self.cy != 3, "LBXBCD_BLOD"].isna().all())


if __name__ == "__main__":
    unittest.main()
