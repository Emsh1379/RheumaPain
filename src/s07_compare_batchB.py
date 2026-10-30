"""Side-by-side key results: full sample (40 children) vs batch B only (33 children).
Usage: python s07_compare_batchB.py <full_results_dir> <batchB_results_dir>
"""
import os
import sys

import pandas as pd

FULL, BB = sys.argv[1], sys.argv[2]


def key_rows(R):
    out = {}
    e2 = pd.read_csv(os.path.join(R, "E2_pain.csv"))
    pick = lambda t, f, m, c: float(e2[(e2.task == t) & (e2.features == f) & (e2.model == m)][c].iloc[0])  # noqa: E731
    for f, m in (("Phase only", "LR"), ("Demo+Phase", "LR"), ("S+P", "RF"), ("Sz+P", "RF"), ("Scal+P", "RF")):
        out[f"AUC pain>=2: {f} {m}"] = pick("binary", f, m, "AUC")
    for f, m in (("Phase only", "Ridge"), ("S+P", "RF"), ("Sz+P", "Ridge"), ("Scal+P", "RF")):
        out[f"MAE: {f} {m}"] = pick("regression", f, m, "MAE")
    d = pd.read_csv(os.path.join(R, "E2_vs_phase_only.csv"))
    dcol = "delta" if "delta" in d else "dAUC"
    for f, m in (("S+P", "RF"), ("Scal+P", "RF")):
        r = d[(d.task == "binary") & (d.features == f) & (d.model == m)].iloc[0]
        out[f"dAUC vs phase: {f} {m}"] = f"{r[dcol]:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]"
    w = pd.read_csv(os.path.join(R, "E2_within_phase.csv"))
    for ph in ("rest", "exercise"):
        out[f"within-phase AUC ({ph}): Scal+P RF"] = float(
            w[(w.features == "Scal+P") & (w.model == "RF") & (w.task == "binary") & (w.phase == ph)].AUC.iloc[0])
    e3 = pd.read_csv(os.path.join(R, "E3_leakage.csv"))
    for _, r in e3.iterrows():
        out[f"RF F1 {r.task}: {r.split}"] = r.F1_macro
    p = pd.read_csv(os.path.join(R, "permutation_tests.csv"))
    for _, r in p.iterrows():
        out[f"perm p: {r.model}"] = r.p_perm
    e7 = pd.read_csv(os.path.join(R, "E7_deep.csv"))
    e7 = e7[(e7.run == "ensemble") | ~e7.design.isin(["random5_binary", "loso_binary"])]
    for _, r in e7.iterrows():
        out[f"CNN F1 {r.design}"] = r.win_F1_macro
        if r.design == "loso_binary":
            out["CNN seg AUC loso_binary"] = r.seg_AUC
            out["CNN dAUC vs phase loso_binary"] = f"{r.dAUC_vs_phase:+.3f} [{r.dAUC_lo:+.3f}, {r.dAUC_hi:+.3f}]"
    return out


t = pd.DataFrame({"full (n=40)": key_rows(FULL), "batch B (n=33)": key_rows(BB)})
t = t.map(lambda v: round(v, 3) if isinstance(v, float) else v)
t.to_csv(os.path.join(BB, "comparison_full_vs_batchB.csv"))
pd.set_option("display.width", 200, "display.max_rows", 100)
print(t.to_string())
