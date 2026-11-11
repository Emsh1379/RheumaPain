"""Compare a fresh run of the pipeline with the result tables saved in results/.

Usage: python tools/compare_rerun.py <RP_OUT>/results
Run the pipeline first (see README, "Reproducing the results"); only tables present in both folders are compared.
Prints the largest absolute difference per table; docs/REPRODUCIBILITY.md lists the values obtained on 30 Oct 2026.
"""
import os
import sys

import numpy as np
import pandas as pd

SAVED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
NEW = sys.argv[1]
TABLES = ["features_w30.csv", "audit_subjects.csv", "audit_segments.csv", "audit_detail.csv",
          "audit_batch_vs_raw.csv", "descriptive.csv", "E2_pain.csv", "E2_vs_phase_only.csv",
          "E2_within_phase.csv", "E3_leakage.csv"]

for name in TABLES:
    if not os.path.exists(os.path.join(NEW, name)):
        print(f"{name:26s} not in the new run")
        continue
    a, b = pd.read_csv(os.path.join(SAVED, name)), pd.read_csv(os.path.join(NEW, name))
    if len(a) != len(b):
        print(f"{name:26s} ROW COUNT DIFFERS saved {len(a)} vs new {len(b)}")
        continue
    shared = [c for c in a.columns if c in b.columns]  # older saved tables can have a different column layout
    a, b = a[shared], b[shared]
    num = [c for c in a.select_dtypes("number").columns if c in b]
    diff = np.nanmax(np.abs(a[num].to_numpy(float) - b[num].to_numpy(float))) if num else 0.0
    text_same = all((a[c].astype(str) == b[c].astype(str)).all() for c in a.columns if c not in num)
    print(f"{name:26s} rows {len(a):5d}  max abs difference {diff:.2e}  text columns equal: {text_same}")

# E2 point estimates against the saved scikit-learn 1.9.1 replication
rep, new = os.path.join(SAVED, "E2_pain_local_replication.csv"), os.path.join(NEW, "E2_pain.csv")
if os.path.exists(new):
    a, b = pd.read_csv(rep), pd.read_csv(new)
    pts = ["AUC", "BalAcc", "F1_macro", "MAE", "Spearman"]
    print("E2 point estimates vs local replication:", (a[pts] - b[pts]).abs().max().round(4).to_dict())
