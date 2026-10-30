"""Shared helpers for the RheumaPain analysis (works locally and on Kaggle).

Environment variables:
  RP_DATA  folder that contains the 'RheumaPain' directory (default /kaggle/working)
  RP_OUT   output folder for results/figures (default /kaggle/working/output)
"""
import os
import re

import numpy as np
import pandas as pd

DATA = os.path.join(os.environ.get("RP_DATA", "/kaggle/working"), "RheumaPain")
OUT = os.environ.get("RP_OUT", "/kaggle/working/output")
RES = os.path.join(OUT, "results")
FIG = os.path.join(OUT, "figures")
for _d in (RES, FIG):
    os.makedirs(_d, exist_ok=True)

PROC = os.path.join(DATA, "Biosignal Data", "Processed Data")
RAW = os.path.join(DATA, "Biosignal Data", "Raw Data")
DEMO = os.path.join(DATA, "Demographic Data", "patients.xlsx")
SUBJECTS = [f"s{i:03d}" for i in range(1, 43)]
# s023/s024/s025 have identical processed signals (see audit); keep s023 only in the main analysis.
DUPLICATES = ["s024", "s025"]
SIGNALS = ["bvp", "eda", "x", "y", "z", "temperature"]


TRIM_A_S = 10  # seconds trimmed at both ends of batch-A files (resampling ringing, see audit)
# Batch A = files with float-typed pain labels in the 4 Hz folder and resampling ringing (s01_audit / s01c).
BATCH_A = ["s001", "s002", "s003", "s004", "s005", "s011", "s038"]


def load_processed(sid, hz, clean=True):
    """Load a processed file. With clean=True (default) the two processing artefacts found in the audit
    are removed: batch B files start with the E4 header row (bvp=64, eda=4, acc=32, temp=4), and batch A
    files (float pain labels: s001-s005, s011, s038) show resampling ringing in the first/last seconds."""
    df = pd.read_csv(os.path.join(PROC, f"{hz}Hz", f"{sid}.csv"))
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"exercise/rest": "phase"})
    df["phase"] = df["phase"].astype(str).str.strip().str.lower()
    df["batch"] = "A" if sid in BATCH_A else "B"
    if clean:
        r0 = df.iloc[0]
        if r0.bvp == 64 and r0.eda == 4 and r0.temperature == 4:
            df = df.iloc[1:]
        if df["batch"].iat[0] == "A":
            k = TRIM_A_S * hz
            df = df.iloc[k:len(df) - k]
        df = df.reset_index(drop=True)
    df["pain_scale"] = df["pain_scale"].round().astype(int)
    return df


def segments(df, hz=None, min_s=0.5):
    """Contiguous runs of identical (phase, pain_scale); runs shorter than min_s seconds are dropped
    (single-row label fragments at file ends). Returns start/stop row indices."""
    key = df["phase"] + "|" + df["pain_scale"].astype(str)
    change = key.ne(key.shift()).to_numpy()
    starts = np.flatnonzero(change)
    stops = np.append(starts[1:], len(df))
    seg = pd.DataFrame({"start": starts, "stop": stops,
                        "phase": df["phase"].to_numpy()[starts], "pain": df["pain_scale"].to_numpy()[starts]})
    if hz:
        seg = seg[(seg.stop - seg.start) >= min_s * hz]
    return seg.reset_index(drop=True).assign(seg=lambda s: np.arange(len(s)))


def load_demographics():
    d = pd.read_excel(DEMO)
    d.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in d.columns]
    d = d.rename(columns={"uid": "subject", "Gender": "sex", "Age": "age", "Diagnosis": "diagnosis",
                          "Pain Scale (rest)": "pain_rest", "Pain Scale (exercise)": "pain_ex",
                          "Exercise Durations (seconds)": "ex_durations"})
    d["subject"] = d["subject"].astype(str).str.strip()
    d["pain_ex_values"] = d["pain_ex"].astype(str).apply(lambda s: [int(v) for v in re.findall(r"\d+", s)])
    d["diag_group"] = d["diagnosis"].apply(diag_group)
    return d


def diag_group(x):
    """Grouping used in the data article (Fig. 7): JIA (incl. oligoarticular, RA, SpA), SLE, JDM, JSkl, Others."""
    s = str(x).lower()
    if s == "nan":
        return "Unknown"
    if any(k in s for k in ("jia", "rheumatoid", "spondylo")):
        return "JIA"
    if "lupus" in s:
        return "SLE"
    if "dermatomyositis" in s:
        return "JDM"
    if "scleroderma" in s:
        return "JSkl"
    return "Others"
