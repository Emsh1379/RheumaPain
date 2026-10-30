"""Stage 1 - data audit: structure, label consistency, duplicates, signal plausibility."""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, RES, SIGNALS, SUBJECTS, load_demographics, load_processed, segments  # noqa: E402

demo = load_demographics().set_index("subject")
subj_rows, seg_rows, hashes = [], [], {}

for sid in SUBJECTS:
    d4, d32, d64 = (load_processed(sid, hz, clean=False) for hz in (4, 32, 64))
    s4, s32, s64 = segments(d4), segments(d32), segments(d64)
    hashes[sid] = hashlib.md5(d4[SIGNALS].to_numpy().tobytes()).hexdigest()

    raw_eda = pd.read_csv(os.path.join(RAW, sid, "EDA.csv"), header=None)[0].to_numpy()
    raw_eda_vals = raw_eda[2:]
    acc_g = np.sqrt((d32[["x", "y", "z"]] ** 2).sum(axis=1)) / 64.0

    dm = demo.loc[sid]
    pains_rest = sorted(set(s4.loc[s4.phase == "rest", "pain"]))
    pains_ex = sorted(set(s4.loc[s4.phase == "exercise", "pain"]))
    subj_rows.append({
        "subject": sid, "rows_4hz": len(d4), "dur_s": len(d4) / 4,
        "ratio_32_4": len(d32) / len(d4), "ratio_64_4": len(d64) / len(d4),
        "n_seg_4": len(s4), "n_seg_32": len(s32), "n_seg_64": len(s64),
        "rest_s": (d4.phase == "rest").sum() / 4, "exercise_s": (d4.phase == "exercise").sum() / 4,
        "pain_rest_rows": pains_rest, "pain_ex_rows": pains_ex,
        "pain_rest_xlsx": dm.pain_rest, "pain_ex_xlsx": dm.pain_ex_values,
        "rest_label_match": pains_rest == [int(dm.pain_rest)] if pains_rest else None,
        "ex_label_match": pains_ex == sorted(set(dm.pain_ex_values)) if pains_ex else None,
        "nan_rows": int(d4[SIGNALS].isna().any(axis=1).sum()),
        "eda_min": d4.eda.min(), "eda_neg_frac": (d4.eda < 0).mean(),
        "raw_eda_min": raw_eda_vals.min(), "raw_eda_dur_s": len(raw_eda_vals) / 4,
        "temp_min": d4.temperature.min(), "temp_lt25_frac": (d4.temperature < 25).mean(),
        "acc_g_median": acc_g.median(), "bvp_std": d64.bvp.std(),
        "sex": dm.sex, "age": dm.age, "diag_group": dm.diag_group,
    })
    for _, r in s4.iterrows():
        seg_rows.append({"subject": sid, "seg": r.seg, "phase": r.phase, "pain": r.pain,
                         "dur_s": (r.stop - r.start) / 4})

subj = pd.DataFrame(subj_rows)
seg = pd.DataFrame(seg_rows)
subj["md5_4hz"] = subj.subject.map(hashes)
dups = subj[subj.duplicated("md5_4hz", keep=False)].groupby("md5_4hz").subject.apply(list).tolist()
subj.to_csv(os.path.join(RES, "audit_subjects.csv"), index=False)
seg.to_csv(os.path.join(RES, "audit_segments.csv"), index=False)

pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 100)
print("Subjects:", len(subj), "| total duration (h): %.2f" % (subj.dur_s.sum() / 3600))
print("Rows per pain level at 4 Hz:", pd.concat([load_processed(s, 4, clean=False).pain_scale for s in SUBJECTS])
      .value_counts().sort_index().to_dict())
print("Identical processed signals:", dups)
print("Rate ratios 32/4 and 64/4 (min-max): %.3f-%.3f, %.3f-%.3f" % (
    subj.ratio_32_4.min(), subj.ratio_32_4.max(), subj.ratio_64_4.min(), subj.ratio_64_4.max()))
print("Segment count equal across 4/32/64 Hz:", ((subj.n_seg_4 == subj.n_seg_32) & (subj.n_seg_4 == subj.n_seg_64)).all())
print("Segments per subject:", subj.n_seg_4.describe().round(1).to_dict())
print("Segment duration (s):", seg.dur_s.describe().round(1).to_dict())
print("Rest label matches xlsx:", subj.rest_label_match.value_counts(dropna=False).to_dict(),
      "| exercise:", subj.ex_label_match.value_counts(dropna=False).to_dict())
print(subj.loc[(subj.rest_label_match == False) | (subj.ex_label_match == False),  # noqa: E712
               ["subject", "pain_rest_rows", "pain_rest_xlsx", "pain_ex_rows", "pain_ex_xlsx"]].to_string())
print("Subjects with rest only / exercise only:",
      subj.loc[subj.exercise_s == 0, "subject"].tolist(), subj.loc[subj.rest_s == 0, "subject"].tolist())
print("Subjects with negative processed EDA:", int((subj.eda_neg_frac > 0).sum()),
      "| raw EDA min over subjects: %.3f" % subj.raw_eda_min.min())
print("Processed vs raw EDA duration ratio (median): %.2f" % (subj.dur_s / subj.raw_eda_dur_s).median())
print("Subjects with TEMP < 25 C in any row:", int((subj.temp_lt25_frac > 0).sum()),
      "| median ACC magnitude (g): %.2f" % subj.acc_g_median.median())
print(subj[["subject", "dur_s", "n_seg_4", "rest_s", "exercise_s", "pain_rest_rows", "pain_ex_rows",
            "eda_neg_frac", "temp_min"]].to_string(index=False))

