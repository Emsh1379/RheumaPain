"""Stage 1b - detailed checks on header-row artefacts, processing batches and cross-rate segment alignment."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, RES, SUBJECTS, load_processed, segments  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 30)
rows = []
for sid in SUBJECTS:
    d4, d32, d64 = (load_processed(sid, hz, clean=False) for hz in (4, 32, 64))
    raw_t = pd.read_csv(os.path.join(RAW, sid, "TEMP.csv"), header=None)[0].to_numpy()
    raw_e = pd.read_csv(os.path.join(RAW, sid, "EDA.csv"), header=None)[0].to_numpy()
    s4, s32, s64 = segments(d4), segments(d32), segments(d64)
    t = d4.temperature.to_numpy()
    e = d4.eda.to_numpy()
    # does the processed 4 Hz signal equal the raw signal (after the 2 header rows) at some offset?
    n = min(len(e), len(raw_e) - 2)
    rows.append({
        "subject": sid, "pain_dtype": d4.batch.iat[0],
        "temp_eq4_rows": int((t == 4.0).sum()), "temp_first3": np.round(t[:3], 2).tolist(),
        "eda_first3": np.round(e[:3], 3).tolist(), "raw_eda_3to5": np.round(raw_e[2:5], 3).tolist(),
        "eda_equals_raw": bool(n > 0 and np.allclose(e[:n], raw_e[2:2 + n], atol=1e-6)),
        "temp_equals_raw": bool(np.allclose(t[:min(len(t), len(raw_t) - 2)], raw_t[2:2 + min(len(t), len(raw_t) - 2)], atol=1e-6)),
        "bvp_max_abs_64": float(np.abs(d64.bvp).max()), "acc_has_32": bool((d32[["x", "y", "z"]] == 32).all(axis=1).any()),
        "n_seg": (len(s4), len(s32), len(s64)),
        "seg_dur_4": (s4.stop - s4.start).div(4).round(1).tolist(),
        "seg_dur_64": (s64.stop - s64.start).div(64).round(1).tolist() if len(s64) != len(s4) else "same-count",
    })
a = pd.DataFrame(rows)
a.to_csv(os.path.join(RES, "audit_detail.csv"), index=False)
print(a.groupby("pain_dtype").subject.apply(list).to_string())
print("\nTEMP == 4.0 rows per subject:", a.set_index("subject").temp_eq4_rows[lambda s: s > 0].to_dict())
print("\nProcessed EDA identical to raw EDA:", a.eda_equals_raw.value_counts().to_dict(),
      "| TEMP identical to raw:", a.temp_equals_raw.value_counts().to_dict())
print(a[["subject", "pain_dtype", "temp_first3", "eda_first3", "raw_eda_3to5", "eda_equals_raw", "bvp_max_abs_64"]]
      .head(12).to_string(index=False))
mis = a[a.n_seg.apply(lambda x: len(set(x)) > 1)]
print("\nSegment-count mismatch across 4/32/64 Hz:")
print(mis[["subject", "n_seg", "seg_dur_4", "seg_dur_64"]].to_string(index=False))
d = load_processed("s006", 4, clean=False)
print("\ns006 rows where temperature == 4:", d.index[d.temperature == 4.0].tolist()[:10])
print(d.iloc[[0, 1, 2]].to_string())

