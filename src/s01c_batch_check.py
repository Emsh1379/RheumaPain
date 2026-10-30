"""Stage 1c - compare processed 4 Hz EDA/TEMP with the raw native recordings, per processing batch."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, RES, SUBJECTS, load_processed  # noqa: E402

rows = []
for sid in SUBJECTS:
    d4 = load_processed(sid, 4, clean=False)
    batch = d4.batch.iat[0]
    for col, fname in (("eda", "EDA.csv"), ("temperature", "TEMP.csv")):
        raw = pd.read_csv(os.path.join(RAW, sid, fname), header=None)[0].to_numpy()[2:]
        p = d4[col].to_numpy()[1:] if batch == "B" else d4[col].to_numpy()
        n = min(len(p), len(raw))
        diff = np.abs(p[:n] - raw[:n])
        r = {"subject": sid, "batch": batch, "signal": col, "len_proc": len(p), "len_raw": len(raw),
             "exact_frac": float((diff < 1e-6).mean()), "corr": float(np.corrcoef(p[:n], raw[:n])[0, 1])}
        bins = {"mad_0-1s": (0, 4), "mad_1-10s": (4, 40), "mad_10-100s": (40, 400),
                "mad_100s-end10s": (400, n - 40), "mad_last10s": (n - 40, n)}  # 4 samples = 1 s
        for name, (a, b) in bins.items():
            r[name] = float(diff[a:b].mean()) if b > a else np.nan
        rows.append(r)
c = pd.DataFrame(rows)
c.to_csv(os.path.join(RES, "audit_batch_vs_raw.csv"), index=False)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(c.groupby(["batch", "signal"])[["exact_frac", "corr", "len_proc", "len_raw"]].median().round(3).to_string())
print(c[c.batch == "A"].round(3).to_string(index=False))
print(c[c.batch == "B"].head(6).round(3).to_string(index=False))

