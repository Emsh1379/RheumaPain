# Data audit of RheumaPain v3

The released processed files were compared with the raw E4 recordings (`src/s01_audit.py`, `s01b_audit_detail.py`,
`s01c_batch_check.py`). These findings are not reported in the data article. They concern the processed files and the
article text; we found no reason to doubt the recordings themselves.

| Issue | Evidence | Handling in this study |
|---|---|---|
| E4 header row stored as data (35 subjects, "batch B") | First processed row = E4 header values (BVP 64, EDA 4, ACC 32, TEMP 4), which appear as 4 °C temperatures | First row dropped; afterwards processed ≈ raw (r ≈ 1.00) |
| Resampling distortion (7 subjects, "batch A": s001–s005, s011, s038; float-typed labels) | TEMP error 3.7–5.2 °C in the first second, 0.03–0.18 °C in seconds 1–10; negative EDA | 10 s trimmed at both ends |
| Identical processed signals for s023, s024, s025 | Identical MD5 of all signal columns at every rate; the raw files differ | s024 and s025 excluded |
| One-row label fragments at file end (s005, s038, 64 Hz) | 0 s trailing segment | Segments < 0.5 s dropped |
| Total duration 6.02 h (5.79 h without duplicates) vs "≈ 5.5 h" in the article; rows per pain level differ from the article's Table 2 | `results/audit_subjects.csv` | Reported |
| Pain labels in processed files vs `patients.xlsx` | Agree for 42/42 subjects, including multi-valued exercise scores such as "0, 2" | — |
| All 7 batch-A children report identical rest and exercise pain | `results/E4_reactivity.csv` | Sensitivity analysis without batch A (`results/batchB/`) |

**Analysis set after cleaning:** 40 children, 235 phase segments (126 rest, 109 exercise), 1,105 windows of 30 s with 50 % overlap.

Audit tables: `results/audit_subjects.csv`, `audit_segments.csv`, `audit_detail.csv`, `audit_batch_vs_raw.csv`.
Cleaning implementation: `load_processed()` and `segments()` in `src/common.py`.
