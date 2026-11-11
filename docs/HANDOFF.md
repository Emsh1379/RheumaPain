# Project state and how to continue

Written 11 Nov 2026. This file lets the work continue from this repository alone, in a new session or on another
computer. Everything that was run for the project is in the repository; nothing needed lives only in a chat session.

## Where things stand

- Article: *Wrist-worn biosignals do not generalise for pain estimation in children with rheumatic disease: a
  leakage-aware benchmark and audit of a public dataset* (S. Savabieh, M.E. Sharifi), prepared for the
  *International Journal of Medical Informatics* (single-blind review).
- The submission package (14 files) is final; the submission form was being filled in on the journal's site.
- The repository is the public code and data companion cited in the article. No Zenodo release is planned.

## What is in the repository

| Need | Where |
|---|---|
| Analysis pipeline (audit, features, experiments E1–E10, CNN) | `src/s01*.py` … `src/s08_extra.py` |
| Article figures 1–4 and A.1 | `src/s09_publication_figures.py` → `figures/` |
| Graphical abstract | `src/s10_graphical_abstract.py` → `figures/GraphicalAbstract.png` |
| Dataset download with checksum verification | `data/download_data.py`, `data/manifest.json` |
| Every result table and the extracted features | `results/` |
| Rebuild the submission package from the authors' original zip | `tools/build_submission.py`, `tools/submission_sources/` |
| Compare a fresh pipeline run with the saved results | `tools/compare_rerun.py` |
| Manuscript word counts | `tools/word_count.py` |
| Reproducibility report, data audit, TRIPOD+AI checklist, analysis plan | `docs/` |

Not in the repository, by the authors' choice: the manuscript, title page, highlights and appendix (Word files). Keep
the final `IJMI_RheumaPain_submission_final.zip` and the original `IJMI_RheumaPain_submission.zip` in your own storage.

## Commands

```bash
pip install -r requirements.txt

# 1. data (277 MB) and pipeline
python data/download_data.py /path/to/data
export RP_DATA=/path/to/data RP_OUT=/path/to/out
cd src
python s01_audit.py && python s01b_audit_detail.py && python s01c_batch_check.py
python s02_features.py 10 30 60
RP_PARTS=desc,e2,e3 python s03_experiments.py      # about 40 min on 4 CPUs; omit RP_PARTS for everything
cd ..
python tools/compare_rerun.py "$RP_OUT/results"

# 2. figures and graphical abstract from the saved tables (no raw data needed)
python src/s09_publication_figures.py results /path/to/figures
python src/s10_graphical_abstract.py /path/to/Graphical_abstract

# 3. submission package (needs pandoc and the original submission zip)
python tools/build_submission.py IJMI_RheumaPain_submission.zip /path/to/IJMI_RheumaPain_submission
python tools/word_count.py /path/to/IJMI_RheumaPain_submission/Manuscript_IJMI.docx
```

`tools/build_submission.py` was checked on 11 Nov 2026: it reproduces the text of every Word file and the pixels of
every figure in the final package exactly.

## Answers already given on the submission form

- Classifications: 1520 Replicability, robustness and transportability; 1140 Machine Learning; 370 Data Quality;
  1400 Prediction Models in practice; 420 Deep Learning (optional: 1350 Patient-generated Data; 610 Evaluation of
  Assessment Tools/frameworks).
- Main body: 2,493 words (Introduction to Conclusions, with headings); abstract 299 words.
- Scope statement (49 words): "The article evaluates health AI rigorously: a leakage-aware, TRIPOD+AI-reported
  benchmark of machine-learning and deep-learning pain estimation from wearable data in children with rheumatic
  disease. It shows how evaluation design inflates reported accuracy, audits a public clinical dataset, and gives
  reproducible methods and open code that inform trustworthy informatics practice."
- CRediT roles: M.E. Sharifi — Conceptualization, Data curation, Formal analysis, Investigation, Methodology,
  Software, Visualization, Writing – original draft. S. Savabieh — Supervision, Validation, Writing – review & editing.

## Open points

- After acceptance: add the article DOI to `README.md` and `CITATION.cff`.
- The interval of one secondary comparison (raw signals + phase, logistic regression) is borderline between runs;
  see `docs/REPRODUCIBILITY.md`.
- If reviewers ask for changes to the manuscript text, add the edits to `tools/build_submission.py` so the package
  stays reproducible.
