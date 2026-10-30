# TRIPOD+AI reporting checklist

Article: *Wrist-worn biosignals do not generalise for pain estimation in children with rheumatic disease: a leakage-aware benchmark and audit of a public dataset* (International Journal of Medical Informatics, submitted).


Study type: evaluation (benchmark) of prediction models on an existing public dataset; no model is proposed for clinical use. Locations refer to the manuscript (sections, tables, figures), Appendix A and Supplementary Data S1. Reporting guideline: Collins GS et al., BMJ 2024;385:e078378.

Status: **Y** reported, **P** partly reported, **NA** not applicable.

| Item | Topic | Status | Where reported |
|---|---|---|---|
| 1 | Title | Y | Title (benchmark/evaluation; children with rheumatic disease; pain estimation) |
| 2 | Abstract | Y | Structured abstract (data, models, LOSO evaluation, AUC with 95% CI) |
| 3 | Background: health care context and rationale | Y | Section 1 |
| 4 | Objectives | Y | Section 1, last paragraph (three questions) |
| 5 | Data source and setting | P | Section 2.1 (RheumaPain v3, physiotherapy unit); recording dates are not reported in the data article |
| 6 | Participants: eligibility and treatment | P | Section 2.1 (42 children aged 5–18 y); eligibility criteria are not described in the data article |
| 7 | Data preparation, quality and cleaning | Y | Section 2.2, Section 3.1, Table 1, Appendix A.1 |
| 8 | Outcome: definition and assessment | Y | Sections 2.1 and 2.5; Appendix A.2 (Wong–Baker FACES 0–5; pain ≥ 2) |
| 9 | Predictors: definition and measurement | Y | Section 2.3; Appendix A.1 (26 features at native sampling rates; phase; demographics) |
| 10 | Sample size | Y | Fixed public dataset; minimum detectable AUC gain ≈ 0.06 (Sections 2.5, 3.3, 4.5) |
| 11 | Missing data | Y | Appendix A.1 (heart rate estimable in all windows; no imputation required) |
| 12 | Analytical methods: models, hyperparameters, validation, performance measures | Y | Sections 2.4 and 2.5; Appendix A.2 and A.3 (fixed untuned hyperparameters, LOSO, bootstrap CIs, permutation tests) |
| 13 | Class imbalance | Y | Section 2.4 (balanced class weights) |
| 14 | Fairness and subgroups | P | Section 3.2 (pain not associated with age, sex or diagnosis); model errors by subgroup not evaluated (n = 40) |
| 15 | Model output and thresholds | Y | Section 2.5 (segment-level probability of pain ≥ 2; threshold-free AUC) |
| 16 | Differences between development and evaluation data | Y | Sections 2.5 and 3.4 (subject-independent vs random window split) |
| 17 | Ethical approval and consent | Y | Section 2.1; ethics statement (title page) |
| 18a | Funding | Y | Title page (no specific funding) |
| 18b | Conflicts of interest | Y | Title page (none) |
| 18c | Protocol | P | Analysis plan written before the experiments ([`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md)); not formally published |
| 18d | Registration | NA | Not registered (secondary analysis of a public dataset) |
| 18e | Data and code sharing | Y | Data availability statement; repository link on the title page |
| 19 | Patient and public involvement | NA | None (secondary analysis of public, de-identified data) |
| 20 | Participant flow and characteristics | Y | Sections 3.1 and 3.2; Table 1 (42 → 40 children; 235 segments; 1,105 windows) |
| 21 | Numbers analysed per analysis | Y | Section 3.2; Tables 2 and 3; Supplementary Data S1 |
| 22 | Full model specification | Y | Appendix A.2; code repository (models are reference models, not for clinical use) |
| 23 | Model performance with uncertainty | P | Tables 2 and 3; Fig. 2; Supplementary Data S1 (AUC, MAE, F1 with 95% CIs); calibration not assessed because discrimination is at chance |
| 24 | Model updating | NA | No model updating |
| 25 | Interpretation | Y | Sections 4.1–4.3 |
| 26 | Limitations | Y | Section 4.5 |
| 27 | Implications for practice and research | Y | Section 4.4 |

Note: item wording is abbreviated. Transfer the status and locations to the official TRIPOD+AI form (https://www.tripod-statement.org) if the submission system requires that form.
