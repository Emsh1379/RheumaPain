# RheumaPain leakage-aware benchmark

**Wrist-worn biosignals do not generalise for pain estimation in children with rheumatic disease: a leakage-aware benchmark and audit of a public dataset**

Shekoofeh Savabieh<sup>a,b</sup>, Mohammad Emad Sharifi<sup>c,d,*</sup>

<sup>a</sup> Children Growth Disorder Research Center, Shahid Sadoughi University of Medical Sciences, Yazd, Iran  
<sup>b</sup> Department of Pediatrics, Shahid Sadoughi University of Medical Sciences, Yazd, Iran  
<sup>c</sup> Mother and Newborn Health Research Center, Shahid Sadoughi University of Medical Sciences, Yazd, Iran  
<sup>d</sup> School of Dentistry, Tehran University of Medical Sciences, Tehran, Iran  
<sup>*</sup> Corresponding author: Mohammad Emad Sharifi ([ORCID 0000-0003-2846-6964](https://orcid.org/0000-0003-2846-6964)), emad.sharifi2001@gmail.com

Code, extracted features, result tables and figures for the original research article submitted to the
*International Journal of Medical Informatics* (Elsevier).

[![License: MIT](https://img.shields.io/badge/code%20license-MIT-blue.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/dataset-RheumaPain%20v3-informational)](https://doi.org/10.17632/y6pgjdj22f.3)
[![Reporting](https://img.shields.io/badge/reporting-TRIPOD%2BAI-success)](docs/TRIPOD_AI_checklist.md)

> **Status:** submitted to the *International Journal of Medical Informatics*. The article DOI will be added on acceptance.

---

## At a glance

Wearable sensors are often proposed as an objective measure of pain. We tested this claim on
**RheumaPain**, the first public dataset of wrist-worn biosignals from children with rheumatic
disease (42 children aged 5–18 years, Empatica E4, rest and exercise phases during physiotherapy,
Wong–Baker FACES pain ratings 0–5). Every model was tested on **children it had never seen during
training** (leave-one-subject-out).

| Question | Answer |
|---|---|
| Are the released files fit for analysis? | Only after correction. Device header stored as data in 35 files, resampling distortion in 7 files, one child triplicated (s023 = s024 = s025). |
| Do wrist signals estimate pain in new children? | **No improvement over context.** The rule "exercise → pain" reaches AUC 0.71; no feature model or 1D-CNN beats it, and within a phase pain ranking is at chance. |
| Why do subject-mixed evaluations show ~90 % accuracy? | **Subject leakage.** Random window splits raise macro-F1 from 0.59 to 0.91, because a 30 s window identifies the child with 90 % accuracy. |
| Are the findings robust? | Yes. Unchanged in the 33 consistently processed children; the study could detect AUC gains of about 0.06. |

### Summary table

| What was already known on the topic | What this study added to our knowledge |
|---|---|
| • Wrist-worn autonomic signals are proposed as objective pain measures, but pediatric chronic-pain evidence is scarce. | • In held-out children with rheumatic disease, no feature or deep model beat a phase-only rule; within a phase, pain ranking was at chance. |
| • Evaluations that place windows from the same person in training and test data can overestimate accuracy. | • Random window splits inflated macro-F1 from 0.59 to 0.91 because 30-s windows identified the child; a data audit found errors in the public release. |

---

## Key results

All numbers below are traceable to files in [`results/`](results/) (see [`results/RESULTS.md`](results/RESULTS.md)); complete results are in
[`results/Supplementary_Data_S1_results.xlsx`](results/Supplementary_Data_S1_results.xlsx) (Supplementary Data S1 of the article).
Evaluation: leave-one-subject-out (LOSO); prediction unit = phase segment; 95 % CIs from 500 subject-level bootstraps.

**Pain ≥ 2 vs 0–1, segment level, LOSO** (40 children, 235 segments, 1,105 windows of 30 s; article Table 2)

| Features | Model | AUC [95 % CI] | ΔAUC vs phase only [95 % CI] |
|---|---|---|---|
| **Phase only (exercise → pain)** | rule | **0.71 [0.64, 0.79]** | reference |
| Rest-calibrated signals + phase | RF | 0.71 [0.62, 0.80] | +0.00 [−0.04, +0.05] |
| Raw signals + phase | RF | 0.61 [0.50, 0.71] | −0.10 [−0.16, −0.04] |
| Raw signals only | LR | 0.48 [0.35, 0.60] | – |
| 1D-CNN on raw signals + phase | CNN | 0.53 [0.41, 0.66] | −0.18 [−0.30, −0.06] |

**Effect of the evaluation design (window-level macro-F1; article Fig. 3)**

| Model | Random 5-fold windows (binary / 6-class) | Leave-one-child-out (binary / 6-class) |
|---|---|---|
| Random forest, 26 features | 0.91 / 0.85 | 0.59 / 0.24 |
| 1D-CNN, raw signals | 0.87 / 0.60 | 0.57 / 0.19 |

Further results: within-phase permutation tests p = 0.20–1.00 for all five models; the child is identified
from one 30 s window with 89.7 % accuracy (chance 2.5 %); learning curve plateaus at the phase-only level;
minimum detectable ΔAUC ≈ 0.06; an apparent movement–pain association is driven by the
differently processed batch-A files and is not robust (q = 0.23 in batch B).

---

## Repository structure

```
RheumaPain/
├── src/                     Analysis pipeline (Python)
│   ├── common.py            Paths, loading and the cleaning rules from the audit
│   ├── s01_audit.py         Data audit: durations, labels, duplicates, plausibility
│   ├── s01b_audit_detail.py Processed vs raw comparison per subject
│   ├── s01c_batch_check.py  Batch A / batch B processing check
│   ├── s02_features.py      26 window features at native sampling rates (10/30/60 s)
│   ├── s03_experiments.py   E1–E5, permutation tests, mixed models, batch-B subset
│   ├── s04_diagnosis.py     E6 diagnosis task (JIA vs other)
│   ├── s05_deep.py          E7 multi-branch 1D-CNN (PyTorch)
│   ├── s06_fig_leakage.py   RF vs CNN leakage comparison table (E7_leakage_comparison.csv)
│   ├── s07_compare_batchB.py Full sample vs batch-B comparison table
│   ├── s08_extra.py         E8 learning curve, E9 subject identification, E10 minimum detectable effect
│   ├── s09_publication_figures.py  Article Figures 1–4 and A.1 (600 dpi TIFF/PNG + vector PDF)
│   ├── s10_graphical_abstract.py   Graphical abstract
│   └── run_all.py           Runs audit → features → main experiments
├── results/                 Every result table used in the paper (CSV)
│   ├── RESULTS.md           Narrative of all experiments with numbers and provenance
│   ├── Supplementary_Data_S1_results.xlsx   Supplementary Data S1 (complete results)
│   ├── features_w{10,30,60}.csv         Extracted window features (rerun experiments without raw data)
│   └── batchB/              Sensitivity analysis without the 7 batch-A children
├── figures/                 Article figures: Figure1–4, FigureA1 and the graphical abstract (600 dpi PNG)
├── data/
│   ├── manifest.json        All 547 files of RheumaPain v3 with download URL and SHA-256
│   ├── download_data.py     Downloads the dataset and verifies every checksum
│   └── README.md            How to obtain and verify the dataset
├── docs/
│   ├── ANALYSIS_PLAN.md     Analysis plan written before the experiments
│   ├── TRIPOD_AI_checklist.md   Reporting checklist, item by item
│   ├── data_audit.md        Audit findings and cleaning rules
│   └── REPRODUCIBILITY.md   Independent re-run of the pipeline and agreement with the paper
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Experiments

| ID | Question | Script | Main outputs |
|---|---|---|---|
| Audit | Are the processed files consistent with the raw recordings? | `s01*.py` | `audit_*.csv` |
| E1 | Can exercise vs rest be recognised from the wrist? | `s03_experiments.py` | `E1_phase.csv` |
| E2 | Do signals estimate pain beyond the phase? (Table 2, Fig. 2) | `s03_experiments.py` | `E2_pain.csv`, `E2_vs_phase_only.csv`, `E2_within_phase.csv` |
| E3 | How much does a random window split inflate scores? (Fig. 3) | `s03_experiments.py` | `E3_leakage.csv` |
| E4 | Does exercise–rest reactivity track pain change? (Fig. A.1) | `s03_experiments.py` | `E4_reactivity.csv`, `mixed_models.csv` |
| E5 | Modality and window-length ablations (Table A.2) | `s03_experiments.py` | `E5_ablation.csv` |
| Perm. | Within-phase label permutation tests (Table A.1) | `s03_experiments.py` | `permutation_tests.csv` |
| E6 | Diagnosis group from signals | `s04_diagnosis.py` | `E6_diagnosis.csv` |
| E7 | Deep learning on raw signals (Table 2, Fig. 3) | `s05_deep.py` | `E7_deep.csv`, `E7_leakage_comparison.csv` |
| E8 | Learning curve (5–39 training children; Fig. 4) | `s08_extra.py` | `E8_learning_curve.csv` |
| E9 | Can a window identify the child? | `s08_extra.py` | `E9_subject_identification.csv`, `E9_between_child_variance.csv` |
| E10 | Minimum detectable AUC gain | `s08_extra.py` | `E10_minimum_detectable_effect.csv` |
| Sens. | All main analyses in the 33 batch-B children (Table 3) | `RP_SUBSET=batchB`, `s07_compare_batchB.py` | `results/batchB/` |

## Figures

| Article | File | Content |
|---|---|---|
| Fig. 1 | [`Figure1.png`](figures/Figure1.png) | Self-reported pain (Wong–Baker FACES 0–5) per phase segment at rest (n = 126) and during exercise (n = 109) in 40 children. |
| Fig. 2 | [`Figure2.png`](figures/Figure2.png) | Segment-level AUC (95 % subject-bootstrap CI) for pain ≥ 2 under LOSO, by feature set and model; dotted line = phase-only rule, dashed = chance. |
| Fig. 3 | [`Figure3.png`](figures/Figure3.png) | Window-level macro-F1 of the random forest and the 1D-CNN under a random window split and under leave-one-child-out evaluation. |
| Fig. 4 | [`Figure4.png`](figures/Figure4.png) | LOSO AUC vs number of children in the training set (mean and range over 10 draws). |
| Fig. A.1 | [`FigureA1.png`](figures/FigureA1.png) | Rest-to-exercise change in movement variability and mean heart rate vs change in pain per child; triangles = batch A. |
| Graphical abstract | [`GraphicalAbstract.png`](figures/GraphicalAbstract.png) | Study summary (drawn by `src/s10_graphical_abstract.py`). |

`python src/s09_publication_figures.py results <out_dir>` redraws all five figures from the saved tables.

---

## Reproducing the results

### 1. Environment

Python 3.10+ is recommended.

```bash
git clone https://github.com/Emsh1379/RheumaPain.git
cd RheumaPain
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Versions used for the paper: scikit-learn 1.6.1 (Kaggle: audit, features, E1–E4, mixed models) and
scikit-learn 1.9.1 with PyTorch 2.11 (local: E5–E10, CNN, sensitivity analysis). A GPU is optional (used only by `s05_deep.py`).

### 2. Data

The raw data are **not redistributed** here. Download RheumaPain v3 (CC BY 4.0) from
<https://doi.org/10.17632/y6pgjdj22f.3> and check the files against [`data/manifest.json`](data/manifest.json);
see [`data/README.md`](data/README.md).

### 3. Run

```bash
export RP_DATA=/path/to/folder/containing/RheumaPain   # the folder that holds "RheumaPain/"
export RP_OUT=/path/for/outputs
cd src
python s01_audit.py && python s01b_audit_detail.py && python s01c_batch_check.py
python s02_features.py 10 30 60              # windows of 10, 30 and 60 s
python s03_experiments.py                    # E1–E5, permutation tests, mixed models
python s04_diagnosis.py                      # E6
python s05_deep.py                           # E7 (PyTorch)
python s06_fig_leakage.py && python s08_extra.py   # leakage comparison, E8–E10
RP_SUBSET=batchB python s03_experiments.py   # sensitivity analysis (batch B only)
python s07_compare_batchB.py
python s09_publication_figures.py "$RP_OUT/results" ../figures   # article figures (or ../results: saved tables)
```

**Shortcut without the raw data:** copy `results/features_w*.csv` to `$RP_OUT/results/` and start at
`s03_experiments.py`.

Useful environment variables: `RP_PARTS` (subset of parts of `s03_experiments.py`, e.g. `e2,perm`),
`RP_NPERM` (permutations, default used: 100), `RP_SEEDS` and `RP_EPOCHS` (CNN; paper: 3 seeds, 30 epochs).

### Methodological safeguards

- **Subject-independent evaluation:** leave-one-subject-out for every reported model.
- **No tuning leakage:** all hyperparameters were fixed in advance; CNN channel normalisation is fitted on training folds only.
- **Context baseline:** every model is compared with the phase-only rule by paired subject bootstrap.
- **Phase-preserving permutation null:** pain labels are shuffled within phase, so the test asks whether signals add anything beyond context.
- **Native sampling rates:** BVP 64 Hz, ACC 32 Hz, EDA/TEMP 4 Hz, as recommended in the data article.
- **Pre-specified plan:** [`docs/ANALYSIS_PLAN.md`](docs/ANALYSIS_PLAN.md) was written before the experiments.
- **Independent re-run:** audit, features, E2 and E3 were re-run from the raw data in a clean environment; see [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

---

## Data audit and cleaning

Summary (details in [`docs/data_audit.md`](docs/data_audit.md)); the rules are implemented in `src/common.py`:

1. Drop the first row of processed files whose values equal the E4 header (BVP 64, EDA 4, ACC 32, TEMP 4) — 35 files.
2. Trim 10 s at both ends of the "batch A" files (s001–s005, s011, s038), which show resampling distortion.
3. Exclude s024 and s025 (signals identical to s023); drop label fragments shorter than 0.5 s.

## Reporting and declarations

- Reporting guideline: **TRIPOD+AI** (Collins et al., BMJ 2024;385:e078378) — [`docs/TRIPOD_AI_checklist.md`](docs/TRIPOD_AI_checklist.md).
- The analysis code was written and executed with the assistance of Claude (Anthropic) under the authors' direction;
  all outputs were reviewed by the authors.

## Citation

Please cite the article and the dataset. Software citation metadata: [`CITATION.cff`](CITATION.cff).

```
Savabieh S, Sharifi ME. Wrist-worn biosignals do not generalise for pain estimation in children with rheumatic
disease: a leakage-aware benchmark and audit of a public dataset. Int J Med Inform. Submitted, 2026.
```

Dataset: Yıldırım E, Patlar Akbulut F. RheumaPain: a multimodal biosignal dataset for objective pain
assessment in pediatric rheumatic diseases. *Data in Brief*. 2026;68:113131. doi:10.1016/j.dib.2026.113131.
Data: doi:10.17632/y6pgjdj22f.3.

## License

Code is released under the [MIT License](LICENSE). Result tables and figures derived from RheumaPain
follow the dataset's CC BY 4.0 terms (attribute the dataset authors). The raw dataset is not included.

## Authors and contributions (CRediT)

- **Mohammad Emad Sharifi** — Conceptualization, Data curation, Methodology, Software, Formal analysis, Investigation, Visualization, Writing – original draft.
- **Shekoofeh Savabieh** ([ORCID 0009-0000-4973-1793](https://orcid.org/0009-0000-4973-1793)) — Supervision, Validation, Writing – review & editing.

Ethics: secondary analysis of a public, de-identified dataset whose original collection was approved by the
data-collecting institution's ethics committee; ethical approval and informed consent were waived.
Funding: this research did not receive any specific grant from funding agencies in the public, commercial, or
not-for-profit sectors. Competing interests: none.

## Acknowledgements

The authors thank the creators of the RheumaPain dataset for making their data publicly available.
