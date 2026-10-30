# Independent re-run (reproducibility check)

To check that the repository reproduces the paper, the pipeline was re-run from scratch in a clean Linux
container (4 CPU, no GPU; Python 3, numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1) on 30 Oct 2026.

## Data
All 547 files of RheumaPain v3 were downloaded from Mendeley Data with `data/download_data.py`;
every file matched its SHA-256 checksum in `data/manifest.json`.

## Results

| Step | Script | Compared with | Agreement |
|---|---|---|---|
| Data audit | `s01_audit.py`, `s01b_audit_detail.py`, `s01c_batch_check.py` | `results/audit_*.csv` | Identical (max. difference ≈ 1e-12, floating-point rounding) |
| Feature extraction, 30 s windows | `s02_features.py 30` | `results/features_w30.csv` | Identical: 1,131 windows (1,105 after excluding s024/s025), max. difference ≈ 1e-12 |
| Descriptive statistics | `s03_experiments.py` (`desc`) | `results/descriptive.csv` | Identical |
| E2 point estimates (AUC, MAE, balanced accuracy, F1, ρ) | `s03_experiments.py` (`e2`) | `results/E2_pain_local_replication.csv` (scikit-learn 1.9.1) | Identical |
| E2 point estimates | same | `results/E2_pain.csv` (Kaggle, scikit-learn 1.6.1) | Within 0.006 AUC and 0.013 MAE, as stated in the paper |
| E2 bootstrap 95 % CIs | same | `results/E2_pain.csv`, `E2_vs_phase_only.csv` | CI bounds differ by ≤ 0.02 for the key comparisons (≤ 0.05 for any bound) — Monte-Carlo variation of 500 bootstrap resamples |
| E3 leakage (random split vs LOSO) | `s03_experiments.py` (`e3`) | `results/E3_leakage.csv` | Identical |
| Publication figures | `s09_publication_figures.py` | `figures/` | All 5 article figures (Figs. 1–4, A.1) redrawn from the saved tables (also checked by GitHub Actions on every push) |

Key E2 comparisons, original vs re-run (AUC for pain ≥ 2, LOSO):

| Model | Original | Re-run |
|---|---|---|
| Phase only | 0.708 [0.638, 0.787] | 0.708 [0.646, 0.788] |
| Raw signals + phase, RF: ΔAUC vs phase | −0.098 [−0.155, −0.042] | −0.099 [−0.157, −0.040] |
| Rest-calibrated signals + phase, RF: ΔAUC vs phase | +0.003 [−0.036, +0.045] | +0.002 [−0.044, +0.045] |

Note: the CI of one secondary comparison (raw signals + phase, **logistic regression**, ΔAUC −0.072) is borderline:
upper bound −0.003 in the original run and +0.004 in the re-run. Conclusions should rest on the random-forest
comparison above, which is stable.

E3 leakage (`s03_experiments.py`, `e3`) reproduced `results/E3_leakage.csv` **exactly** (window-level RF):

| Split | Binary macro-F1 | 6-class macro-F1 |
|---|---|---|
| Random 5-fold over windows | 0.908 | 0.848 |
| Leave-one-subject-out | 0.593 | 0.241 |

Not re-run here: E1, E4–E10, permutation tests, mixed models and the 1D-CNN (E7; PyTorch/GPU), which take several
hours on CPU. Their saved outputs are in `results/`.
