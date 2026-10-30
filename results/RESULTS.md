# RheumaPain – Results (Kaggle run, 15 Oct 2026)

> Working notes of all experiments (written during the analysis). Paths `kaggle_output/results/` and `rp/` refer to
> the original working folders, now `results/` and `src/`. Figure numbers below (Fig. 4, 8, 9) refer to working
> figures; the article figures are listed in the main README. Article Supplementary Data S1 =
> `Supplementary_Data_S1_results.xlsx`.


Pipeline: `rp/` (audit → features → experiments), run on Kaggle (4 CPU, scikit-learn 1.6.1).
All model evaluation is **leave-one-subject-out (LOSO)** with fixed hyperparameters; pain is scored per
phase segment (window predictions averaged); 95 % CIs are subject-level bootstraps (500 resamples).
Raw tables: `kaggle_output/results/`, figures: `kaggle_output/figures/`.

## 1. Data audit (findings not reported in the data article)

| Issue | Evidence | Handling |
|---|---|---|
| Header row stored as data (35 subjects, "batch B") | first processed row = E4 header values (BVP 64, EDA 4, ACC 32, TEMP 4 → the 4 °C minima) | first row dropped; afterwards processed ≡ raw (r ≈ 1.00) |
| Resampling ringing (7 subjects, "batch A": s001–s005, s011, s038; float-typed labels) | TEMP error 3.7–5.2 °C in the first second, 0.03–0.18 °C in seconds 1–10; negative EDA | 10 s trimmed at both ends |
| Identical processed signals for s023, s024, s025 | identical MD5 of all signal columns; raw files differ | s024, s025 excluded |
| One-row label fragments at file end (s005, s038, 64 Hz) | 0-s trailing segment | segments < 0.5 s dropped |
| Total duration 6.02 h (5.79 h without duplicates) vs "≈ 5.5 h" in the article; rows per pain level differ from Table 2 | `audit_subjects.csv` | reported |
| Pain labels in the processed files vs `patients.xlsx` | agree for 42/42 subjects (incl. multi-valued exercise scores such as "0, 2") | — |

Analysed data: 40 subjects, 235 phase segments (126 rest, 109 exercise), 1,105 windows of 30 s (50 % overlap).

## 2. Descriptive statistics
- Exercise pain > rest pain in 31/40 children, equal in 9, never lower (median 3 vs 1; Wilcoxon p < 0.0001).
- No association of pain with age (ρ ≈ 0), sex or diagnosis group (all p > 0.13).
- Segment labels: rest {0: 46, 1: 26, 2: 40, 3: 11, 4: 3}; exercise {0: 9, 1: 10, 2: 29, 3: 34, 4: 21, 5: 6}.
  Binary target pain ≥ 2: prevalence 0.61.

## 3. E1 – Exercise vs rest from the wrist signals (window level, LOSO)
Best AUC 0.61 (all signals, RF); ACC alone 0.60, EDA alone 0.60, physiology without ACC 0.57, TEMP 0.52,
BVP ≤ 0.49. The two phases are barely separable across children, i.e. the physiotherapy "exercise" phase is
not physiologically distinct from "rest" at the wrist (also: no feature differs between phases at the
subject level after FDR correction, E4).

## 4. E2 – Pain estimation (segment level, LOSO)

| Features | Model | AUC (pain ≥ 2) [95 % CI] | MAE (0–5) [95 % CI] | ρ |
|---|---|---|---|---|
| Majority / mean | – | 0.50 | 1.15 | – |
| **Phase only (exercise → pain)** | rule / Ridge | **0.71 [0.64, 0.79]** | **1.02 [0.84, 1.20]** | 0.51 |
| Demographics + phase | LR / Ridge | 0.75 [0.66, 0.84] | 0.99 [0.81, 1.17] | 0.42 |
| Signals only | LR / Ridge | 0.48 [0.35, 0.60] | 1.27 [1.07, 1.48] | −0.15 |
| Signals + phase | RF | 0.61 [0.50, 0.71] | 1.20 [1.02, 1.38] | 0.31 |
| Signals + demographics + phase | LR / Ridge | 0.70 [0.58, 0.82] | 1.12 [0.92, 1.35] | 0.36 |
| Subject-z-scored signals + phase | RF / Ridge | 0.70 [0.62, 0.77] | 1.00 [0.84, 1.20] | 0.43 |
| Rest-calibrated signals + phase | RF / HGB | 0.71 [0.62, 0.80] | 0.99 [0.83, 1.18] | 0.47 |

Paired subject-bootstrap differences versus phase only:
- Uncalibrated signals + phase are **worse** than phase alone (RF ΔAUC −0.10 [−0.16, −0.04]; ΔMAE +0.18 [+0.10, +0.27]).
- Calibrated signals tie with phase alone: ΔAUC −0.03 to +0.00 (all CIs include 0); ΔMAE −0.03 to +0.03,
  all CIs include 0 except subject-z-scored Ridge (−0.015 [−0.030, −0.002], i.e. 0.015 points on a 0–5 scale).
- Within one phase, no calibrated model ranks pain above chance (AUC 0.46–0.53, ρ −0.13…+0.07); the
  uncalibrated signals + phase RF ranks *inversely* (AUC 0.27 rest, 0.39 exercise).

**Interpretation:** the apparent pain-prediction performance is explained by the exercise/rest label;
wrist signals add no subject-independent information about pain intensity.

## 5. E3 – Leakage
Same RF, same features, window level:

| Split | 6-class macro-F1 | Binary macro-F1 |
|---|---|---|
| Random 5-fold over windows | **0.85** | **0.91** |
| Leave-one-subject-out | 0.24 | 0.59 |

A random window split reproduces the ~92 % F1 reported in the 2024 thesis; it measures subject identification,
not pain recognition.

## 6. E4 – Reactivity (exercise − rest, subject level, n = 40, BH-FDR)
- No feature differs between rest and exercise at the subject level (all q_phase ≥ 0.63).
- In the full sample, the change in wrist movement variability correlates **negatively** with the pain
  increase: ACC magnitude SD ρ = −0.61, z-axis SD ρ = −0.60, x/y SD ρ ≈ −0.59, jerk ρ = −0.57 (all q ≤ 0.001);
  ENMO ρ = −0.48 (q = 0.008). Mean HR, HRV (RMSSD), tonic EDA and temperature show none (q > 0.2).
- **Batch sensitivity — the effect is not robust.** All 7 batch-A children (different file processing, see §1)
  report *identical* rest and exercise pain (Δpain = 0) and have larger movement increases. Restricted to the
  33 batch-B children, the associations shrink (ACC magnitude SD ρ = −0.39, p = 0.026; z-axis SD ρ = −0.39,
  p = 0.025; HR SD ρ = −0.44, p = 0.010) and **none survives FDR correction (min q = 0.23)**.
  The "movement guarding" pattern is therefore at most a hypothesis for a prospective study.

## 7. E5 – Ablations (signals + phase RF)
Dropping any modality changes AUC by ≤ 0.04 (dropping ACC or TEMP slightly *improves* it); 10 s windows
AUC 0.65, 30 s 0.61, 60 s 0.59. No modality carries transferable pain information on its own.

## 8. Permutation tests (100 permutations, pain labels shuffled across segments *within* phase)
The null keeps the exercise/rest–pain relation and destroys only the signal–pain relation, so it tests
whether signals add anything beyond phase.

| Model | Metric | Observed | Null mean | p |
|---|---|---|---|---|
| Signals + phase, RF | AUC | 0.611 | 0.670 | 0.98 |
| Signals + phase, LR | AUC | 0.636 | 0.687 | 0.99 |
| Rest-calibrated signals + phase, RF | AUC | 0.705 | 0.690 | 0.29 |
| Signals + phase, RF | −MAE | −1.195 | −1.057 | 1.00 |
| Rest-calibrated signals + phase, RF | −MAE | −1.010 | −1.026 | 0.20 |

Uncalibrated models score *below* their null: with real labels they learn between-child signal–pain
associations that do not transfer to new children. No model beats the phase-preserving null.

## 9. Linear mixed models: segment pain ~ z(feature) + exercise + (1 | subject), BH-FDR over 26 features
- Full sample: only ACC variability features are associated with lower pain (z-axis SD β = −0.14
  [−0.23, −0.06] per SD, q = 0.016; x/y SD β ≈ −0.12 to −0.13, q = 0.034; dominant frequency β = +0.11, q = 0.039).
  HR, HRV, EDA and temperature: all q ≥ 0.14.
- Adding a batch covariate leaves these estimates unchanged (subject intercepts absorb batch), but
  **in batch-B children only, every effect shrinks to |β| ≤ 0.06 and none is significant (min q = 0.69)**.

## 9b. E7 – Deep-learning baseline (multi-branch 1D-CNN on raw signals; `rp/s05_deep.py`)
Same 1,105 windows as the feature models (verified), each signal at its native rate: BVP 64 Hz, ACC 32 Hz,
EDA + TEMP 4 Hz in three convolutional branches; the phase flag joins the dense head. Fixed recipe, no tuning
and no early stopping (AdamW, lr 1e-3, 30 epochs, balanced class weights); channel normalisation from training
folds only. Main designs averaged over 3 seeds (ensemble); trained locally on an RTX 5050 (PyTorch 2.11).

| Design | Window macro-F1 | Segment AUC [95 % CI] | ΔAUC vs phase only [95 % CI] |
|---|---|---|---|
| Random 5-fold windows, binary | **0.87** | 0.86 [0.79, 0.92] | – |
| Random 5-fold windows, 6-class | 0.60 | – | – |
| Leave-one-child-out, binary (CNN + phase) | 0.57 | 0.53 [0.41, 0.66] | **−0.18 [−0.30, −0.06]** |
| Leave-one-child-out, binary (signals only) | 0.50 | 0.43 [0.31, 0.57] | −0.28 [−0.39, −0.14] |
| Leave-one-child-out, 6-class | 0.19 | – | – |

Seed-to-seed spread (leave-one-child-out, binary): AUC 0.49–0.54. Within-phase AUC 0.41 (rest and exercise).
Side-by-side with the feature RF (Fig. 8, `E7_leakage_comparison.csv`): random split 0.87 vs 0.91 (binary) and
0.60 vs 0.85 (6-class); leave-one-child-out 0.57 vs 0.59 and 0.19 vs 0.24.

**Interpretation:** deep learning on raw signals behaves like the feature models: high scores only when windows
of the same child appear in training and test, chance-level or worse on new children, and significantly worse
than knowing the phase alone.

## 9c. Sensitivity analysis: batch-B children only (n = 33; `kaggle_output/batchB/`)
Batch A (s001–s005, s011, s038) was processed differently and all 7 children report identical rest and
exercise pain. All main analyses were re-run without them (`RP_SUBSET=batchB`, same feature files;
local run, scikit-learn 1.9.1 / PyTorch 2.11). Full table: `kaggle_output/batchB/comparison_full_vs_batchB.csv`.

| Result | Full (n = 40) | Batch B (n = 33) |
|---|---|---|
| Exercise pain > rest pain | 31 / 40 | 31 / 33 (Wilcoxon p = 8.7e-07) |
| Phase only: AUC / MAE | 0.71 / 1.02 | 0.75 / 0.95 |
| Signals + phase RF: AUC, ΔAUC vs phase | 0.61, −0.10 [−0.16, −0.04] | 0.67, −0.08 [−0.13, −0.02] |
| Rest-calibrated + phase RF: AUC, ΔAUC | 0.71, +0.00 [−0.04, +0.05] | 0.71, −0.03 [−0.07, +0.00] |
| Within-phase AUC, calibrated RF (rest / exercise) | 0.51 / 0.52 | 0.47 / 0.24 |
| Permutation p, all 5 models | 0.20–1.00 | 0.61–0.99 |
| RF macro-F1 binary: random split / leave-one-child-out | 0.91 / 0.59 | 0.90 / 0.58 |
| RF macro-F1 6-class: random split / leave-one-child-out | 0.85 / 0.24 | 0.85 / 0.26 |
| CNN macro-F1 binary: random split / leave-one-child-out | 0.87 / 0.57 | 0.85 / 0.60 |
| CNN (+phase) leave-one-child-out: AUC, ΔAUC vs phase | 0.53, −0.18 [−0.30, −0.06] | 0.63, −0.12 [−0.22, +0.00] |
| E1 exercise vs rest, best physiology-only AUC | 0.57 | 0.50 (ACC alone 0.62) |

**Interpretation:** every main conclusion holds, and the null result is, if anything, clearer without batch A:
phase alone is the strongest predictor (no signal model beats it; several are significantly worse), all
permutation tests are non-significant, within-phase ranking is at or below chance, and both random-split
inflations persist. Only the movement–pain reactivity association (E4) depended on batch A (§6, §9).

## 9d. Extra analyses for the manuscript (`rp/s08_extra.py`, local, Kaggle features)
- **E8 learning curve** (`E8_learning_curve.csv`, Fig. 9): rest-calibrated RF AUC 0.59 (5 training children)
  → 0.70 (15) → 0.70–0.72 (20–39), i.e. it plateaus at the phase-only level (0.71); raw-signal LR plateaus
  at 0.61–0.64. More children do not add pain information beyond phase.
- **E9 subject identification** (`E9_subject_identification.csv`, `E9_between_child_variance.csv`): RF identifies
  the child from one 30 s window with 89.7 % accuracy (balanced 85.7 %; chance 2.5 %). Median share of feature
  variance between children 0.32; > 0.5 for 38 % of features (TEMP mean 0.98, tonic EDA 0.68, HR mean 0.54).
- **E10 minimum detectable effect** (`E10_minimum_detectable_effect.csv`): bootstrap SE of ΔAUC vs phase only
  0.022 → minimum detectable gain ≈ 0.06 AUC (80 % power, two-sided α 0.05).

## 10. Overall conclusions
1. The processed RheumaPain files contain two processing artefacts and one triplicated subject; corrected
   loading is required (header-row drop, edge trimming, de-duplication).
2. Exercise vs rest phase — not the wrist signals — carries the information about reported pain
   (phase alone: AUC 0.71, MAE 1.02).
3. Subject-independent models using EDA, BVP/HR/HRV, temperature and accelerometry do not improve on phase
   alone (paired bootstrap and permutation tests), and rank pain at chance within a phase.
4. Random window splits inflate performance to the ~90 % range reported in earlier unpublished work
   (macro-F1 0.91 binary / 0.85 six-class vs 0.59 / 0.24 subject-independent). The same holds for a
   1D-CNN on raw signals (0.87 vs 0.57 binary), which on new children is worse than phase alone.
5. An apparent movement-restriction signal is driven by the differently processed batch-A subset and is
   not robust within batch B.

## Provenance
- Kaggle (scikit-learn 1.6.1): audit, features, descriptive statistics, E1–E4, mixed models, the three binary
  permutation tests and E2 tables.
- Local re-run on the **Kaggle-produced feature files** (scikit-learn 1.9.1), after the Kaggle session timed out:
  E5, the two regression permutation tests, the E2 replication used for Fig. 4, and all figures.
  The local runs reproduce Kaggle within 0.003 AUC (E5) and 0.006 AUC / 0.013 MAE across all 40 E2 rows
  (`E2_pain_local_replication.csv`).
