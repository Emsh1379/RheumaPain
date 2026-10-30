# RheumaPain – Analysis Plan

Working title: *Subject-independent pain estimation from wrist-worn physiological signals in pediatric
rheumatic disease: a leakage-aware benchmark on RheumaPain.*

## Constraints taken from the data article (Yıldırım & Patlar Akbulut, Data in Brief 68:113131, 2026)
- 42 patients, Empatica E4; native rates BVP 64 Hz, ACC 32 Hz, EDA/TEMP 4 Hz.
- Pain = Wong-Baker FACES 0–5, one self-report per phase (rest / exercise); phases timed by stopwatch.
- Resampled files are "convenience representations"; BVP < 64 Hz must not be used for beat/IBI/HRV.
  → Each signal is taken at its native rate: BVP from the 64 Hz file, ACC from the 32 Hz file,
    EDA/TEMP from the 4 Hz file (all carry the phase and pain columns).
- Segments were selected around periods of stronger/weaker pain → possible selection bias (discussed, not fixable).
- Labels are per phase, not per moment → the effective sample size is ~42 subjects × 2–5 phase segments.

## 0. Data audit (reported as a paper section)
- Per-subject duration, number/length of rest and exercise segments, rows per pain level vs Table 2.
- Consistency of per-row pain labels with `patients.xlsx` (rest/exercise pain; multi-valued entries like "0, 2").
- Duplicates: s023/s024/s025 processed signals are identical → keep s023 only (sensitivity analysis with all).
- Physiological plausibility: negative EDA, TEMP < 20 °C, flat-line/contact-loss segments; processed vs raw EDA.
- Cross-rate consistency (4 Hz × 16 ≈ 64 Hz rows).

## 1. Descriptive statistics
- Pain distribution per phase; Wilcoxon signed-rank rest vs exercise pain (paired by subject).
- Pain vs age (Spearman), sex (Mann-Whitney), diagnosis group (JIA vs other; Kruskal-Wallis).

## 2. Feature extraction (windowed, within a single phase segment)
- Windows: 30 s, 50 % overlap (ablation: 10 s, 60 s); windows never cross a segment boundary.
- EDA (4 Hz): mean, std, min, max, slope, tonic level, phasic peak count/amplitude (NeuroKit2 if available).
- TEMP (4 Hz): mean, std, slope.
- ACC (32 Hz): magnitude mean/std, ENMO, per-axis std, dominant frequency, jerk.
- BVP (64 Hz): pulse rate and HRV (mean HR, SDNN, RMSSD, pNN50) via peak detection, with a quality flag.
- Per-subject normalisation variant: features z-scored by the subject's rest segment (reactivity).

## 3. Experiments (all subject-independent: Leave-One-Subject-Out, tuning nested inside training folds)
- **E1 Exercise vs rest** (sanity check): window-level AUC/F1; modality ablation (ACC-only vs physiology-only).
- **E2 Pain estimation**:
  - Tasks: binary (pain 0–1 vs ≥ 2, threshold fixed from the label distribution) and regression (MAE, Spearman ρ).
  - Prediction unit: subject-phase segment (window predictions averaged).
  - Models: logistic / ridge, random forest, gradient boosting (simple models first).
  - Baselines: majority / mean, demographics + phase only (age, sex, diagnosis group, exercise flag).
  - Feature sets: signals only, signals + demographics.
- **E3 Leakage demonstration**: same model with random window split vs LOSO (explains inflated prior results).
- **E4 Reactivity**: per-subject Δfeature (exercise − rest) vs Δpain and exercise pain; Spearman with FDR correction.
- **E5 Ablations**: modality dropping, window length, per-subject normalisation.
- **Statistics**: subject-bootstrap 95 % CIs; label-permutation test (permuting at subject level) for the main LOSO model;
  linear mixed model pain ~ feature + phase + (1 | subject).

## 4. Outputs
- `results/*.csv` (audit, descriptive, metrics), `figures/*.png`, and a short results summary.
