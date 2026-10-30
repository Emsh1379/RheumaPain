"""Stage 3 - descriptive statistics and subject-independent experiments (E1-E5).

All model evaluation is Leave-One-Subject-Out with fixed hyperparameters (no tuning, so no nested CV
is needed). Pain predictions are scored per phase segment (window predictions averaged), because
each segment carries a single self-reported score.

Env:
  RP_PARTS   comma list of parts to run (default all): desc,e1,e2,e3,e4,e5,perm,mixed,figs
             Every part writes its own CSV right away; 'figs' reads those CSVs, so parts can be re-run
             separately (e.g. after a lost session).
  RP_NPERM   permutations (default 100)   RP_TREES  RF size (default 300)   RP_NBOOT  bootstraps (default 500)
  RP_WIN     window length in s (default 30)   RP_KEEP_DUPS=1 keeps s024/s025
  RP_SUBSET  'batchB' restricts every analysis to the 33 batch-B children (sensitivity analysis)
"""
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,  # noqa: E402
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,  # noqa: E402
                             mean_absolute_error, roc_auc_score)
from sklearn.model_selection import KFold, LeaveOneGroupOut  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BATCH_A, DUPLICATES, FIG, RES, load_demographics  # noqa: E402

warnings.filterwarnings("ignore")
ALL_PARTS = ["desc", "e1", "e2", "e3", "e4", "e5", "perm", "mixed", "figs"]
PARTS = [p.strip() for p in os.environ.get("RP_PARTS", ",".join(ALL_PARTS)).split(",") if p.strip()]
WIN = int(os.environ.get("RP_WIN", 30))
NPERM = int(os.environ.get("RP_NPERM", 100))
TREES = int(os.environ.get("RP_TREES", 300))  # RF size; lower only for quick tests
NBOOT = int(os.environ.get("RP_NBOOT", 500))
PAIN_THR = 2  # binary task: pain >= 2 ("hurts a little more" or worse) vs 0-1
LOG = []


def rng(part):
    """Independent, fixed random stream per part, so re-running one part reproduces its numbers."""
    return np.random.default_rng(ALL_PARTS.index(part) + 1)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def res(name):
    return os.path.join(RES, name)


# ----------------------------------------------------------------------------- data
def load_features(win):
    d = pd.read_csv(res(f"features_w{win}.csv"))
    if os.environ.get("RP_KEEP_DUPS") != "1":
        d = d[~d.subject.isin(DUPLICATES)]
    if os.environ.get("RP_SUBSET") == "batchB":  # sensitivity analysis: drop the differently processed batch A
        d = d[~d.subject.isin(BATCH_A)]
    d["sex_f"] = (d.sex.astype(str).str.upper().str.startswith("F")).astype(int)
    d["diag_jia"] = (d.diag_group == "JIA").astype(int)
    d["segid"] = d.subject + "_" + d.seg.astype(str)
    d["y_bin"] = (d.pain >= PAIN_THR).astype(int)
    return d.reset_index(drop=True)


df = load_features(WIN)
MOD = {
    "EDA": [c for c in df if c.startswith("eda_")],
    "TEMP": [c for c in df if c.startswith("temp_")],
    "ACC": [c for c in df if c.startswith("acc_")],
    "BVP": ["bvp_amp_std", "hr_mean", "hr_std", "ibi_sdnn", "ibi_rmssd", "ibi_pnn50"],
}
SIG = sum(MOD.values(), [])
DEMO = ["age", "sex_f", "diag_jia"]
PHASE = ["is_exercise"]

# Within-subject calibrated signals: z-score each feature by the subject's own rest windows.
cal = df[SIG].copy()
for sid, idx in df.groupby("subject").groups.items():
    ref = df.loc[idx]
    ref = ref[ref.is_exercise == 0] if (ref.is_exercise == 0).sum() >= 2 else ref
    cal.loc[idx] = (df.loc[idx, SIG] - ref[SIG].mean()) / ref[SIG].std().replace(0, np.nan)
CAL = [f"cal_{c}" for c in SIG]
df[CAL] = cal.to_numpy()
# Unsupervised per-subject z-scoring over all of the subject's windows (needs no phase information).
SZ = [f"sz_{c}" for c in SIG]
df[SZ] = df.groupby("subject")[SIG].transform(lambda c: (c - c.mean()) / (c.std() if c.std() > 0 else np.nan)).to_numpy()

FEATSETS = {"S": SIG, "S+P": SIG + PHASE, "S+D": SIG + DEMO + PHASE, "Sz+P": SZ + PHASE, "Scal+P": CAL + PHASE}
YB, YR, G = df.y_bin.to_numpy(), df.pain.to_numpy().astype(float), df.subject.to_numpy()


def clf_models(n_trees=None, n_jobs=-1):
    return {
        "LR": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)),
        "RF": make_pipeline(SimpleImputer(strategy="median"),
                            RandomForestClassifier(n_trees or TREES, min_samples_leaf=5, class_weight="balanced_subsample",
                                                   n_jobs=n_jobs, random_state=0)),
        "HGB": make_pipeline(SimpleImputer(strategy="median"),
                             HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                            class_weight="balanced", random_state=0)),
    }


def reg_models(n_trees=None, n_jobs=-1):
    return {
        "Ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10)),
        "RF": make_pipeline(SimpleImputer(strategy="median"),
                            RandomForestRegressor(n_trees or TREES, min_samples_leaf=5, n_jobs=n_jobs, random_state=0)),
        "HGB": make_pipeline(SimpleImputer(strategy="median"),
                             HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                                                           random_state=0)),
    }


# ----------------------------------------------------------------------------- evaluation helpers
def oof_predict(model, X, y, groups, kind, splitter=None):
    """Out-of-fold predictions; LOSO by default."""
    pred = np.full(len(y), np.nan)
    for tr, te in (splitter or LeaveOneGroupOut()).split(X, y, groups):
        if model is None or (kind == "clf" and len(np.unique(y[tr])) < 2):  # majority / mean baseline
            pred[te] = y[tr].mean()
            continue
        m = clone(model).fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1] if kind == "clf" else m.predict(X[te])
    return pred


def seg_level(d, pred):
    g = d.assign(pred=pred).groupby("segid")
    return pd.DataFrame({"subject": g.subject.first(), "pain": g.pain.first(), "y_bin": g.y_bin.first(),
                         "is_exercise": g.is_exercise.first(), "pred": g.pred.mean()})


def clf_metrics(y, p):
    return {"AUC": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
            "BalAcc": balanced_accuracy_score(y, p >= 0.5), "F1_macro": f1_score(y, p >= 0.5, average="macro")}


def reg_metrics(y, p):
    return {"MAE": mean_absolute_error(y, p),
            "Spearman": float(stats.spearmanr(y, p)[0]) if np.std(p) > 0 else np.nan}


def auc_of(b):
    return roc_auc_score(b.y_bin, b.pred) if b.y_bin.nunique() > 1 else np.nan


def mae_of(b):
    return mean_absolute_error(b.pain, b.pred)


def boot_ci(seg, fn, r, n):
    subs = seg.subject.unique()
    by = {s: seg[seg.subject == s] for s in subs}
    vals = [fn(pd.concat([by[s] for s in r.choice(subs, len(subs))])) for _ in range(n)]
    return np.nanpercentile(vals, [2.5, 97.5])


def paired_delta(a, b, fn, r, n):
    """Subject-bootstrap of metric(a) - metric(b) on the same segments."""
    m = a[["subject", "pain", "y_bin"]].assign(pa=a.pred, pb=b.pred.reindex(a.index))
    subs = m.subject.unique()
    by = {s: m[m.subject == s] for s in subs}
    obs = fn(m.assign(pred=m.pa)) - fn(m.assign(pred=m.pb))
    vals = []
    for _ in range(n):
        bb = pd.concat([by[s] for s in r.choice(subs, len(subs))])
        vals.append(fn(bb.assign(pred=bb.pa)) - fn(bb.assign(pred=bb.pb)))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return obs, lo, hi


def subject_phase_means():
    sub_means = df.groupby(["subject", "phase"])[SIG + ["pain"]].mean().unstack()
    return sub_means.dropna(subset=[("pain", "rest"), ("pain", "exercise")])


SEG_TAB = df.groupby("segid").agg(subject=("subject", "first"), phase=("phase", "first"),
                                  pain=("pain", "first"), n_win=("pain", "size")).reset_index()


# ============================================================================= parts
def part_desc():
    demo = load_demographics()
    subj_phase = SEG_TAB.groupby(["subject", "phase"]).pain.mean().unstack()
    paired = subj_phase.dropna()
    log(f"\n=== Data used: window {WIN}s | {df.subject.nunique()} subjects | {len(SEG_TAB)} segments | {len(df)} windows")
    for ph in ("rest", "exercise"):
        log(f"Segment-level pain counts ({ph}):", SEG_TAB[SEG_TAB.phase == ph].pain.value_counts().sort_index().to_dict())
    log(f"Binary label (pain >= {PAIN_THR}) segment prevalence: {(SEG_TAB.pain >= PAIN_THR).mean():.2f}")
    w = stats.wilcoxon(paired["exercise"], paired["rest"])
    log(f"Pain exercise vs rest (subject means, n={len(paired)}): median {paired.exercise.median():.1f} vs "
        f"{paired.rest.median():.1f}; higher in exercise for {(paired.exercise > paired.rest).sum()}, equal for "
        f"{(paired.exercise == paired.rest).sum()}; Wilcoxon p={w.pvalue:.2g}")
    sp = subj_phase.join(demo.set_index("subject")[["age", "sex", "diag_group"]])
    rows = []
    for ph in ("rest", "exercise"):
        r = stats.spearmanr(sp.age, sp[ph], nan_policy="omit")
        mw = stats.mannwhitneyu(sp.loc[sp.sex == "F", ph].dropna(), sp.loc[sp.sex == "M", ph].dropna())
        kw = stats.kruskal(*[g[ph].dropna() for _, g in sp.groupby("diag_group") if g[ph].notna().sum() > 1])
        rows.append({"phase": ph, "rho_age": r[0], "p_age": r[1], "p_sex": mw.pvalue, "p_diag": kw.pvalue})
        log(f"  {ph}: pain~age Spearman rho={r[0]:.2f} p={r[1]:.3f} | sex MWU p={mw.pvalue:.3f} | "
            f"diagnosis group KW p={kw.pvalue:.3f}")
    pd.DataFrame(rows).assign(wilcoxon_ex_vs_rest_p=w.pvalue, n_paired=len(paired)).to_csv(res("descriptive.csv"), index=False)


def part_e1():
    log("\n=== E1 exercise vs rest (window level, LOSO)")
    rows = []
    y1 = df.is_exercise.to_numpy()
    for fs_name, cols in {**MOD, "Physio(EDA+TEMP+BVP)": MOD["EDA"] + MOD["TEMP"] + MOD["BVP"], "All": SIG,
                          "All (subject z-scored)": SZ}.items():
        for mname, m in clf_models().items():
            p = oof_predict(m, df[cols].to_numpy(), y1, G, "clf")
            rows.append({"features": fs_name, "model": mname, **clf_metrics(y1, p)})
    e1 = pd.DataFrame(rows)
    e1.to_csv(res("E1_phase.csv"), index=False)
    log(e1.round(3).to_string(index=False))


def part_e2():
    r = rng("e2")
    log(f"\n=== E2 pain estimation (LOSO, scored per segment) | binary = pain >= {PAIN_THR}")
    rows, store = [], {}
    for task, y, kind, models in (("binary", YB, "clf", clf_models), ("regression", YR, "reg", reg_models)):
        sets = {"Majority/Mean": (None, {"-": None}),
                "Phase only": (PHASE, {k: v for k, v in models().items() if k in ("LR", "Ridge")}),
                "Demo+Phase": (DEMO + PHASE, models()), **{k: (v, models()) for k, v in FEATSETS.items()}}
        for fs_name, (cols, ms) in sets.items():
            for mname, m in ms.items():
                X = df[cols].to_numpy() if cols else np.zeros((len(df), 1))
                p = oof_predict(m, X, y, G, kind)
                if fs_name == "Phase only" and kind == "clf":
                    # Phase-only is the rule "exercise -> pain >= 2". Scoring the fold-wise refitted LR instead
                    # would let tiny fold-to-fold probability shifts break the ties and distort the pooled AUC.
                    p = df.is_exercise.to_numpy().astype(float)
                seg = seg_level(df, p)
                store[(task, fs_name, mname)] = seg
                met = clf_metrics(seg.y_bin, seg.pred) if kind == "clf" else reg_metrics(seg.pain, seg.pred)
                if fs_name == "Phase only" and kind == "reg":  # ranking of the same rule; MAE uses fold means
                    met["Spearman"] = float(stats.spearmanr(seg.pain, seg.is_exercise)[0])
                if m is None:  # a constant predictor has no ranking ability; pooled LOSO AUC/rho of the
                    met.update({"AUC": 0.5} if kind == "clf" else {"Spearman": np.nan})  # fold means is an artefact
                rows.append({"task": task, "features": fs_name, "model": mname, **met})
    e2 = pd.DataFrame(rows)
    for task, key, fn in (("binary", "AUC", auc_of), ("regression", "MAE", mae_of)):
        for i in e2.index[(e2.task == task) & (e2.features != "Majority/Mean")]:
            row = e2.loc[i]
            e2.loc[i, [f"{key}_lo", f"{key}_hi"]] = boot_ci(store[(task, row.features, row.model)], fn, r, NBOOT)
    e2.to_csv(res("E2_pain.csv"), index=False)
    pd.concat([s.assign(task=t, features=f, model=m) for (t, f, m), s in store.items()]).rename_axis("segid") \
        .reset_index().to_csv(res("E2_segment_predictions.csv"), index=False)
    log(e2.round(3).to_string(index=False))

    log("\n  Paired subject-bootstrap differences vs 'Phase only' (positive dAUC / negative dMAE = better than phase):")
    rows = []
    for task, ref, fn, key in (("binary", ("binary", "Phase only", "LR"), auc_of, "dAUC"),
                               ("regression", ("regression", "Phase only", "Ridge"), mae_of, "dMAE")):
        for (t, fs_name, mname), seg in store.items():
            if t != task or fs_name in ("Majority/Mean", "Phase only"):
                continue
            obs, lo, hi = paired_delta(seg, store[ref], fn, r, NBOOT)
            rows.append({"task": task, "features": fs_name, "model": mname, "metric": key,
                         "delta": obs, "lo": lo, "hi": hi})
            log(f"    {task:10s} {fs_name:10s} {mname:5s} {key}={obs:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    pd.DataFrame(rows).to_csv(res("E2_vs_phase_only.csv"), index=False)

    log("\n  Within-phase performance (segments of one phase only):")
    rows = []
    for key in [("S+P", "RF"), ("Sz+P", "RF"), ("Scal+P", "RF"), ("Scal+P", "HGB"), ("Demo+Phase", "LR")]:
        for task in ("binary", "regression"):
            mname = {"LR": "Ridge"}.get(key[1], key[1]) if task == "regression" else key[1]
            for ph, sub in store[(task, key[0], mname)].groupby("is_exercise"):
                met = clf_metrics(sub.y_bin, sub.pred) if task == "binary" else reg_metrics(sub.pain, sub.pred)
                rows.append({"features": key[0], "model": mname, "task": task,
                             "phase": "exercise" if ph else "rest", "n": len(sub), **met})
    wp = pd.DataFrame(rows)
    wp.to_csv(res("E2_within_phase.csv"), index=False)
    log(wp.round(3).to_string(index=False))

    y3 = (df.pain >= 3).astype(int).to_numpy()
    s3 = seg_level(df.assign(y_bin=y3), oof_predict(clf_models()["RF"], df[SIG + PHASE].to_numpy(), y3, G, "clf"))
    m3 = clf_metrics(s3.y_bin, s3.pred)
    pd.DataFrame([{"threshold": 3, "features": "S+P", "model": "RF", **m3}]).to_csv(res("E2_sensitivity_thr3.csv"), index=False)
    log("  Sensitivity pain>=3, S+P RF:", {k: round(v, 3) for k, v in m3.items()})


def part_e3():
    log("\n=== E3 leakage: random window split vs LOSO (RF, window-level, signals+phase)")
    rows = []
    X, y6 = df[SIG + PHASE].to_numpy(), df.pain.to_numpy()
    for split_name, splitter in (("Random 5-fold (windows)", KFold(5, shuffle=True, random_state=0)),
                                 ("LOSO (subjects)", LeaveOneGroupOut())):
        for task, y in (("6-class", y6), ("binary", YB)):
            pred = np.empty(len(y))
            for tr, te in splitter.split(X, y, G):
                m = make_pipeline(SimpleImputer(strategy="median"),
                                  RandomForestClassifier(TREES, n_jobs=-1, random_state=0)).fit(X[tr], y[tr])
                pred[te] = m.predict(X[te])
            rows.append({"split": split_name, "task": task, "accuracy": accuracy_score(y, pred),
                         "F1_macro": f1_score(y, pred, average="macro")})
    e3 = pd.DataFrame(rows)
    e3.to_csv(res("E3_leakage.csv"), index=False)
    log(e3.round(3).to_string(index=False))


def part_e4():
    log("\n=== E4 reactivity: exercise - rest feature change vs pain (subject level, BH-FDR)")
    both = subject_phase_means()
    in_b = ~both.index.isin(BATCH_A)  # sensitivity: batch-A files were processed (and possibly labelled) differently
    rows = []
    for f in SIG:
        rest, ex = both[(f, "rest")], both[(f, "exercise")]
        ok = rest.notna() & ex.notna()
        delta = (ex - rest)[ok]
        dpain = (both[("pain", "exercise")] - both[("pain", "rest")])[ok]
        wt = stats.wilcoxon(ex[ok], rest[ok]) if ok.sum() > 5 else None
        r1, r2 = stats.spearmanr(delta, dpain), stats.spearmanr(delta, both[("pain", "exercise")][ok])
        rb = stats.spearmanr(delta[in_b[ok]], dpain[in_b[ok]])
        rows.append({"feature": f, "n": int(ok.sum()), "median_rest": rest[ok].median(), "median_ex": ex[ok].median(),
                     "p_phase": wt.pvalue if wt else np.nan, "rho_dfeat_dpain": r1[0], "p_dpain": r1[1],
                     "rho_dfeat_expain": r2[0], "p_expain": r2[1],
                     "n_batchB": int(in_b[ok].sum()), "rho_dpain_batchB": rb[0], "p_dpain_batchB": rb[1]})
    e4 = pd.DataFrame(rows)
    log("Batch A subjects (all with identical rest/exercise pain): delta pain = " +
        str((both[("pain", "exercise")] - both[("pain", "rest")])[~in_b].tolist()))
    for c in ("p_phase", "p_dpain", "p_expain", "p_dpain_batchB"):
        ok = e4[c].notna()
        e4.loc[ok, c.replace("p_", "q_")] = multipletests(e4.loc[ok, c], method="fdr_bh")[1]
    e4.to_csv(res("E4_reactivity.csv"), index=False)
    log(f"subjects with both phases: {len(both)}")
    log(e4.sort_values("p_dpain").round(3).to_string(index=False))


def part_e5():
    log("\n=== E5 ablations (RF, signals+phase, LOSO, segment level)")
    rows = []
    for drop in [None] + list(MOD):
        cols = [c for c in SIG if drop is None or c not in MOD[drop]] + PHASE
        sb = seg_level(df, oof_predict(clf_models()["RF"], df[cols].to_numpy(), YB, G, "clf"))
        sr = seg_level(df, oof_predict(reg_models()["RF"], df[cols].to_numpy(), YR, G, "reg"))
        rows.append({"ablation": "none" if drop is None else f"-{drop}", **clf_metrics(sb.y_bin, sb.pred),
                     **reg_metrics(sr.pain, sr.pred)})
    for w in (10, 60):
        if os.path.exists(res(f"features_w{w}.csv")):
            dw = load_features(w)
            Xw, gw = dw[SIG + PHASE].to_numpy(), dw.subject.to_numpy()
            sb = seg_level(dw, oof_predict(clf_models()["RF"], Xw, dw.y_bin.to_numpy(), gw, "clf"))
            sr = seg_level(dw, oof_predict(reg_models()["RF"], Xw, dw.pain.to_numpy().astype(float), gw, "reg"))
            rows.append({"ablation": f"window {w}s", **clf_metrics(sb.y_bin, sb.pred), **reg_metrics(sr.pain, sr.pred)})
    e5 = pd.DataFrame(rows)
    e5.to_csv(res("E5_ablation.csv"), index=False)
    log(e5.round(3).to_string(index=False))


def _perm_score(model, X, kind, pain_map):
    pain = df.segid.map(pain_map).to_numpy()
    y = (pain >= PAIN_THR).astype(int) if kind == "clf" else pain.astype(float)
    seg = seg_level(df.assign(pain=pain, y_bin=(pain >= PAIN_THR).astype(int)), oof_predict(model, X, y, G, kind))
    return auc_of(seg) if kind == "clf" else -mae_of(seg)


def part_perm():
    r = rng("perm")
    log(f"\n=== Permutation tests ({NPERM} permutations; pain labels shuffled across segments within phase)")
    segs = df.drop_duplicates("segid")[["segid", "phase", "pain"]].reset_index(drop=True)
    phase_idx = [idx for _, idx in segs.groupby("phase").groups.items()]
    perms = []  # the same permuted label sets are used for every model
    for _ in range(NPERM):
        pain = segs.pain.to_numpy().copy()
        for idx in phase_idx:
            pain[idx] = r.permutation(pain[idx])
        perms.append(dict(zip(segs.segid, pain)))
    n_tr = min(100, TREES)
    specs = (("binary S+P RF", SIG + PHASE, "clf", clf_models(n_tr, n_jobs=1)["RF"]),
             ("binary S+P LR", SIG + PHASE, "clf", clf_models()["LR"]),
             ("binary Scal+P RF", CAL + PHASE, "clf", clf_models(n_tr, n_jobs=1)["RF"]),
             ("regression S+P RF", SIG + PHASE, "reg", reg_models(n_tr, n_jobs=1)["RF"]),
             ("regression Scal+P RF", CAL + PHASE, "reg", reg_models(n_tr, n_jobs=1)["RF"]))
    only = [s.strip() for s in os.environ.get("RP_PERM_MODELS", "").split(",") if s.strip()]
    rows = []
    if only and os.path.exists(res("permutation_tests.csv")):  # resume: keep results of models not re-run
        rows = [r for r in pd.read_csv(res("permutation_tests.csv")).to_dict("records") if r["model"] not in only]
    for name, cols, kind, model in specs:
        if only and name not in only:
            continue
        X = df[cols].to_numpy()
        obs = _perm_score(model, X, kind, dict(zip(segs.segid, segs.pain)))
        null = np.array(Parallel(n_jobs=-1)(delayed(_perm_score)(model, X, kind, pm) for pm in perms))
        p = (1 + np.sum(null >= obs)) / (1 + NPERM)
        rows.append({"model": name, "metric": "AUC" if kind == "clf" else "-MAE", "observed": obs,
                     "null_mean": null.mean(), "null_p95": np.percentile(null, 95), "p_perm": p, "n_perm": NPERM})
        log(f"  {name}: observed {rows[-1]['metric']}={obs:.3f}, null mean={null.mean():.3f}, p={p:.3f}")
        pd.DataFrame(rows).to_csv(res("permutation_tests.csv"), index=False)  # saved after every model


def part_mixed():
    import statsmodels.formula.api as smf
    log("\n=== Linear mixed models: segment pain ~ z(feature) + exercise + (1|subject)")
    segm = df.groupby("segid").agg({**{c: "mean" for c in SIG}, "pain": "first", "is_exercise": "first",
                                    "subject": "first"}).reset_index()
    segm["batch_a"] = segm.subject.isin(BATCH_A).astype(int)
    rows = []
    for f in SIG:
        d = segm[[f, "pain", "is_exercise", "subject", "batch_a"]].dropna()
        d["z"] = (d[f] - d[f].mean()) / d[f].std()
        row = {"feature": f, "n_seg": len(d)}
        # main model; sensitivity 1: + batch covariate; sensitivity 2: batch B subjects only
        for tag, formula, data in (("", "pain ~ z + is_exercise", d),
                                   ("_adj_batch", "pain ~ z + is_exercise + batch_a", d),
                                   ("_batchB", "pain ~ z + is_exercise", d[d.batch_a == 0])):
            try:
                fit = smf.mixedlm(formula, data, groups=data.subject).fit(reml=True)
                ci = fit.conf_int().loc["z"]
                row.update({f"beta{tag}": fit.params["z"], f"ci_lo{tag}": ci[0], f"ci_hi{tag}": ci[1],
                            f"p{tag}": fit.pvalues["z"], f"converged{tag}": fit.converged})
            except Exception as ex:  # convergence failures are reported, not hidden
                row.update({f"beta{tag}": np.nan, f"p{tag}": np.nan, f"error{tag}": str(ex)[:60]})
        rows.append(row)
    mm = pd.DataFrame(rows)
    for tag in ("", "_adj_batch", "_batchB"):
        ok = mm[f"p{tag}"].notna()
        mm.loc[ok, f"q{tag}"] = multipletests(mm.loc[ok, f"p{tag}"], method="fdr_bh")[1]
    mm.to_csv(res("mixed_models.csv"), index=False)
    log(mm.sort_values("p").round(4).to_string(index=False))


def part_figs():
    r = rng("figs")
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9})
    save = lambda fig, name: (fig.tight_layout(), fig.savefig(os.path.join(FIG, name)), plt.close(fig))  # noqa: E731

    fig, ax = plt.subplots(figsize=(5, 3))
    ct = SEG_TAB.groupby(["pain", "phase"]).size().unstack(fill_value=0).reindex(range(6), fill_value=0)
    ct[["rest", "exercise"]].plot.bar(ax=ax, rot=0, color=["#4C72B0", "#DD8452"])
    ax.set(xlabel="Wong-Baker FACES score", ylabel="Segments", title="Pain labels per phase segment")
    save(fig, "fig1_pain_distribution.png")

    if os.path.exists(res("E3_leakage.csv")):
        e3 = pd.read_csv(res("E3_leakage.csv"))
        fig, ax = plt.subplots(figsize=(5, 3))
        e3.pivot(index="task", columns="split", values="F1_macro").plot.bar(ax=ax, rot=0, color=["#C44E52", "#55A868"])
        ax.set(ylabel="Macro-F1 (windows)", title="Leave-one-subject-out vs random window split", ylim=(0, 1))
        save(fig, "fig2_leakage.png")

    if os.path.exists(res("E2_pain.csv")):
        b = pd.read_csv(res("E2_pain.csv")).query("task == 'binary' and features != 'Majority/Mean'").reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.errorbar(range(len(b)), b.AUC, yerr=[b.AUC - b.AUC_lo, b.AUC_hi - b.AUC], fmt="o", capsize=3)
        ref = b.loc[b.features == "Phase only", "AUC"].iat[0]
        ax.axhline(0.5, ls="--", c="grey", lw=0.8)
        ax.axhline(ref, ls=":", c="#DD8452", lw=1, label="phase only")
        ax.set_xticks(range(len(b)), b.features + " / " + b.model, rotation=70, ha="right")
        ax.set(ylabel="Segment AUC (95% CI)", title=f"Pain >= {PAIN_THR} vs < {PAIN_THR}, leave-one-subject-out")
        ax.legend(loc="lower right", fontsize=7)
        save(fig, "fig3_pain_auc.png")

    if os.path.exists(res("E2_segment_predictions.csv")):
        sp = pd.read_csv(res("E2_segment_predictions.csv"))
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4), sharey=True)
        for ax, (f, m) in zip(axes, [("S+P", "RF"), ("Scal+P", "HGB")]):
            seg = sp.query("task == 'regression' and features == @f and model == @m")
            ax.scatter(seg.pain + r.uniform(-0.15, 0.15, len(seg)), seg.pred, c=seg.is_exercise, cmap="coolwarm", s=16)
            ax.plot([0, 5], [0, 5], "k--", lw=0.8)
            ax.set(xlabel="Self-reported pain", title=f"{f} {m} (red = exercise)")
        axes[0].set_ylabel("Predicted pain (LOSO)")
        save(fig, "fig4_pred_vs_true.png")

    both = subject_phase_means()
    key = ["hr_mean", "ibi_rmssd", "eda_tonic_mean", "eda_scr_rate", "temp_slope", "acc_mag_std"]
    fig, axes = plt.subplots(1, len(key), figsize=(12, 2.8))
    for ax, f in zip(axes, key):
        ax.boxplot([both[(f, "rest")].dropna(), both[(f, "exercise")].dropna()])
        ax.set_xticks([1, 2], ["rest", "exercise"]); ax.set_title(f, fontsize=8)
    fig.suptitle("Subject-level feature means by phase")
    save(fig, "fig5_phase_features.png")

    rf = clf_models()["RF"].fit(df[SIG + PHASE].to_numpy(), YB)
    imp = pd.Series(rf[-1].feature_importances_, index=SIG + PHASE).sort_values()[-15:]
    fig, ax = plt.subplots(figsize=(4, 4))
    imp.plot.barh(ax=ax); ax.set(title="RF importance (pain >= 2, all data)", xlabel="Mean decrease in impurity")
    save(fig, "fig6_importance.png")

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    dpain = both[("pain", "exercise")] - both[("pain", "rest")]
    is_a = both.index.isin(BATCH_A)
    jit = r.uniform(-0.1, 0.1, len(dpain))
    for ax, f in zip(axes, ["acc_mag_std", "hr_mean"]):
        dx = both[(f, "exercise")] - both[(f, "rest")]
        rho, p = stats.spearmanr(dx, dpain, nan_policy="omit")
        rho_b, p_b = stats.spearmanr(dx[~is_a], dpain[~is_a], nan_policy="omit")
        ax.scatter(dx[~is_a], (dpain + jit)[~is_a], s=18, c="#4C72B0", label="batch B")
        ax.scatter(dx[is_a], (dpain + jit)[is_a], s=22, c="#C44E52", marker="^", label="batch A")
        ax.axvline(0, c="grey", lw=0.6)
        ax.set(xlabel=f"delta {f} (exercise - rest)", ylabel="delta pain (exercise - rest)",
               title=f"all rho={rho:.2f} (p={p:.1g}); batch B rho={rho_b:.2f} (p={p_b:.1g})")
        ax.title.set_fontsize(8)
    axes[0].legend(fontsize=7)
    fig.suptitle("Within-child reactivity vs pain increase (n = %d)" % len(dpain))
    save(fig, "fig7_reactivity.png")
    log("\nFigures written to", FIG)


if __name__ == "__main__":
    for part in PARTS:
        globals()[f"part_{part}"]()
    with open(res(f"summary_log_{'-'.join(PARTS) if PARTS != ALL_PARTS else 'all'}.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG))
    log("\nDone:", ",".join(PARTS), "| results in", RES)
