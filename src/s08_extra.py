"""Stage 8 - extra analyses for the manuscript.

E8  Learning curve: leave-one-subject-out AUC when the model is trained on k randomly drawn children
    (k = 5 ... 39, 10 repeats). Does performance rise with more children?
E9  Subject identification: how well do window features identify the child (random window split)?
    Plus the share of each feature's variance that lies between children (eta squared).
E10 Minimum detectable effect: the smallest AUC gain over phase-only that this sample could detect
    (80 % power, alpha 0.05), from the subject-bootstrap SE of the paired difference.
Env: RP_OUT (results dir with features_w30.csv and E2_segment_predictions.csv), RP_REPS (default 10).
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
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s03_experiments as X  # noqa: E402  (module level only loads data and helpers)

warnings.filterwarnings("ignore")
REPS = int(os.environ.get("RP_REPS", 10))
df, G = X.df, X.G
subjects = np.unique(G)
res = X.res


# ----------------------------------------------------------------------------- E8 learning curve
def lc_run(k, rep, cols, model):
    r = np.random.default_rng(1000 * k + rep)
    Xm, y = df[cols].to_numpy(), X.YB
    pred = np.full(len(y), np.nan)
    for s in subjects:
        te = np.flatnonzero(G == s)
        pool = subjects[subjects != s]
        train_subj = r.choice(pool, size=min(k, len(pool)), replace=False)
        tr = np.flatnonzero(np.isin(G, train_subj))
        if len(np.unique(y[tr])) < 2:
            pred[te] = y[tr].mean()
            continue
        pred[te] = clone(model).fit(Xm[tr], y[tr]).predict_proba(Xm[te])[:, 1]
    return X.auc_of(X.seg_level(df, pred))


def part_e8():
    print("\n=== E8 learning curve (leave-one-subject-out, k training children, binary pain >= 2)", flush=True)
    specs = {"Rest-calibrated signals + phase (RF)": (X.CAL + X.PHASE, X.clf_models(100, n_jobs=1)["RF"]),
             "Signals + phase (LR)": (X.SIG + X.PHASE, X.clf_models()["LR"])}
    ks = [5, 10, 15, 20, 25, 30, 35, len(subjects) - 1]
    rows = []
    for name, (cols, model) in specs.items():
        jobs = [(k, rep) for k in ks for rep in range(REPS if k < len(subjects) - 1 else 1)]
        aucs = Parallel(n_jobs=12, backend="threading")(delayed(lc_run)(k, rep, cols, model) for k, rep in jobs)
        for (k, rep), a in zip(jobs, aucs):
            rows.append({"model": name, "k_train_children": k, "rep": rep, "AUC": a})
    lc = pd.DataFrame(rows)
    lc.to_csv(res("E8_learning_curve.csv"), index=False)
    summ = lc.groupby(["model", "k_train_children"]).AUC.agg(["mean", "min", "max", "count"]).round(3)
    print(summ.to_string(), flush=True)
    phase_auc = X.auc_of(X.seg_level(df, df.is_exercise.to_numpy().astype(float)))
    # slope of AUC on log2(k) per model, as a simple trend summary
    for name, g in lc.groupby("model"):
        b = np.polyfit(np.log2(g.k_train_children), g.AUC, 1)[0]
        print(f"  {name}: AUC change per doubling of training children = {b:+.3f}", flush=True)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for name, g in lc.groupby("model"):
        s = g.groupby("k_train_children").AUC
        ax.errorbar(s.mean().index, s.mean(), yerr=[s.mean() - s.min(), s.max() - s.mean()], fmt="o-",
                    capsize=3, label=name)
    ax.axhline(phase_auc, ls=":", c="#222222", lw=1.2, label=f"phase only ({phase_auc:.2f})")
    ax.axhline(0.5, ls="--", c="grey", lw=0.8)
    ax.set(xlabel="Children in the training set", ylabel="Segment AUC (held-out child)",
           title="Learning curve, leave-one-subject-out", ylim=(0.3, 0.9))
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(X.FIG, "fig9_learning_curve.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- E9 subject identification
def part_e9():
    print("\n=== E9 subject identification from window features (signals only, random 5-fold)", flush=True)
    Xm = df[X.SIG].to_numpy()
    y = pd.factorize(df.subject)[0]
    pred = np.empty(len(y), dtype=int)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xm, y):
        m = X.make_pipeline(X.SimpleImputer(strategy="median"),
                            RandomForestClassifier(300, n_jobs=-1, random_state=0)).fit(Xm[tr], y[tr])
        pred[te] = m.predict(Xm[te])
    acc, bacc = accuracy_score(y, pred), balanced_accuracy_score(y, pred)
    n_cls = len(np.unique(y))
    # pain within a single child is constant inside a phase, so a model that recognises the child
    # (and the phase) recovers the label without learning anything about pain.
    rows = []
    for f in X.SIG:
        d = df[[f, "subject"]].dropna()
        grand = d[f].mean()
        ss_tot = ((d[f] - grand) ** 2).sum()
        ss_between = d.groupby("subject")[f].apply(lambda v: len(v) * (v.mean() - grand) ** 2).sum()
        rows.append({"feature": f, "eta2_between_children": ss_between / ss_tot if ss_tot > 0 else np.nan})
    eta = pd.DataFrame(rows).sort_values("eta2_between_children", ascending=False)
    eta.to_csv(res("E9_between_child_variance.csv"), index=False)
    out = pd.DataFrame([{"n_children": n_cls, "chance_accuracy": 1 / n_cls, "accuracy": acc, "balanced_accuracy": bacc,
                         "median_eta2_between_children": eta.eta2_between_children.median(),
                         "share_features_eta2_over_0.5": (eta.eta2_between_children > 0.5).mean()}])
    out.to_csv(res("E9_subject_identification.csv"), index=False)
    print(out.round(3).to_string(index=False))
    print(eta.head(8).round(3).to_string(index=False), flush=True)


# ----------------------------------------------------------------------------- E10 minimum detectable effect
def part_e10():
    print("\n=== E10 minimum detectable AUC gain over phase only (80 % power, two-sided alpha 0.05)", flush=True)
    sp = pd.read_csv(res("E2_segment_predictions.csv"))
    r = np.random.default_rng(10)
    rows = []
    for f, m in (("Scal+P", "RF"), ("Sz+P", "RF"), ("S+P", "RF")):
        a = sp.query("task == 'binary' and features == @f and model == @m").set_index("segid")
        b = a.assign(pred=a.is_exercise.astype(float))
        subs = a.subject.unique()
        by = {s: (a[a.subject == s], b[b.subject == s]) for s in subs}
        d = []
        for _ in range(2000):
            pick = r.choice(subs, len(subs))
            aa = pd.concat([by[s][0] for s in pick]); bb = pd.concat([by[s][1] for s in pick])
            d.append(X.auc_of(aa) - X.auc_of(bb))
        se = np.nanstd(d)
        rows.append({"features": f, "model": m, "observed_dAUC": X.auc_of(a) - X.auc_of(b), "bootstrap_SE": se,
                     "MDE_dAUC_80power": (1.96 + 0.84) * se})
    mde = pd.DataFrame(rows)
    mde.to_csv(res("E10_minimum_detectable_effect.csv"), index=False)
    print(mde.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    for p in os.environ.get("RP_EXTRA", "e10,e9,e8").split(","):
        globals()[f"part_{p.strip()}"]()
    print("\nEXTRA DONE")
