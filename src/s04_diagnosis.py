"""Stage 4 - can wrist signals predict the diagnosis? JIA group vs other rheumatic/other diagnoses.

Only this binary task is feasible: after de-duplication there are 32 JIA-group children and 7 others
(SLE 1, JDM 1, JSkl 1, Others 4); s007 has no diagnosis and is excluded. Evaluation is leave-one-subject-out;
subject-level predictions; significance by label permutation across subjects.
Env: RP_NPERM (default 200).
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BATCH_A, DUPLICATES, RES  # noqa: E402

warnings.filterwarnings("ignore")
NPERM = int(os.environ.get("RP_NPERM", 200))
rng = np.random.default_rng(42)

df = pd.read_csv(os.path.join(RES, "features_w30.csv"))
df = df[~df.subject.isin(DUPLICATES) & (df.diag_group != "Unknown")]
SIG = [c for c in df if c.startswith(("eda_", "temp_", "acc_"))] + \
      ["bvp_amp_std", "hr_mean", "hr_std", "ibi_sdnn", "ibi_rmssd", "ibi_pnn50"]

# Subject-level table: mean of each feature in rest and in exercise, plus their difference (reactivity).
m = df.groupby(["subject", "phase"])[SIG].mean().unstack()
X = pd.concat({"rest": m.xs("rest", axis=1, level=1), "ex": m.xs("exercise", axis=1, level=1)}, axis=1)
X.columns = [f"{a}_{b}" for a, b in X.columns]
for f in SIG:
    X[f"delta_{f}"] = X[f"ex_{f}"] - X[f"rest_{f}"]
info = df.groupby("subject").agg(diag=("diag_group", "first"), age=("age", "first"), sex=("sex", "first"))
y = (info.loc[X.index, "diag"] == "JIA").astype(int).to_numpy()
demo = pd.DataFrame({"age": info.loc[X.index, "age"],
                     "sex_f": info.loc[X.index, "sex"].astype(str).str.upper().str.startswith("F").astype(int)})
batch_a = X.index.isin(BATCH_A).astype(int)

MODELS = {
    "LR": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)),
    "RF": make_pipeline(SimpleImputer(strategy="median"),
                        RandomForestClassifier(200, min_samples_leaf=3, class_weight="balanced_subsample",
                                               random_state=0, n_jobs=-1)),
}


def loso(model, Xm, yv):
    p = np.empty(len(yv))
    for i in range(len(yv)):
        tr = np.arange(len(yv)) != i
        p[i] = clone(model).fit(Xm[tr], yv[tr]).predict_proba(Xm[[i]])[0, 1]
    return p


sets = {
    "Demographics (age, sex)": demo.to_numpy(),
    "Signals rest+exercise+delta": X.to_numpy(),
    "Signals + demographics": np.hstack([X.to_numpy(), demo.to_numpy()]),
    "Batch label only (confound check)": batch_a.reshape(-1, 1).astype(float),
}
print(f"Subjects: {len(y)} | JIA group: {y.sum()} | other: {len(y) - y.sum()} | features: {X.shape[1]}")
rows = []
for name, Xm in sets.items():
    for mname, model in MODELS.items():
        if name.startswith("Batch") and mname == "RF":
            continue
        p = loso(model, Xm, y)
        auc = roc_auc_score(y, p)
        n_perm = NPERM if mname == "LR" else min(NPERM, 50)
        null = np.array([roc_auc_score(yp, loso(model, Xm, yp)) for yp in (rng.permutation(y) for _ in range(n_perm))])
        pval = (1 + np.sum(null >= auc)) / (1 + n_perm)
        rows.append({"features": name, "model": mname, "AUC": auc, "BalAcc": balanced_accuracy_score(y, p >= 0.5),
                     "null_mean_AUC": null.mean(), "p_perm": pval, "n_perm": n_perm})
        print(f"  {name:36s} {mname}: AUC={auc:.3f} BalAcc={rows[-1]['BalAcc']:.3f} "
              f"null={null.mean():.3f} p={pval:.3f} ({n_perm} perms)", flush=True)
out = pd.DataFrame(rows)
out.to_csv(os.path.join(RES, "E6_diagnosis.csv"), index=False)
