"""Stage 5 - deep-learning baseline: multi-branch 1D-CNN on raw signal windows (E7).

Same cleaned 30 s windows (50 % overlap, within phase segments) as the feature models; each signal at its
native rate: BVP 64 Hz (1 ch), ACC 32 Hz (3 ch, in g), EDA + TEMP 4 Hz (2 ch). Windows shorter than 30 s
(segments of 10-30 s) are edge-padded. Channels are z-scored with training-fold statistics only.
Fixed training recipe (no tuning, no early stopping -> no validation leakage).

Designs (mirroring E2/E3):
  random5_*   random 5-fold split over windows (leaky; same child in train and test)
  loso_*      leave-one-subject-out
Tasks: binary (pain >= 2) and 6-class (pain 0-5). Variants: CNN+phase (phase flag joins the dense head)
and CNN signals-only.
Env: RP_SEEDS (default 3), RP_EPOCHS (default 30), RP_DESIGNS (comma list, default all), RP_DEVICE.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BATCH_A, DUPLICATES, RES, SUBJECTS, load_processed, segments  # noqa: E402

warnings.filterwarnings("ignore")
WIN, STEP = 30, 15
SEEDS = int(os.environ.get("RP_SEEDS", 3))
EPOCHS = int(os.environ.get("RP_EPOCHS", 30))
DEV = os.environ.get("RP_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
PAIN_THR = 2
ALL_DESIGNS = ["random5_binary", "random5_6class", "loso_binary", "loso_binary_nophase", "loso_6class"]
DESIGNS = [d for d in os.environ.get("RP_DESIGNS", ",".join(ALL_DESIGNS)).split(",") if d]
RATES = {"bvp": 64, "acc": 32, "slow": 4}


# ----------------------------------------------------------------------------- data
def build_windows():
    """Raw windows, same segmentation/window rule as s02_features.py."""
    cache = os.path.join(RES, f"deep_windows_w{WIN}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in ("bvp", "acc", "slow")}, pd.DataFrame(z["meta"].tolist(), columns=z["meta_cols"])
    B, A, S, meta = [], [], [], []
    for sid in SUBJECTS:
        d4, d32, d64 = (load_processed(sid, hz) for hz in (4, 32, 64))
        s4, s32, s64 = segments(d4, 4), segments(d32, 32), segments(d64, 64)
        for k in range(len(s4)):
            r4, r32, r64 = s4.iloc[k], s32.iloc[k], s64.iloc[k]
            dur = (r4.stop - r4.start) / 4
            starts = np.arange(0, max(dur - WIN, 0) + 1e-9, STEP) if dur >= WIN else ([0.0] if dur >= 10 else [])
            for t0 in starts:
                t1 = min(t0 + WIN, dur)
                def cut(df_, r, hz, cols):
                    x = df_[cols].to_numpy(dtype=np.float32)[int(r.start + t0 * hz):int(r.start + t1 * hz)]
                    return np.pad(x, ((0, WIN * hz - len(x)), (0, 0)), mode="edge").T
                B.append(cut(d64, r64, 64, ["bvp"]))
                A.append(cut(d32, r32, 32, ["x", "y", "z"]) / 64.0)
                S.append(cut(d4, r4, 4, ["eda", "temperature"]))
                meta.append((sid, k, r4.phase, int(r4.pain), t0))
    arrays = {"bvp": np.stack(B), "acc": np.stack(A), "slow": np.stack(S)}
    cols = ["subject", "seg", "phase", "pain", "t0"]
    np.savez_compressed(cache, **arrays, meta=np.array(meta, dtype=object), meta_cols=np.array(cols))
    return arrays, pd.DataFrame(meta, columns=cols)


# ----------------------------------------------------------------------------- model
def branch(c_in, ch, k, pool):
    layers, c = [], c_in
    for c_out in ch:
        layers += [nn.Conv1d(c, c_out, k, padding=k // 2), nn.BatchNorm1d(c_out), nn.ReLU(), nn.MaxPool1d(pool)]
        c = c_out
    return nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten())


class PainCNN(nn.Module):
    def __init__(self, n_out, use_phase):
        super().__init__()
        self.use_phase = use_phase
        self.bvp = branch(1, (16, 32, 32), 7, 4)    # 1920 -> 30
        self.acc = branch(3, (16, 32, 32), 7, 4)    # 960 -> 15
        self.slow = branch(2, (16, 32), 5, 2)       # 120 -> 30
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(96 + use_phase, 32), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(32, n_out))

    def forward(self, b, a, s, ph):
        h = [self.bvp(b), self.acc(a), self.slow(s)] + ([ph] if self.use_phase else [])
        return self.head(torch.cat(h, dim=1))


def fit_predict(X, y, phase, tr, te, n_cls, use_phase, seed):
    """Train on tr, return class probabilities for te. Normalisation uses training windows only."""
    torch.manual_seed(seed); np.random.seed(seed)
    stats_ = {k: (X[k][tr].mean(axis=(0, 2), keepdims=True), X[k][tr].std(axis=(0, 2), keepdims=True) + 1e-6)
              for k in X}
    T = lambda k, idx: torch.tensor((X[k][idx] - stats_[k][0]) / stats_[k][1], device=DEV)  # noqa: E731
    xb_tr, xa_tr, xs_tr = T("bvp", tr), T("acc", tr), T("slow", tr)
    ph_tr = torch.tensor(phase[tr, None], dtype=torch.float32, device=DEV)
    y_tr = torch.tensor(y[tr], device=DEV)
    counts = np.bincount(y[tr], minlength=n_cls).astype(np.float32)
    w = np.where(counts > 0, counts.sum() / (n_cls * np.maximum(counts, 1)), 0.0)  # balanced class weights
    model = PainCNN(n_cls, use_phase).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEV))
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(EPOCHS):
        model.train()
        for idx in torch.randperm(len(tr), generator=g).split(64):
            idx = idx.to(DEV)
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = loss_fn(model(xb_tr[idx], xa_tr[idx], xs_tr[idx], ph_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        out = model(T("bvp", te), T("acc", te), T("slow", te),
                    torch.tensor(phase[te, None], dtype=torch.float32, device=DEV))
        return torch.softmax(out, dim=1).cpu().numpy()


# ----------------------------------------------------------------------------- evaluation
def seg_table(meta, p_pos):
    m = meta.assign(pred=p_pos)
    g = m.groupby("segid")
    return pd.DataFrame({"subject": g.subject.first(), "pain": g.pain.first(), "is_exercise": g.is_exercise.first(),
                         "y_bin": (g.pain.first() >= PAIN_THR).astype(int), "pred": g.pred.mean()})


def boot(seg, fn, r, n=500):
    subs = seg.subject.unique()
    by = {s: seg[seg.subject == s] for s in subs}
    return np.nanpercentile([fn(pd.concat([by[s] for s in r.choice(subs, len(subs))])) for _ in range(n)], [2.5, 97.5])


def auc_of(b):
    return roc_auc_score(b.y_bin, b.pred) if b.y_bin.nunique() > 1 else np.nan


def main():
    t_start = time.time()
    X, meta = build_windows()
    drop = list(DUPLICATES) + (list(BATCH_A) if os.environ.get("RP_SUBSET") == "batchB" else [])
    keep = ~meta.subject.isin(drop).to_numpy()
    X = {k: v[keep] for k, v in X.items()}
    meta = meta[keep].reset_index(drop=True)
    meta["segid"] = meta.subject + "_" + meta.seg.astype(str)
    meta["is_exercise"] = (meta.phase == "exercise").astype(int)
    feat = pd.read_csv(os.path.join(RES, "features_w30.csv"))
    feat = feat[~feat.subject.isin(drop)]
    assert len(feat) == len(meta) and (feat.subject.to_numpy() == meta.subject.to_numpy()).all() \
        and np.allclose(feat.t0.to_numpy(), meta.t0.to_numpy()), "windows differ from the feature windows"
    print(f"device={DEV} | windows={len(meta)} | subjects={meta.subject.nunique()} | segments={meta.segid.nunique()} "
          f"| seeds={SEEDS} | epochs={EPOCHS}", flush=True)

    y6 = meta.pain.to_numpy().astype(np.int64)
    yb = (y6 >= PAIN_THR).astype(np.int64)
    phase = meta.is_exercise.to_numpy().astype(np.float32)
    subj = meta.subject.to_numpy()
    subjects = np.unique(subj)
    r = np.random.default_rng(7)
    rows, seg_rows = [], []

    for design in DESIGNS:
        t0 = time.time()
        task = "6class" if design.endswith("6class") else "binary"
        y, n_cls = (y6, 6) if task == "6class" else (yb, 2)
        use_phase = not design.endswith("nophase")
        if design.startswith("random5"):
            folds = list(KFold(5, shuffle=True, random_state=0).split(y))
        else:
            folds = [(np.flatnonzero(subj != s), np.flatnonzero(subj == s)) for s in subjects]
        n_seeds = SEEDS if design in ("loso_binary", "random5_binary") else 1
        probs = np.zeros((n_seeds, len(y), n_cls), dtype=np.float32)
        for si in range(n_seeds):
            for tr, te in folds:
                probs[si, te] = fit_predict(X, y, phase, tr, te, n_cls, use_phase, seed=si)
        for si in range(n_seeds + (1 if n_seeds > 1 else 0)):  # each seed, plus the seed ensemble
            pr = probs[si] if si < n_seeds else probs.mean(axis=0)
            label = f"seed{si}" if si < n_seeds else "ensemble"
            pred = pr.argmax(1)
            row = {"design": design, "task": task, "phase_input": use_phase, "run": label,
                   "win_accuracy": accuracy_score(y, pred), "win_F1_macro": f1_score(y, pred, average="macro")}
            if task == "binary":
                seg = seg_table(meta, pr[:, 1])
                row.update({"seg_AUC": auc_of(seg), "seg_BalAcc": balanced_accuracy_score(seg.y_bin, seg.pred >= 0.5)})
                for ph, sub in seg.groupby("is_exercise"):
                    row[f"seg_AUC_{'exercise' if ph else 'rest'}"] = auc_of(sub)
                if label == "ensemble" or n_seeds == 1:
                    row["seg_AUC_lo"], row["seg_AUC_hi"] = boot(seg, auc_of, r)
                    if design.startswith("loso"):  # paired difference vs the phase-only rule on the same segments
                        ref = seg.assign(pred=seg.is_exercise.astype(float))
                        subs = seg.subject.unique()
                        by = {s: (seg[seg.subject == s], ref[ref.subject == s]) for s in subs}
                        d = []
                        for _ in range(500):
                            pick = r.choice(subs, len(subs))
                            a = pd.concat([by[s][0] for s in pick]); b = pd.concat([by[s][1] for s in pick])
                            d.append(auc_of(a) - auc_of(b))
                        row["dAUC_vs_phase"] = auc_of(seg) - auc_of(ref)
                        row["dAUC_lo"], row["dAUC_hi"] = np.nanpercentile(d, [2.5, 97.5])
                    seg_rows.append(seg.reset_index().assign(design=design))
            rows.append(row)
        out = pd.DataFrame(rows)
        out.to_csv(os.path.join(RES, "E7_deep.csv"), index=False)
        if seg_rows:
            pd.concat(seg_rows).to_csv(os.path.join(RES, "E7_deep_segment_predictions.csv"), index=False)
        show = out[out.design == design].drop(columns=["design"]).round(3)
        print(f"\n=== {design} ({time.time() - t0:.0f}s)\n{show.to_string(index=False)}", flush=True)
    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min -> {os.path.join(RES, 'E7_deep.csv')}")


if __name__ == "__main__":
    main()
