"""Publication figures of the manuscript, redrawn from the saved result tables.

Usage: python s09_publication_figures.py <results_dir> <out_dir>
  results_dir  folder with features_w30.csv, E2_pain.csv, E7_deep.csv, E7_leakage_comparison.csv, E8_learning_curve.csv
Writes Figure1-Figure4 (main text) and FigureA1 (Appendix A) as 600 dpi TIFF (LZW, RGB), PNG and vector PDF.
"""
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import stats  # noqa: E402

R, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
BATCH_A = ["s001", "s002", "s003", "s004", "s005", "s011", "s038"]
BLUE, ORANGE, GREEN, RED, GREY = "#3A70A0", "#D9822B", "#4F9B5D", "#B5453D", "#222222"
DPI = 600
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor="white")
    plt.close(fig)
    img = Image.open(buf).convert("RGB")  # no alpha channel, as Elsevier asks
    img.save(os.path.join(OUT, f"{name}.tif"), dpi=(DPI, DPI), compression="tiff_lzw")
    img.save(os.path.join(OUT, f"{name}.png"), dpi=(DPI, DPI), optimize=True)


feat = pd.read_csv(os.path.join(R, "features_w30.csv"))
feat = feat[~feat.subject.isin(["s024", "s025"])]
seg = feat.drop_duplicates(["subject", "seg"])

# Figure 1 - pain labels per phase segment
ct = seg.groupby(["pain", "phase"]).size().unstack(fill_value=0).reindex(range(6), fill_value=0)
n_rest, n_ex = int(ct["rest"].sum()), int(ct["exercise"].sum())
fig, ax = plt.subplots(figsize=(4.2, 2.6))
x = np.arange(6)
ax.bar(x - 0.19, ct["rest"], 0.38, color=BLUE, label=f"Rest (n = {n_rest})")
ax.bar(x + 0.19, ct["exercise"], 0.38, color=ORANGE, label=f"Exercise (n = {n_ex})")
ax.set_xticks(x)
ax.set(xlabel="Wong\u2013Baker FACES score (0\u20135)", ylabel="Phase segments")
ax.legend(frameon=False)
save(fig, "Figure1")

# Figure 2 - AUC by feature set (feature models from E2, CNN from E7), horizontal
NAMES = {"Phase only": "Phase only", "Demo+Phase": "Demographics + phase", "S": "Signals",
         "S+P": "Signals + phase", "S+D": "Signals + demographics + phase",
         "Sz+P": "Child-z-scored signals + phase", "Scal+P": "Rest-calibrated signals + phase"}
e2 = pd.read_csv(os.path.join(R, "E2_pain.csv"))
b = e2[(e2.task == "binary") & (e2.features != "Majority/Mean")]
rows = [(f"{NAMES[r.features]} / {r.model}", r.AUC, r.AUC_lo, r.AUC_hi,
         ORANGE if r.features == "Phase only" else BLUE) for r in b.itertuples()]
cnn = pd.read_csv(os.path.join(R, "E7_deep.csv")).query("design == 'loso_binary' and run == 'ensemble'").iloc[0]
rows.append(("1D-CNN raw signals + phase", cnn.seg_AUC, cnn.seg_AUC_lo, cnn.seg_AUC_hi, RED))
ref = float(b.loc[b.features == "Phase only", "AUC"].iloc[0])
fig, ax = plt.subplots(figsize=(5.2, 4.6))
for i, (lab, auc, lo, hi, c) in enumerate(rows):
    ax.errorbar(auc, i, xerr=[[auc - lo], [hi - auc]], fmt="o", color=c, ms=4.5, capsize=2.5, lw=1.2)
ax.axvline(ref, ls=":", c=ORANGE, lw=1.4, label=f"Phase-only rule ({ref:.2f})")
ax.axvline(0.5, ls="--", c="grey", lw=0.9, label="Chance (0.50)")
ax.set_yticks(range(len(rows)), [r[0] for r in rows])
ax.invert_yaxis()
ax.set(xlabel="Segment AUC, pain \u2265 2 (95% CI)", xlim=(0.25, 0.9))
ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
save(fig, "Figure2")

# Figure 3 - leakage, RF vs CNN
t = pd.read_csv(os.path.join(R, "E7_leakage_comparison.csv"))
fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6), sharey=True)
for ax, task, title in zip(axes, ["binary", "6-class"], ["Binary pain (\u2265 2 vs < 2)", "Six-class pain (0\u20135)"]):
    s = t[t.task == task].pivot(index="model", columns="split", values="window_F1_macro")
    s = s.loc[["Random-forest features", "1D-CNN raw signals"], ["Random window split", "Leave-one-child-out"]]
    s.plot.bar(ax=ax, rot=0, color=[GREEN, RED], legend=False, width=0.72)
    ax.set_title(title, fontsize=8.5)
    ax.set(xlabel="", ylim=(0, 1.05))
    ax.set_xticklabels(["Random forest\n(26 features)", "1D-CNN\n(raw signals)"])
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height() + 0.01),
                    ha="center", fontsize=7.5)
axes[0].set_ylabel("Window-level macro-F1")
axes[1].legend(frameon=False, fontsize=7.5, loc="upper right")
fig.tight_layout()
save(fig, "Figure3")

# Figure 4 - learning curve
lc = pd.read_csv(os.path.join(R, "E8_learning_curve.csv"))
fig, ax = plt.subplots(figsize=(4.4, 3.0))
ax.axhline(ref, ls=":", c=ORANGE, lw=1.4, label=f"Phase-only rule ({ref:.2f})")
ax.axhline(0.5, ls="--", c="grey", lw=0.9)
for name, c, mk in [("Rest-calibrated signals + phase (RF)", BLUE, "o"), ("Signals + phase (LR)", RED, "s")]:
    s = lc[lc.model == name].groupby("k_train_children").AUC
    ax.errorbar(s.mean().index, s.mean(), yerr=[s.mean() - s.min(), s.max() - s.mean()], fmt=f"{mk}-", ms=4,
                capsize=3, color=c, label=name)
ax.set(xlabel="Children in the training set", ylabel="Segment AUC (held-out child)", ylim=(0.35, 0.85))
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
save(fig, "Figure4")

# Figure A.1 - rest-to-exercise reactivity (movement and heart rate) vs pain change, batch A marked
sub = feat.groupby(["subject", "phase"])[["acc_mag_std", "hr_mean", "pain"]].mean().unstack().dropna()
dpain = sub[("pain", "exercise")] - sub[("pain", "rest")]
isA = sub.index.isin(BATCH_A)
jit = np.random.default_rng(3).uniform(-0.1, 0.1, len(dpain))
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
for ax, f, lab in zip(axes, ["acc_mag_std", "hr_mean"],
                      ["Change in wrist movement SD (g)", "Change in mean heart rate (beats/min)"]):
    dx = sub[(f, "exercise")] - sub[(f, "rest")]
    rho, p = stats.spearmanr(dx, dpain)
    rb, pb = stats.spearmanr(dx[~isA], dpain[~isA])
    ax.scatter(dx[~isA], (dpain + jit)[~isA], s=18, c=BLUE, label="Batch B")
    ax.scatter(dx[isA], (dpain + jit)[isA], s=24, c=RED, marker="^", label="Batch A")
    ax.axvline(0, c="grey", lw=0.6)
    ax.set_xlabel(f"{lab}\n(exercise \u2212 rest)")
    ax.set_title(f"All: \u03c1 = {rho:.2f} (p = {p:.1g}); batch B: \u03c1 = {rb:.2f} (p = {pb:.2g})", fontsize=7.5)
axes[0].set_ylabel("Change in pain (exercise \u2212 rest)")
axes[0].legend(frameon=False, fontsize=7.5)
fig.tight_layout()
save(fig, "FigureA1")
print("written:", sorted(os.listdir(OUT)))
