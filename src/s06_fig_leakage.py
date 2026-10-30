"""Figure 8 / table: leakage effect for the feature RF (E3) and the 1D-CNN (E7) side by side.
Usage: python s06_fig_leakage.py <results_dir> <figures_dir>
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

R, F = sys.argv[1], sys.argv[2]
e3 = pd.read_csv(os.path.join(R, "E3_leakage.csv"))
e7 = pd.read_csv(os.path.join(R, "E7_deep.csv"))
e7 = e7[(e7.run == "ensemble") | ~e7.design.isin(["random5_binary", "loso_binary"])]

rows = []
for task, t3, t7 in (("binary", "binary", "binary"), ("6-class", "6-class", "6class")):
    for split, s3, s7 in (("Random window split", "Random", "random5"), ("Leave-one-child-out", "LOSO", "loso")):
        rf = e3[(e3.task == t3) & e3.split.str.startswith(s3)].F1_macro.iloc[0]
        cnn = e7[e7.design == f"{s7}_{t7}"].win_F1_macro.iloc[0]
        rows += [("Random-forest features", task, split, rf), ("1D-CNN raw signals", task, split, cnn)]
t = pd.DataFrame(rows, columns=["model", "task", "split", "window_F1_macro"])
t.to_csv(os.path.join(R, "E7_leakage_comparison.csv"), index=False)
print(t.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
for ax, task in zip(axes, ["binary", "6-class"]):
    s = t[t.task == task].pivot(index="model", columns="split", values="window_F1_macro")
    s[["Random window split", "Leave-one-child-out"]].plot.bar(ax=ax, rot=0, color=["#55A868", "#C44E52"],
                                                               legend=False)
    ax.set(title=f"{task} pain", xlabel="", ylim=(0, 1), ylabel="Window macro-F1")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height() + 0.01),
                    ha="center", fontsize=7)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=False)
fig.suptitle("Random window splits inflate both feature models and deep learning")
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(os.path.join(F, "fig8_leakage_cnn_vs_rf.png"), dpi=150)
