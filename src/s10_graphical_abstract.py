"""Graphical abstract (13 x 5 cm, 600 dpi). Values are typed in from results/E2_pain.csv, E7_deep.csv and
E7_leakage_comparison.csv. Usage: python s10_graphical_abstract.py <output_path_without_extension>"""
import io, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
OUT=sys.argv[1]
BLUE, ORANGE, GREEN, RED, GREY, INK = "#3A70A0", "#D9822B", "#4F9B5D", "#B5453D", "#888888", "#1F2A36"
plt.rcParams.update({"font.size": 5.6, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
W, H = 13/2.54, 5/2.54
fig = plt.figure(figsize=(W, H))
fig.patch.set_facecolor("white")
bg = fig.add_axes([0, 0, 1, 1]); bg.axis("off"); bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.set_zorder(-10)
X = [0.008, 0.258, 0.508, 0.758]; PW = 0.234

def panel(x, title, color):
    bg.add_patch(FancyBboxPatch((x, 0.04), PW, 0.90, boxstyle="round,pad=0,rounding_size=0.015",
                                fc="#F6F8FA", ec="#C9D1D9", lw=0.6))
    bg.add_patch(FancyBboxPatch((x, 0.84), PW, 0.10, boxstyle="round,pad=0,rounding_size=0.015",
                                fc=color, ec=color, lw=0.6))
    bg.text(x + PW/2, 0.89, title, ha="center", va="center", color="white", fontsize=5.9, fontweight="bold")

for i in range(3):
    bg.add_patch(FancyArrowPatch((X[i]+PW+0.001, 0.49), (X[i+1]-0.001, 0.49),
                                 arrowstyle="-|>", mutation_scale=6, color=INK, lw=0.8))

# 1 data + audit
panel(X[0], "Public dataset + audit", INK)
t = bg.text
t(X[0]+0.012, 0.79, "RheumaPain dataset", fontsize=5.3, fontweight="bold", color=INK, va="top")
t(X[0]+0.012, 0.715, "• 42 children with\n   rheumatic disease\n• Empatica E4 wristband:\n   EDA, BVP, ACC, TEMP\n• Pain (FACES 0–5) per\n   rest / exercise phase",
  fontsize=4.9, color=INK, va="top", linespacing=1.25)
t(X[0]+0.012, 0.335, "Audit found", fontsize=5.3, fontweight="bold", color=RED, va="top")
t(X[0]+0.012, 0.275, "• header row as data (35 files)\n• resampling distortion (7)\n• one child triplicated\n→ 40 children analysed",
  fontsize=4.9, color=INK, va="top", linespacing=1.25)

# 2 pain estimation in new children
panel(X[1], "Pain in unseen children", BLUE)
ax = fig.add_axes([X[1]+0.09, 0.30, PW-0.105, 0.49]); ax.patch.set_alpha(0)
labs = ["Phase only\n(exercise → pain)", "Calibrated\nsignals + phase", "Raw signals\n+ phase", "1D-CNN\n+ phase"]
vals = [0.71, 0.71, 0.61, 0.53]; cols = [ORANGE, BLUE, BLUE, RED]
ax.barh(range(4), vals, color=cols, height=0.62)
for i, v in enumerate(vals): ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=5.3, color=INK)
ax.axvline(0.5, ls="--", c=GREY, lw=0.6)
ax.set_yticks(range(4), labs, fontsize=4.9); ax.invert_yaxis()
ax.set_xlim(0.4, 0.82); ax.set_xticks([0.5, 0.6, 0.7, 0.8]); ax.tick_params(labelsize=4.8, length=2, width=0.5)
ax.set_xlabel("AUC, pain ≥ 2", fontsize=4.9, labelpad=1.5)
for s in ax.spines.values(): s.set_linewidth(0.5)
t(X[1]+PW/2, 0.10, "No model beat the phase-only rule", ha="center", fontsize=4.8, fontweight="bold", color=BLUE)

# 3 evaluation design
panel(X[2], "Effect of the data split", GREEN)
ax = fig.add_axes([X[2]+0.045, 0.28, PW-0.06, 0.42]); ax.patch.set_alpha(0)
import numpy as np
x = np.arange(2); w = 0.36
r = [0.91, 0.87]; l = [0.59, 0.57]
ax.bar(x - w/2, r, w, color=GREEN, label="Random window split")
ax.bar(x + w/2, l, w, color=RED, label="Leave-one-child-out")
for xi, v in zip(list(x - w/2) + list(x + w/2), r + l):
    ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=5.0, color=INK)
ax.set_xticks(x, ["Random forest", "1D-CNN"], fontsize=5.0); ax.set_ylim(0, 1.08)
ax.set_yticks([0, 0.5, 1.0]); ax.tick_params(labelsize=4.8, length=2, width=0.5)
ax.set_ylabel("Macro-F1 (binary)", fontsize=4.9, labelpad=1)
ax.legend(frameon=False, fontsize=4.6, loc="upper center", bbox_to_anchor=(0.45, 1.3), ncol=1, handlelength=1, borderaxespad=0)
for s in ax.spines.values(): s.set_linewidth(0.5)
t(X[2]+PW/2, 0.13, "A 30-s window identifies\nthe child with 90% accuracy", ha="center", va="center",
  fontsize=4.8, fontweight="bold", color=GREEN, linespacing=1.2)

# 4 take-home
panel(X[3], "Take-home", ORANGE)
t(X[3]+0.012, 0.79, "Exercise/rest context,\nnot wrist biosignals,\nexplains reported pain.", fontsize=5.4, fontweight="bold",
  color=INK, va="top", linespacing=1.25)
t(X[3]+0.012, 0.50, "For wearable pain studies:", fontsize=5.0, fontweight="bold", color=ORANGE, va="top")
t(X[3]+0.012, 0.425, "✓ hold out whole participants\n✓ report context-only\n    baselines\n✓ state the smallest\n    detectable effect\n✓ audit public data first",
  fontsize=4.8, color=INK, va="top", linespacing=1.3)

fig.savefig(OUT + ".pdf")
buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=600, facecolor="white"); plt.close(fig)
img = Image.open(buf).convert("RGB")
img.save(OUT + ".tif", dpi=(600, 600), compression="tiff_lzw"); img.save(OUT + ".png", dpi=(600, 600))
print(img.size)
