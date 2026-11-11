"""Draw the report figures for the guidance-scale sweep, the listening test and MQAScore from their summary CSVs.

Reads outputs/sweep/summary.csv (summarize_sweep.py), outputs/listening/ovl/ovl_summary.csv (listening/analyze.py),
outputs/mqascore/summary.csv (mqascore/summarize.py) and outputs/mqascore/summary_vs5.csv
(mqascore/summarize.py --paired-with 5 --out outputs/mqascore/summary_vs5.csv).

    python plot_figures.py            # writes outputs/figures/fig1a_sweep.png, fig1b_sweep_vs7.png, fig2_listening.png, fig3_mqascore.png, fig3b_mqascore_vs5.png
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent
OUT = REPO / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, GRAY = "#2a78d6", "#8a8984"
plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "savefig.dpi": 200,
})


def ci_legend(ax, gray_label):
    """Legend for the paired-difference plots: gray = 95% CI covers 0, blue = it excludes 0."""
    handles = [Line2D([], [], color=c, lw=2.5, marker="o", ms=8, mec="white", mew=1.5) for c in (GRAY, BLUE)]
    ax.legend(handles, [gray_label, "95% CI excludes 0"], loc="lower right", fontsize=13, frameon=True,
              edgecolor=GRID, framealpha=1)


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# Figure 1a: sweep, absolute values at every scale
sw = rows(REPO / "outputs/sweep/summary.csv")
g = [int(r["guidance_scale"]) for r in sw]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
panels = [
    ("FD-VGG (lower is better)", "FD-VGG", None, None),
    ("KL-PaSST (lower is better)", "KL-PASST-softmax", "KL-PASST-softmax CI low", "KL-PASST-softmax CI high"),
    ("CLAP score (higher is better)", "LAION-CLAP-Score", "LAION-CLAP-Score CI low", "LAION-CLAP-Score CI high"),
]
for ax, (title, m, lo, hi) in zip(axes, panels):
    y = [float(r[m]) for r in sw]
    if lo:
        for r, x in zip(sw, g):
            ax.plot([x, x], [float(r[lo]), float(r[hi])], color=BLUE, lw=2, solid_capstyle="round", zorder=2)
    ax.plot(g, y, "-o", color=BLUE, ms=8, mec="white", mew=1.5, lw=1.2, zorder=3)
    ax.set_xticks(g)
    ax.set_xlim(-0.8, 13.8)
    ax.grid(axis="x", visible=False)
    ax.set_title(title, color=INK, loc="left")
    ax.set_xlabel("Guidance scale")
fig.tight_layout()
fig.savefig(OUT / "fig1a_sweep.png")

# Figure 1b: sweep, paired difference from scale 7, scales 3 to 13 (scales 0 and 1 are in Figure 1a)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
panels = [
    ("KL-PaSST − scale 7", "KL-PASST-softmax", "above 0: worse than 7"),
    ("CLAP score − scale 7", "LAION-CLAP-Score", "below 0: worse than 7"),
]
for ax, (title, m, note) in zip(axes, panels):
    ax.axhline(0, color=INK2, lw=1.5, zorder=1)
    for r, x in zip(sw, g):
        if x < 3 or x == 7:
            continue
        d = float(r[f"{m} minus scale 7"])
        lo, hi = float(r[f"{m} difference CI low"]), float(r[f"{m} difference CI high"])
        c = BLUE if lo > 0 or hi < 0 else GRAY
        ax.plot([x, x], [lo, hi], color=c, lw=2.5, solid_capstyle="round", zorder=2)
        ax.plot(x, d, "o", ms=8, color=c, mec="white", mew=1.5, zorder=3)
    g3 = [x for x in g if x >= 3]
    ax.set_xticks(g3)
    ax.set_xticklabels([str(x) if x != 7 else "7\n(ref.)" for x in g3])
    ax.set_xlim(2, 14)
    ax.grid(axis="x", visible=False)
    ax.set_title(f"{title}  ({note})", color=INK, loc="left")
    ax.set_xlabel("Guidance scale")
ci_legend(axes[1], "95% CI covers 0 (ties with 7)")
fig.tight_layout()
fig.savefig(OUT / "fig1b_sweep_vs7.png")

# Figure 2: listening test
ov = rows(REPO / "outputs/listening/ovl/ovl_summary.csv")
sc = [r for r in ov if r["condition"].startswith("scale")]
anchor = next(r for r in ov if r["condition"] == "anchor")
xs = [int(r["condition"].split()[1]) for r in sc]
clip = {int(r["guidance_scale"]): int(r["clipped clips"].split("/")[0]) for r in sw}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
a1.axhline(float(anchor["mean"]), color=GRAY, ls="--", lw=2, zorder=1)
a1.text(5.3, float(anchor["mean"]) - 1.2, "low anchor (49):\nscale 7 low-passed at 3.5 kHz", ha="left", va="top", color=INK2, fontsize=13)
for r, x in zip(sc, xs):
    a1.plot([x, x], [float(r["ci_low"]), float(r["ci_high"])], color=BLUE, lw=2, zorder=2)
a1.plot(xs, [float(r["mean"]) for r in sc], "-o", color=BLUE, ms=8, mec="white", mew=1.5, lw=1.2, zorder=3)
for r, x in zip(sc, xs):
    a1.text(x + 0.2, float(r["mean"]) + 1.5, f"{float(r['mean']):.0f}", va="bottom", color=INK, fontsize=13)
a1.set_xticks(xs)
a1.grid(axis="x", visible=False)
a1.set_xlim(4, 14)
a1.set_ylim(30, 80)
a1.set_xlabel("Guidance scale")
a1.set_title("OVL, 1–100 (mean ± 95% CI, 10 captions)", color=INK, loc="left")
a2.plot(xs, [clip[x] for x in xs], "-o", color=BLUE, ms=8, mec="white", mew=1.5, lw=1.2, zorder=3)
for x in xs:
    a2.text(x + 0.2, clip[x] + 1.5, str(clip[x]), va="bottom", color=INK, fontsize=13)
a2.set_xticks(xs)
a2.set_xlim(4, 14)
a2.set_ylim(0, 70)
a2.set_xlabel("Guidance scale")
a2.set_title("Clipped clips, of 100", color=INK, loc="left")
a2.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(OUT / "fig2_listening.png")

# Figure 3: MQAScore by dimension, scales 5, 7 and 9 side by side
mq = rows(REPO / "outputs/mqascore/summary.csv")
names = [("MQAScore", "All tags"), ("genre", "Genre"), ("instrument", "Instrument"),
         ("mood_theme", "Mood/theme"), ("context", "Context")]
scales = [("5", "#2a78d6"), ("7", "#eb6834"), ("9", "#1baf7a")]  # categorical slots 1-3 of the reference palette
fig, ax = plt.subplots(figsize=(13, 4.6))
w = 0.26
for j, (sc, col) in enumerate(scales):
    for i, (key, _) in enumerate(names):
        r = next(r for r in mq if r["metric"] == key and r["guidance_scale"] == sc)
        x, m = i + (j - 1) * w, float(r["mean"])
        ax.bar(x, m, width=w - 0.03, color=col, zorder=2, label=f"Scale {sc}" if i == 0 else None)
        ax.plot([x, x], [float(r["ci_low"]), min(float(r["ci_high"]), 1.0)], color=INK, lw=1.5, zorder=3)
        ax.text(x, 0.02, f"{m:.2f}", ha="center", va="bottom", color="white", fontsize=11, zorder=4, rotation=90)
ax.set_xticks(range(len(names)))
ax.set_xticklabels([f"{lab} ({next(r for r in mq if r['metric']==k)['n_captions']})" for k, lab in names])
ax.set_ylim(0, 1.05)
ax.set_ylabel("P(Yes)")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", ncol=3, fontsize=13, frameon=False, bbox_to_anchor=(0, 1.12))
ax.set_title("Mean and 95% CI over the captions that have the dimension", color=INK2, fontsize=13, loc="right")
fig.tight_layout()
fig.savefig(OUT / "fig3_mqascore.png")
# Figure 3b: MQAScore, scales 7 and 9 minus scale 5 (paired over captions), from mqascore/summarize.py --paired-with 5
mq5 = rows(REPO / "outputs/mqascore/summary_vs5.csv")
fig, ax = plt.subplots(figsize=(13, 4.6))
for j, (sc, col) in enumerate(scales[1:]):
    for i, (key, _) in enumerate(names):
        r = next(r for r in mq5 if r["metric"] == key and r["guidance_scale"] == sc)
        d, l, h = float(r["diff_mean"]), float(r["diff_ci_low"]), float(r["diff_ci_high"])
        y = len(names) - 1 - i + (0.15 if j == 0 else -0.15)
        excl = l > 0 or h < 0
        ax.plot([l, h], [y, y], color=col, lw=2.5, solid_capstyle="round", zorder=2)
        ax.plot(d, y, "o", ms=9, color=col if excl else "white", mec=col, mew=2, zorder=3)
ax.set_yticks(range(len(names)))
ax.set_yticklabels([f"{lab} ({next(r for r in mq5 if r['metric']==k)['n_captions']})" for k, lab in names][::-1])
ax.axvline(0, color=INK2, lw=1.5)
ax.set_xlim(-0.05, 0.12)
ax.set_xlabel("Scale 7 or 9 − scale 5, P(Yes)  (mean and 95% CI of the paired difference)")
ax.grid(axis="y", visible=False)
ax.text(-0.004, len(names) - 0.4, "← scale 5 higher", color=INK2, fontsize=13, ha="right")
ax.text(0.004, len(names) - 0.4, "higher than scale 5 →", color=INK2, fontsize=13)
ax.set_ylim(-0.6, len(names) - 0.15)
handles = [Line2D([], [], color=c, lw=2.5, marker="o", ms=9, mfc=f, mec=c, mew=2)
           for c, f in [(scales[1][1], scales[1][1]), (scales[2][1], scales[2][1]), (GRAY, GRAY), (GRAY, "white")]]
ax.legend(handles, ["Scale 7 − 5", "Scale 9 − 5", "filled: 95% CI excludes 0", "hollow: 95% CI covers 0"],
          loc="lower right", fontsize=12, frameon=True, edgecolor=GRID, framealpha=1)
fig.tight_layout()
fig.savefig(OUT / "fig3b_mqascore_vs5.png")

print("done")
