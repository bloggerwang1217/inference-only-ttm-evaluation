"""Draw the report figures for the guidance-scale sweep, the listening test and MQAScore from their summary CSVs.

Reads outputs/sweep/summary.csv (summarize_sweep.py), outputs/listening/ovl/ovl_summary.csv (listening/analyze.py),
outputs/mqascore/summary_with_retrieved.csv (mqascore/summarize.py --runs 5=... 7=... 9=... retrieved=outputs/reference
--out outputs/mqascore/summary_with_retrieved.csv) and outputs/mqascore/summary_vs5.csv
(mqascore/summarize.py --paired-with 5 --out outputs/mqascore/summary_vs5.csv) and outputs/apg/summary.csv
(summarize_apg.py). Figure 4 is an illustration with made-up 2-D values, not data.

    python plot_figures.py            # writes outputs/figures/fig1a_sweep.png, fig1b_sweep_vs7.png, fig2_listening.png, fig3_mqascore.png, fig3b_mqascore_vs5.png,
                                      # fig4_cfg_vs_apg.png, fig5_apg_vs_cfg.png
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

# Figure 3: MQAScore by dimension, scales 5, 7 and 9 and the retrieved MusicCaps clips side by side
mq = rows(REPO / "outputs/mqascore/summary_with_retrieved.csv")
names = [("MQAScore", "All tags"), ("genre", "Genre"), ("instrument", "Instrument"),
         ("mood_theme", "Mood/theme"), ("context", "Context")]
scales = [("5", "#2a78d6"), ("7", "#eb6834"), ("9", "#1baf7a")]  # categorical slots 1-3 of the reference palette
fig, ax = plt.subplots(figsize=(13, 4.6))
w = 0.2
bars = [("retrieved", GRAY, "Retrieved")] + [(sc, col, f"Scale {sc}") for sc, col in scales]
for j, (sc, col, lab) in enumerate(bars):
    for i, (key, _) in enumerate(names):
        r = next(r for r in mq if r["metric"] == key and r["guidance_scale"] == sc)
        x, m = i + (j - 1.5) * w, float(r["mean"])
        ax.bar(x, m, width=w - 0.03, color=col, zorder=2, label=lab if i == 0 else None)
        ax.plot([x, x], [float(r["ci_low"]), min(float(r["ci_high"]), 1.0)], color=INK, lw=1.5, zorder=3)
        ax.text(x, 0.02, f"{m:.2f}", ha="center", va="bottom", color="white", fontsize=11, zorder=4, rotation=90)
ax.set_xticks(range(len(names)))
ax.set_xticklabels([f"{lab} ({next(r for r in mq if r['metric']==k)['n_captions']})" for k, lab in names])
ax.set_ylim(0, 1.05)
ax.set_ylabel("P(Yes)")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", ncol=4, fontsize=13, frameon=False, bbox_to_anchor=(0, 1.12))
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

# Figure 4: CFG and APG side by side as vectors, an illustration with made-up 2-D values and w = 3
CFG_C, APG_C = "#2a78d6", "#eb6834"  # categorical slots 1-2 of the reference palette
dc, du, w = (3.0, 2.0), (2.3, 0.6), 3
dd = (dc[0] - du[0], dc[1] - du[1])
k = (dd[0] * dc[0] + dd[1] * dc[1]) / (dc[0] ** 2 + dc[1] ** 2)
par = (k * dc[0], k * dc[1])
perp = (dd[0] - par[0], dd[1] - par[1])
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))


def arrow(ax, p, q, color, lw=2.5, ls="-"):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                                    shrinkA=0, shrinkB=0, mutation_scale=18))


for ax, title in zip(axes, ["CFG", "APG"]):
    ax.plot([0, dc[0] * 1.75], [0, dc[1] * 1.75], ":", color=GRAY, lw=1.5, zorder=1)
    ax.text(0.45, 0.4, "direction of $D_{cond}$", rotation=33.7, rotation_mode="anchor", ha="left", va="bottom",
            color=INK2, fontsize=12)
    for p, lab, off in [((0, 0), "origin", (-0.1, -0.35)), (du, "$D_{uncond}$", (0.12, -0.3)),
                        (dc, "$D_{cond}$", (0.15, -0.15))]:
        ax.plot(*p, "o", color=INK, ms=7, zorder=4)
        ax.text(p[0] + off[0], p[1] + off[1], lab, color=INK, fontsize=14)
    arrow(ax, du, dc, INK2, lw=2)
    ax.text(du[0] + 0.45, du[1] + 0.55, "$\\Delta D$", color=INK2, fontsize=14, ha="left")
    ax.set_aspect("equal")
    ax.set_xlim(-0.4, 5.6)
    ax.set_ylim(-0.6, 5.2)
    ax.axis("off")
    ax.set_title(title, color=INK, loc="left")
a1, a2 = axes
cfg_out = (du[0] + w * dd[0], du[1] + w * dd[1])
arrow(a1, dc, cfg_out, CFG_C)
a1.plot(*cfg_out, "o", color=CFG_C, ms=8, zorder=4)
a1.text(cfg_out[0] + 0.15, cfg_out[1] - 0.1, "$D_{uncond} + w\\,\\Delta D$", color=INK, fontsize=14)
foot = (dc[0] + par[0], dc[1] + par[1])
arrow(a2, dc, foot, GRAY, lw=2, ls="--")
a2.text(foot[0] + 0.15, foot[1] - 0.25, "$\\Delta D_{\\parallel}$", color=INK2, fontsize=14)
apg_out = (dc[0] + (w - 1) * perp[0], dc[1] + (w - 1) * perp[1])
arrow(a2, dc, apg_out, APG_C)
a2.plot(*apg_out, "o", color=APG_C, ms=8, zorder=4)
a2.text(apg_out[0] - 0.15, apg_out[1] + 0.15, "$D_{cond} + (w-1)\\,\\Delta D_{\\perp}$", color=INK, fontsize=14,
        ha="center", va="bottom")
fig.tight_layout()
fig.savefig(OUT / "fig4_cfg_vs_apg.png")

# Figure 5: APG and CFG at the scales where CFG distorts (9 to 13), with CFG at scale 5 as a reference line,
# from summarize_apg.py
ap = rows(REPO / "outputs/apg/summary.csv")
ref = next(r for r in ap if r["guidance_scale"] == "5")
hi_rows = [r for r in ap if int(r["guidance_scale"]) >= 9]
xs = [int(r["guidance_scale"]) for r in hi_rows]
fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
panels = [("FD-VGG (lower is better)", "FD-VGG", False), ("KL-PaSST (lower is better)", "KL-PASST-softmax", True),
          ("CLAP score (higher is better)", "LAION-CLAP-Score", True), ("Clipped clips, of 100", "clipped clips", False)]


def value(r, key, lab):
    v = r[f"{key} {lab}"]
    return int(v.split("/")[0]) if key == "clipped clips" else float(v)


for a, (title, key, has_ci) in zip(axes, panels):
    a.axhline(value(ref, key, "CFG"), color=CFG_C, ls="--", lw=1.5, zorder=1)
    for j, (lab, col) in enumerate((("CFG", CFG_C), ("APG", APG_C))):
        off = -0.15 if j == 0 else 0.15
        if has_ci:
            for r, x in zip(hi_rows, xs):
                a.plot([x + off, x + off], [float(r[f"{key} {lab} CI low"]), float(r[f"{key} {lab} CI high"])],
                       color=col, lw=2, solid_capstyle="round", zorder=2)
        a.plot([x + off for x in xs], [value(r, key, lab) for r in hi_rows], "-o", color=col, ms=8, mec="white",
               mew=1.5, lw=1.5, zorder=3, label=f"{lab} at the scale")
    a.set_title(title, color=INK, loc="left")
    a.set_xticks(xs)
    a.set_xlim(8, 14)
    a.grid(axis="x", visible=False)
    a.set_xlabel("Guidance scale")
axes[3].set_ylim(0, 70)
handles = [Line2D([], [], color=CFG_C, lw=1.5, marker="o", ms=8, mec="white"),
           Line2D([], [], color=APG_C, lw=1.5, marker="o", ms=8, mec="white"),
           Line2D([], [], color=CFG_C, lw=1.5, ls="--")]
axes[3].legend(handles, ["CFG", "APG", "CFG at scale 5"], loc="lower right", fontsize=12, frameon=True,
               edgecolor=GRID, framealpha=1)
fig.tight_layout()
fig.savefig(OUT / "fig5_apg_vs_cfg.png")

print("done")
