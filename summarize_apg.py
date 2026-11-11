"""Table of APG against CFG at the same guidance scale: one row per scale.

For each scale, reads the CFG run of the sweep and the APG run (generate_all.py --apg) from their eval_cache
(evaluate.sh). Both use seed k for caption k, so clip k of the two runs differs only in the guidance method.
Columns: FD-VGG of each; KL-PASST-softmax and LAION-CLAP-Score of each as mean and 95% t-interval, and the mean of
the clip-by-clip difference APG - CFG with a 95% t-interval; and the clipped clips of each, counted as in
summarize_sweep.py. Writes --out as CSV and prints it.

    python summarize_apg.py
"""

import argparse
import csv
import json
from pathlib import Path

from summarize_sweep import METRICS, clipped_clips, mean_ci, per_clip

CFG_RUNS = {5: "outputs/sweep/scale_5", 7: "outputs/baseline_seed_per_caption", 9: "outputs/sweep/scale_9",
            11: "outputs/sweep/scale_11", 13: "outputs/sweep/scale_13"}


def fd(run_dir):
    with open(Path(run_dir) / "eval_cache" / "summary.json") as f:
        return json.load(f)["FD-VGG"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apg-dir", default="outputs/apg/scale_{scale}_r11", help="APG run per scale")
    parser.add_argument("--out", default="outputs/apg/summary.csv")
    args = parser.parse_args()

    table = []
    for scale, cfg_dir in CFG_RUNS.items():
        apg_dir = args.apg_dir.format(scale=scale)
        names, cfg = per_clip(cfg_dir)
        apg_names, apg = per_clip(apg_dir)
        assert names == apg_names, f"{apg_dir} does not hold the same clips as {cfg_dir}"
        row = {"guidance_scale": scale, "CFG run": cfg_dir, "APG run": apg_dir,
               "FD-VGG CFG": f"{fd(cfg_dir):.4f}", "FD-VGG APG": f"{fd(apg_dir):.4f}"}
        for m in METRICS:
            for label, values in (("CFG", cfg), ("APG", apg)):
                mean, lo, hi = mean_ci(values[m])
                row.update({f"{m} {label}": f"{mean:.4f}", f"{m} {label} CI low": f"{lo:.4f}",
                            f"{m} {label} CI high": f"{hi:.4f}"})
            d, lo, hi = mean_ci(apg[m] - cfg[m])
            row.update({f"{m} APG minus CFG": f"{d:+.4f}", f"{m} difference CI low": f"{lo:+.4f}",
                        f"{m} difference CI high": f"{hi:+.4f}"})
        for label, run_dir in (("CFG", cfg_dir), ("APG", apg_dir)):
            clipped, n = clipped_clips(run_dir)
            row[f"clipped clips {label}"] = f"{clipped}/{n}"
        table.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    kl, cl = METRICS
    print(f"{'scale':>5} | {'FD CFG':>6} {'FD APG':>6} | {'KL APG - CFG [95% CI]':>28} | "
          f"{'CLAP APG - CFG [95% CI]':>28} | clipped CFG, APG")
    for r in table:
        print(f"{r['guidance_scale']:>5} | {r['FD-VGG CFG']:>6} {r['FD-VGG APG']:>6} | "
              f"{r[kl + ' APG minus CFG']} [{r[kl + ' difference CI low']}, {r[kl + ' difference CI high']}] | "
              f"{r[cl + ' APG minus CFG']} [{r[cl + ' difference CI low']}, {r[cl + ' difference CI high']}] | "
              f"{r['clipped clips CFG']}, {r['clipped clips APG']}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
