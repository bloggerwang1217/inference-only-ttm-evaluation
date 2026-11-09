"""Summarize the overall-quality (OVL) listening test built by listening/make_test.py.

Joins <dir>/ovl_ratings.csv (saved by the page) with <dir>/key.json, then reports for every condition the mean
score over the captions with a 95% t-interval, and, for every condition, the mean of its score minus the score of
--paired-with on the same caption, with a 95% t-interval. Also reports on how many pages the anchor got the lowest
score (strictly, tied, or above another clip), and how many clips were played to the end. Writes <dir>/ovl_summary.csv and prints the table.

    python listening/analyze.py
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def mean_ci(x, level=0.95):
    x = np.asarray(x, dtype=float)
    half = stats.t.ppf((1 + level) / 2, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return x.mean(), x.mean() - half, x.mean() + half


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="outputs/listening/ovl")
    parser.add_argument("--paired-with", default="scale 7")
    args = parser.parse_args()
    d = Path(args.dir)

    key = json.load(open(d / "key.json"))
    scores = defaultdict(dict)  # condition -> caption_id -> score
    played = defaultdict(int)
    pages = defaultdict(dict)  # page -> condition -> score
    with open(d / "ovl_ratings.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(key), f"{len(rows)} ratings for {len(key)} clips"
    for r in rows:
        k = key[f"p{int(r['page']):02d}_{r['letter']}"]
        assert int(r["caption_id"]) == k["caption_id"], r
        assert r["score"] != "", f"page {r['page']} clip {r['letter']} is not rated"
        cond = "anchor" if k["condition"].startswith("anchor") else k["condition"]
        scores[cond][k["caption_id"]] = float(r["score"])
        pages[k["page"]][cond] = float(r["score"])
        played[cond] += int(r["played_full"])

    def order(c):
        return (1, 0) if c == "anchor" else (0, int(c.split()[1]))

    base = scores[args.paired_with]
    table = []
    for cond in sorted(scores, key=order):
        ids = sorted(scores[cond])
        x = [scores[cond][i] for i in ids]
        m, lo, hi = mean_ci(x)
        row = {"condition": cond, "n": len(x), "mean": m, "ci_low": lo, "ci_high": hi, "played_full": played[cond]}
        if cond != args.paired_with:
            dm, dlo, dhi = mean_ci([scores[cond][i] - base[i] for i in ids])
            row.update({"diff_mean": dm, "diff_ci_low": dlo, "diff_ci_high": dhi})
        table.append(row)
    others = {page: [v for c, v in p.items() if c != "anchor"] for page, p in pages.items()}
    anchor_strict = sum(p["anchor"] < min(others[page]) for page, p in pages.items())
    anchor_tied = sum(p["anchor"] == min(others[page]) for page, p in pages.items())
    anchor_above = sorted(page for page, p in pages.items() if p["anchor"] > min(others[page]))

    with open(d / "ovl_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "n", "mean", "ci_low", "ci_high", "diff_mean",
                                               "diff_ci_low", "diff_ci_high", "played_full"])
        writer.writeheader()
        writer.writerows(table)

    print(f"{'condition':<10} {'n':>3} {'mean [95% CI]':>22}   {'minus ' + args.paired_with + ' [95% CI]':>28}   played to end")
    for r in table:
        diff = (f"{r['diff_mean']:+6.1f} [{r['diff_ci_low']:+6.1f}, {r['diff_ci_high']:+6.1f}]"
                if "diff_mean" in r else "—")
        print(f"{r['condition']:<10} {r['n']:>3} {r['mean']:6.1f} [{r['ci_low']:5.1f}, {r['ci_high']:5.1f}]   "
              f"{diff:>28}   {r['played_full']}/{r['n']}")
    print(f"anchor below every other clip on {anchor_strict} of {len(pages)} pages, tied for lowest on {anchor_tied}, "
          f"above at least one other clip on pages {anchor_above}")
    print(f"Saved to {d / 'ovl_summary.csv'}")


if __name__ == "__main__":
    main()
