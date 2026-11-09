"""Build a blind listening test of overall quality (OVL) across guidance scales, as a local HTML page.

For each of --n-captions captions drawn at random (--seed), one page holds the clips generated at --scales plus a
low anchor: the clip at --anchor-scale low-pass filtered at 3.5 kHz, the standard anchor of ITU-R BS.1534-3
(MUSHRA). The clips on a page are shuffled and named by letter only; which letter is which scale is written to
key.json, which the page does not read. Every clip is loudness-normalized to --target-lufs (ITU-R BS.1770, via
torchaudio) so that no clip is louder than another; the script stops if that would push any clip past full scale.
The submitted files are not touched.

Output in --out-dir: index.html, audio/*.flac, key.json, config.json. Open index.html in a browser, rate each clip
from 1 to 100, and download the ratings as CSV from the page.

    python listening/make_test.py
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy import signal

RUN_DIRS = {s: f"outputs/sweep/scale_{s}" for s in [0, 1, 3, 5, 7, 9, 11, 13]}


def anchor_filter(sr):
    """Linear-phase FIR low-pass meeting BS.1534-3: fc 3.5 kHz, ripple <= 0.1 dB, >= 25 dB at 4 kHz, >= 50 dB at 4.5 kHz."""
    numtaps, beta = signal.kaiserord(60, 500 / (sr / 2))
    numtaps |= 1  # odd length, so the delay is a whole number of samples
    taps = signal.firwin(numtaps, 3750, window=("kaiser", beta), fs=sr)
    freqs, response = signal.freqz(taps, worN=[1000, 3500, 4000, 4500], fs=sr)
    gain_db = 20 * np.log10(np.abs(response))
    check = {f"{f:.0f} Hz": round(float(g), 3) for f, g in zip(freqs, gain_db)}
    assert gain_db[1] > -0.1 and gain_db[2] < -25 and gain_db[3] < -50, check
    return taps, check


def loudness(x, sr):
    return float(torchaudio.functional.loudness(torch.from_numpy(x.T.astype(np.float32)), sr))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", default="test.csv")
    parser.add_argument("--scales", type=int, nargs="+", default=[5, 7, 9, 11, 13])
    parser.add_argument("--anchor-scale", type=int, default=7)
    parser.add_argument("--n-captions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-lufs", type=float, default=-23.0)
    parser.add_argument("--out-dir", default="outputs/listening/ovl")
    args = parser.parse_args()

    out = Path(args.out_dir)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    n_total = sum(1 for _ in open(args.captions)) - 1
    caption_ids = rng.sample(range(n_total), args.n_captions)

    taps, filter_check = None, None
    pages, key = [], {}
    for page, cid in enumerate(caption_ids, start=1):
        stimuli = [(f"scale {s}", RUN_DIRS[s], False) for s in args.scales]
        stimuli.append((f"anchor (scale {args.anchor_scale}, 3.5 kHz low-pass)", RUN_DIRS[args.anchor_scale], True))
        rng.shuffle(stimuli)
        items = []
        for letter, (label, run_dir, is_anchor) in zip("ABCDEFGHIJKL", stimuli):
            x, sr = sf.read(Path(run_dir) / "wav" / f"{cid}.wav", dtype="float64")
            if is_anchor:
                if taps is None:
                    taps, filter_check = anchor_filter(sr)
                x = signal.lfilter(taps, 1.0, np.pad(x, ((0, len(taps) // 2), (0, 0))), axis=0)[len(taps) // 2:]
            gain = 10 ** ((args.target_lufs - loudness(x, sr)) / 20)
            y = x * gain
            peak = float(np.abs(y).max())
            assert peak <= 1.0, f"caption {cid} {label}: peak {peak:.3f} after normalizing to {args.target_lufs} LUFS"
            name = f"p{page:02d}_{letter}.flac"
            sf.write(out / "audio" / name, y, sr, subtype="PCM_16")
            items.append({"letter": letter, "file": f"audio/{name}"})
            key[f"p{page:02d}_{letter}"] = {"page": page, "caption_id": cid, "letter": letter, "condition": label,
                                            "gain_db": round(20 * np.log10(gain), 2), "peak_after": round(peak, 3)}
        pages.append({"page": page, "caption_id": cid, "items": items})

    with open(out / "key.json", "w") as f:
        json.dump(key, f, indent=2)
    with open(out / "config.json", "w") as f:
        json.dump({**vars(args), "caption_ids": caption_ids, "anchor_filter_gain_db": filter_check,
                   "anchor_filter_taps": len(taps)}, f, indent=2)
    html = (Path(__file__).parent / "page.html").read_text()
    (out / "index.html").write_text(html.replace("__PAGES__", json.dumps(pages)))
    print(f"{len(pages)} pages x {len(pages[0]['items'])} clips in {out}; captions {caption_ids}")
    print(f"anchor filter gain: {filter_check}")
    print(f"max peak after normalization: {max(v['peak_after'] for v in key.values())}")


if __name__ == "__main__":
    main()
