"""Text-to-audio retrieval with LAION-CLAP: for each test caption, find the closest MusicCaps clip.

Audio preprocessing: mono, resample to 48 kHz, keep the first `--seconds` seconds. Unlike the CLAP path of
av-benchmark (av_bench/data/audio_dataset.py), the DC offset is not removed.

    python retrieve.py --out retrieval.json
"""

import argparse
import csv
import json
from pathlib import Path

import laion_clap
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

CLAP_SR = 48_000


def load_audio(path, seconds):
    waveform, sr = torchaudio.load(path)
    waveform = waveform.mean(dim=0)  # mono
    if sr != CLAP_SR:
        waveform = torchaudio.functional.resample(waveform, sr, CLAP_SR)
    return waveform[: int(CLAP_SR * seconds)]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", default="test.csv")
    parser.add_argument("--audio-dir", default="data/audio_mp3")
    parser.add_argument("--ckpt", default="third_party/av-benchmark/weights/music_audioset_epoch_15_esc_90.14.pt")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--out", default="retrieval.json")
    args = parser.parse_args()

    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base").cuda().eval()
    model.load_ckpt(args.ckpt, verbose=False)

    # 1. Text embeddings of the test captions
    with open(args.captions, newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [row["id"] for row in rows]
    text = model.get_text_embedding([row["caption"] for row in rows], use_tensor=True)  # (num_captions, D)

    # 2. Audio embeddings of the MusicCaps clips, one clip at a time (clips may differ slightly in length)
    paths = sorted(Path(args.audio_dir).glob("*.mp3"))
    audio = []
    for path in tqdm(paths, desc="CLAP audio"):
        wav = load_audio(path, args.seconds).unsqueeze(0).cuda()
        audio.append(model.get_audio_embedding_from_data(wav, use_tensor=True)[0])
    audio = torch.stack(audio)  # (num_clips, D)

    # 3. Cosine similarity, best clip per caption
    sim = F.normalize(text, dim=-1) @ F.normalize(audio, dim=-1).T  # (num_captions, num_clips)
    best = sim.argmax(dim=1).tolist()
    retrieval = {i: paths[j].stem for i, j in zip(ids, best)}

    with open(args.out, "w") as f:
        json.dump(retrieval, f, indent=4)
    print(f"Saved {len(retrieval)} captions to {args.out} ({len(paths)} clips searched)")


if __name__ == "__main__":
    main()
