#!/usr/bin/env bash
# Evaluate one generated run against the reference set (prepare_eval.py) with av-benchmark.
#
# Reference features are extracted once into $REF_DIR/cache and reused for every run; they only
# change if retrieval.json or test.csv changes, in which case delete $REF_DIR/cache. Features of the
# run are recomputed on every call into <run-dir>/eval_cache, which also receives output_metrics.json.
# Run from the deepmir_hw3_eval environment:
#
#     CUDA_VISIBLE_DEVICES=0 bash evaluate.sh outputs/baseline
set -euo pipefail
cd "$(dirname "$0")"

RUN_DIR=${1:?usage: bash evaluate.sh <run-dir>}
REF_DIR=${REF_DIR:-outputs/reference}
AV_BENCH=third_party/av-benchmark

# 1. CLAP text embeddings of the captions, saved into the reference cache (evaluate.py reads them there)
if [ ! -f "$REF_DIR/cache/clap_laion_text.pth" ]; then
    python "$AV_BENCH/extract_text.py" --text_csv "$REF_DIR/captions.csv" --output_cache_path "$REF_DIR/cache"
fi

# 2. Audio features of the reference (first call only) and of the run, then all metrics.
#    Pairing is by filename: the generated k.wav against the reference k.wav and caption k.
python "$AV_BENCH/evaluate.py" \
    --gt_audio "$REF_DIR/wav" --gt_cache "$REF_DIR/cache" \
    --pred_audio "$RUN_DIR/wav" --pred_cache "$RUN_DIR/eval_cache" --recompute_pred_cache \
    --audio_length 10 --skip_video_related

# 3. The three required metrics, with 95% confidence intervals for the two that are means over the pairs
python summarize_eval.py --ref-cache "$REF_DIR/cache" --run-cache "$RUN_DIR/eval_cache"
