# HW3 sample code: classifier-free guidance for Stable Audio Open

## Setup
```bash
conda create -n deepmir_hw3 python=3.11 && conda activate deepmir_hw3
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
`stabilityai/stable-audio-open-1.0` is gated on Hugging Face: accept its license, then run `huggingface-cli login`.
Generation needs about 9 GB of GPU memory (fp16).

## Evaluation environment
Retrieval and evaluation run in a separate environment, which installs `requirements.txt` unchanged plus av-benchmark
and its dependencies:
```bash
git submodule update --init
conda create -n deepmir_hw3_eval python=3.11 && conda activate deepmir_hw3_eval
pip install torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements_eval.txt
hf download lukewys/laion_clap music_audioset_epoch_15_esc_90.14.pt --local-dir third_party/av-benchmark/weights
wget https://github.com/hkchengrex/MMAudio/releases/download/v0.1/synchformer_state_dict.pth -O third_party/av-benchmark/weights/synchformer_state_dict.pth
```
`third_party/av-benchmark` is a git submodule on the `hw3` branch of
[bloggerwang1217/av-benchmark](https://github.com/bloggerwang1217/av-benchmark), a fork of
[hkchengrex/av-benchmark](https://github.com/hkchengrex/av-benchmark) at commit `f351b9a` with these changes
(`git -C third_party/av-benchmark diff f351b9a`):
- LAION-CLAP loads `music_audioset_epoch_15_esc_90.14.pt`, the checkpoint required by the homework, instead of
  `music_speech_audioset_epoch_15_esc_89.98.pt`.
- MS-CLAP is removed. It calls `tokenizer.encode_plus`, which transformers 5 no longer has.
- VGGish is loaded with `trust_repo=True`, so torch.hub does not stop to ask whether to trust `harritaylor/torchvggish`.
- The git dependencies laion-clap, passt and imagebind are pinned to fixed commits.

On the first run, the weights of ImageBind, PaSST, VGGish and PANNs are downloaded into `~/.cache`. ImageBind and
Synchformer are only used for video metrics, but av-benchmark loads them on start.

## Files
| File | Description |
| --- | --- |
| `cfg.py` | **Exercise**: implement `cfg(noise_pred_uncond, noise_pred_cond, guidance_scale)`. |
| `generate.py` | The `StableAudioPipeline` sampling loop with the CFG step replaced by your `cfg`. |

## Exercise
1. Implement `cfg` in `cfg.py`.
2. Generate audio and compare with diffusers' own pipeline:
```bash
python generate.py --prompt "jazz, electric guitar, piano, relaxing" --out out.wav --check
```
With a correct implementation the difference is `0` (written in the same form as diffusers) or very small.

Note: Stable Audio Open is a v-prediction model, so the model outputs are v rather than noise. CFG is linear, so the
formula is the same.
