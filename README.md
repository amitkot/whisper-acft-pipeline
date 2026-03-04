# whisper-acft-pipeline

Reproducible [ACFT (Audio-Context Fine-Tuning)](https://github.com/futo-org/whisper-acft) pipeline for Whisper models, with conversion to ggml format for use with [FUTO Keyboard](https://keyboard.futo.org/) and [whisper.cpp](https://github.com/ggml-org/whisper.cpp).

Includes a training config for Hebrew ([mike249/whisper-tiny-he-2](https://huggingface.co/mike249/whisper-tiny-he-2) on [Google FLEURS](https://huggingface.co/datasets/google/fleurs)), but can be used with any Whisper model and language by adding a YAML config.

## Setup

```bash
git clone --recurse-submodules https://github.com/amitkot/whisper-acft-pipeline.git
cd whisper-acft-pipeline
./scripts/bootstrap.sh
```

Or if already cloned:

```bash
./scripts/bootstrap.sh
```

This will:
1. Initialize git submodules (`external/whisper.cpp`, `external/whisper`)
2. Build whisper.cpp (cmake)
3. Install Python dependencies via `uv sync`

## Usage

### Train

```bash
uv run python scripts/acft_train.py --config configs/hebrew_tiny_acft.yaml
```

Training checkpoints are saved to `outputs/<run_name>/` (e.g. `outputs/hebrew_tiny_acft/`). Training auto-resumes from the latest checkpoint if one exists.

### Full pipeline (train + convert + quantize)

```bash
uv run python scripts/pipeline.py --config configs/hebrew_tiny_acft.yaml
```

This runs training, converts the final HF checkpoint to ggml format, and produces quantized bins. Output ggml files land in `out/`.

Use `--skip-train` if you already have a trained checkpoint and only want conversion/quantization.

## Project structure

```
configs/                  Training configs (YAML)
scripts/
  acft_train.py           ACFT training script
  pipeline.py             Train → convert → quantize pipeline
  bootstrap.sh            One-command project setup
external/
  whisper.cpp/            (submodule) ggml whisper inference + converter
  whisper/                (submodule) OpenAI Whisper (used by converter)
pyproject.toml            Python dependencies
uv.lock                   Locked dependency versions
```

### Generated directories (gitignored)

- `outputs/` — training checkpoints
- `out/` — ggml binary files
- `data/` — downloaded models/datasets (HF cache)
