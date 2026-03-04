# whisper-acft-pipeline

Reproducible [ACFT (Audio-Context Fine-Tuning)](https://github.com/futo-org/whisper-acft) pipeline for Whisper models, with conversion to ggml format for use with [FUTO Keyboard](https://keyboard.futo.org/) and [whisper.cpp](https://github.com/ggml-org/whisper.cpp).

Includes training configs for Hebrew models on [Google FLEURS](https://huggingface.co/datasets/google/fleurs), but can be used with any Whisper model and language by adding a YAML config.

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

### Available configs

| Config | Base model | run_name |
|--------|-----------|----------|
| `configs/hebrew_tiny_acft.yaml` | [mike249/whisper-tiny-he-2](https://huggingface.co/mike249/whisper-tiny-he-2) | `hebrew_tiny_acft` |
| `configs/hebrew_small_acft_mike_v4.yaml` | [mike249/whisper-small-he-v4](https://huggingface.co/mike249/whisper-small-he-v4) | `hebrew_small_mike_v4` |
| `configs/hebrew_small_acft_eeizenman.yaml` | [eeizenman/whisper-small-he](https://huggingface.co/eeizenman/whisper-small-he) | `hebrew_small_eeizenman` |

### Train

```bash
uv run python scripts/acft_train.py --config configs/hebrew_tiny_acft.yaml
```

Training checkpoints are saved to `<output_dir>/<run_name>/` (as set in the YAML config). Training auto-resumes from the latest checkpoint if one exists.

### Full pipeline (train + convert + quantize)

```bash
# Tiny (mike249)
uv run python scripts/pipeline.py --config configs/hebrew_tiny_acft.yaml

# Small (mike249 v4)
uv run python scripts/pipeline.py --config configs/hebrew_small_acft_mike_v4.yaml

# Small (eeizenman)
uv run python scripts/pipeline.py --config configs/hebrew_small_acft_eeizenman.yaml
```

The pipeline reads `run_name` and `output_dir` from the config to automatically resolve checkpoint and output paths. GGML files land in `out/<run_name>/`.

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

- `outputs/` — training checkpoints (tiny config)
- `runs/` — training checkpoints (small configs)
- `out/` — ggml binary files
- `data/` — downloaded models/datasets (HF cache)
