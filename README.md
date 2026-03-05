# whisper-acft-pipeline

Reproducible [ACFT (Audio-Context Fine-Tuning)](https://github.com/futo-org/whisper-acft) pipeline for Whisper models, with conversion to ggml format for use with [FUTO Keyboard](https://keyboard.futo.org/) and [whisper.cpp](https://github.com/ggml-org/whisper.cpp).

Includes training configs for Hebrew models on [Google FLEURS](https://huggingface.co/datasets/google/fleurs) and [ivrit.ai](https://huggingface.co/datasets/ivrit-ai/whisper-training), but can be used with any Whisper model and language by adding a YAML config.

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

**Fine-tuning** (improve recognition quality):

| Config | Base model | Dataset | run_name |
|--------|-----------|---------|----------|
| `configs/hebrew_tiny_finetune.yaml` | [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) | [ivrit-ai/whisper-training](https://huggingface.co/datasets/ivrit-ai/whisper-training) (~400h) | `hebrew_tiny_ft` |

**ACFT** (optimize for short audio / FUTO Keyboard):

| Config | Base model | run_name |
|--------|-----------|----------|
| `configs/hebrew_tiny_acft.yaml` | [mike249/whisper-tiny-he-2](https://huggingface.co/mike249/whisper-tiny-he-2) | `hebrew_tiny_acft` |
| `configs/hebrew_small_acft_mike_v4.yaml` | [mike249/whisper-small-he-v4](https://huggingface.co/mike249/whisper-small-he-v4) | `hebrew_small_mike_v4` |
| `configs/hebrew_small_acft_eeizenman.yaml` | [eeizenman/whisper-small-he](https://huggingface.co/eeizenman/whisper-small-he) | `hebrew_small_eeizenman` |

### Fine-tune

```bash
uv run python scripts/finetune.py --config configs/hebrew_tiny_finetune.yaml
```

Fine-tunes a Whisper model on a speech-to-text dataset using standard supervised training with `Seq2SeqTrainer`. Supports streaming datasets (no full download needed), WER evaluation, and auto-resume from checkpoints.

The fine-tuned model is saved to `<output_dir>/<run_name>/final/`. This can then be used as the base model for ACFT.

> **Note:** The `ivrit-ai/whisper-training` dataset may be gated. Run `huggingface-cli login` and accept the dataset terms first.

### Train (ACFT)

```bash
uv run python scripts/acft_train.py --config configs/hebrew_tiny_acft.yaml
```

Training runs for 8 epochs with batch_size=1, matching the [FUTO reference implementation](https://github.com/futo-org/whisper-acft). The key technique is a partial encoder with truncated positional embeddings, which teaches the model to handle short audio without repeating.

Checkpoints are saved to `<output_dir>/<run_name>/` (as set in the YAML config). Training auto-resumes from the latest checkpoint if one exists.

Config files only need model-specific fields (run_name, base_model, dataset, device). All ACFT-critical params (batch_size, lr, num_epochs, etc.) default to FUTO-aligned values.

### Full pipeline (train + convert + quantize)

```bash
# ACFT only (existing models)
uv run python scripts/pipeline.py --config configs/hebrew_tiny_acft.yaml

# Fine-tune + ACFT + convert + quantize
uv run python scripts/pipeline.py \
  --finetune-config configs/hebrew_tiny_finetune.yaml \
  --config configs/hebrew_tiny_acft.yaml
```

When `--finetune-config` is provided, the pipeline runs fine-tuning first, then uses the fine-tuned model as the ACFT base model, followed by ggml conversion and quantization. If omitted, the pipeline works as before (ACFT only).

The pipeline reads `run_name` and `output_dir` from the config to automatically resolve checkpoint and output paths. GGML files land in `out/<run_name>/`.

Use `--skip-train` if you already have a trained checkpoint and only want conversion/quantization.

## Project structure

```
configs/                  Training configs (YAML)
scripts/
  finetune.py             Supervised fine-tuning (Seq2SeqTrainer)
  acft_train.py           ACFT training script
  pipeline.py             [Finetune →] ACFT → convert → quantize pipeline
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

## Resources

- [whisper-acft](https://github.com/futo-org/whisper-acft) — ACFT fine-tuning method by FUTO
- [android-keyboard](https://github.com/futo-org/android-keyboard) — FUTO Keyboard (uses ggml Whisper models)
