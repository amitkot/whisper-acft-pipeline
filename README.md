# whisper-acft-pipeline

Reproducible pipeline for fine-tuning, distilling, and converting Whisper models for Hebrew speech recognition. Designed for keyboard dictation on [FUTO Keyboard](https://keyboard.futo.org/) via [whisper.cpp](https://github.com/ggml-org/whisper.cpp).

## Hebrew Model Results

All WER measured on `ivrit-ai/whisper-training` test split, 2000 samples, `jiwer.wer()`, no normalization.

### Fine-tuned models

| Model | Params | WER (untuned) | WER (fine-tuned) | HuggingFace |
|-------|--------|:-------------:|:----------------:|-------------|
| whisper-tiny | 39M | 1.004 | **0.581** | [amitkot/whisper-tiny-he](https://huggingface.co/amitkot/whisper-tiny-he) |
| whisper-base | 74M | 0.851 | **0.596** | [amitkot/whisper-base-he](https://huggingface.co/amitkot/whisper-base-he) |
| whisper-small | 244M | 0.503 | **0.367** | [amitkot/whisper-small-he](https://huggingface.co/amitkot/whisper-small-he) |

### ACFT models (optimized for short audio / FUTO Keyboard)

| Model | Base | HuggingFace |
|-------|------|-------------|
| whisper-tiny-he-acft | amitkot/whisper-tiny-he | [amitkot/whisper-tiny-he-acft](https://huggingface.co/amitkot/whisper-tiny-he-acft) |
| whisper-small-he-acft | amitkot/whisper-small-he | [amitkot/whisper-small-he-acft](https://huggingface.co/amitkot/whisper-small-he-acft) |

### Teacher models (for distillation)

| Model | Params | WER | Source |
|-------|--------|:---:|--------|
| whisper-large-v3-turbo | 0.8B | **0.189** | [ivrit-ai/whisper-large-v3-turbo](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo) |
| whisper-large-v3 | 2B | **0.186** | [ivrit-ai/whisper-large-v3](https://huggingface.co/ivrit-ai/whisper-large-v3) |

### In progress: distillation

Distilling `ivrit-ai/whisper-large-v3-turbo` (WER 0.189) into whisper-tiny and whisper-base
to achieve better WER at fast inference speed. See [ai_specs/improve-hebrew-models.md](ai_specs/improve-hebrew-models.md) for the full roadmap.

## Setup

```bash
git clone --recurse-submodules https://github.com/amitkot/whisper-acft-pipeline.git
cd whisper-acft-pipeline
./scripts/bootstrap.sh
```

This initializes submodules, builds whisper.cpp, and installs Python dependencies via `uv sync`.

> **Note:** The `ivrit-ai/whisper-training` dataset is gated. Run `huggingface-cli login` and accept the dataset terms first.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/finetune.py` | Supervised fine-tuning with `Seq2SeqTrainer`. Streaming datasets, WER eval, auto-resume. |
| `scripts/distill.py` | Knowledge distillation (online or offline with precomputed logits). |
| `scripts/precompute_teacher.py` | Precompute teacher top-K logits for offline distillation. |
| `scripts/acft_train.py` | ACFT training (FUTO-aligned: partial encoder, truncated positional embeddings). |
| `scripts/pipeline.py` | End-to-end: [finetune →] ACFT → ggml convert → quantize. |
| `scripts/eval.py` | Evaluate one or more models, report WER table. |

## Usage

### Fine-tune

```bash
uv run python scripts/finetune.py --config configs/hebrew_tiny_ft_v2.yaml
```

### Distill (with precomputed teacher logits)

```bash
# Step 1: Precompute teacher logits (~13h, one-time, reusable for all students)
uv run python scripts/precompute_teacher.py --config configs/hebrew_base_distill.yaml

# Step 2: Distill into student (~6-9h per student)
uv run python scripts/distill.py --config configs/hebrew_tiny_distill.yaml
uv run python scripts/distill.py --config configs/hebrew_base_distill.yaml
```

### ACFT + convert + quantize

```bash
uv run python scripts/pipeline.py \
  --finetune-config configs/hebrew_tiny_ft_v2.yaml \
  --config configs/hebrew_tiny_acft.yaml
```

### Evaluate

```bash
uv run python scripts/eval.py \
    outputs/hebrew_tiny_ft_v2/final \
    outputs/hebrew_base_ft/final \
    --samples 2000
```

## Configs

Each YAML config has a companion `.md` file explaining all parameter decisions.

| Config | Model | Status |
|--------|-------|--------|
| `hebrew_tiny_ft_v2.yaml` | whisper-tiny fine-tune | done (WER 0.581) |
| `hebrew_base_ft.yaml` | whisper-base fine-tune | done (WER 0.596) |
| `hebrew_small_finetune.yaml` | whisper-small fine-tune | done (WER 0.367) |
| `hebrew_tiny_distill.yaml` | distill turbo → tiny | in progress |
| `hebrew_base_distill.yaml` | distill turbo → base | in progress |
| `hebrew_tiny_acft.yaml` | ACFT on tiny | done |
| `hebrew_small_acft.yaml` | ACFT on small | done |

## Project structure

```
configs/                  Training configs (YAML) + companion docs (.md)
ai_specs/                 Research notes, dataset analysis, improvement roadmap
scripts/
  finetune.py             Supervised fine-tuning (Seq2SeqTrainer)
  distill.py              Knowledge distillation (online + offline modes)
  precompute_teacher.py   Precompute teacher logits for offline distillation
  acft_train.py           ACFT training script
  pipeline.py             [Finetune →] ACFT → convert → quantize pipeline
  eval.py                 Multi-model WER evaluation
  bootstrap.sh            One-command project setup
external/
  whisper.cpp/            (submodule) ggml whisper inference + converter
  whisper/                (submodule) OpenAI Whisper (used by converter)
```

## Resources

- [whisper-acft](https://github.com/futo-org/whisper-acft) — ACFT method by FUTO
- [android-keyboard](https://github.com/futo-org/android-keyboard) — FUTO Keyboard
- [ivrit.ai](https://www.ivrit.ai/) — Hebrew speech datasets and models
- [ivrit.ai blog: Fine Tune Whisper the Right Way](https://www.ivrit.ai/en/2025/02/13/training-whisper/)
