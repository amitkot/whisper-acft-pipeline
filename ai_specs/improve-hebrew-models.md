# Improving Hebrew Whisper Models

Comprehensive guide for improving WER on Hebrew ASR for keyboard dictation.
Supersedes `improve-hebrew-tiny-model.md`.

---

## Current state (as of 2026-03-18)

### Reliable WER numbers (2000-sample eval via `scripts/eval.py`)

| Model | Params | WER (untuned) | WER (fine-tuned) | Config | Notes |
|---|---|---|---|---|---|
| whisper-tiny | 39M | 1.004 | 0.581 | `hebrew_tiny_ft_v2` | 3 clean epochs, cosine |
| whisper-base | 74M | 0.851 | 0.596 | `hebrew_base_ft` | 3 clean epochs, cosine |
| whisper-small | 244M | 0.503 | 0.367 | `hebrew_small_ft` | only ~1 epoch, undertrained |

WER numbers: `ivrit-ai/whisper-training` test split, 2000 samples, `jiwer.wer()`, **no normalization**.
Comparable to each other but not to published benchmarks (which use OpenAI's text normalizer).

**Eval command:**
```bash
uv run python scripts/eval.py \
    openai/whisper-tiny openai/whisper-base openai/whisper-small \
    outputs/hebrew_tiny_ft_v2/final outputs/hebrew_base_ft/final outputs/hebrew_small_ft/final \
    --samples 2000
```

### Earlier training-time WER (200 samples — treat as noisy)

Training logs report WER every `eval_steps` on 200 samples. These numbers showed
base (0.541) beating tiny (0.557), which turned out to be noise.
**Always use `scripts/eval.py` with ≥1000 samples for reliable comparisons.**

---

## Key findings from experimentation

### Finding 1: Base does not beat tiny

Base (74M, 0.596) is slightly *worse* than tiny (39M, 0.581) on the proper 2000-sample eval.
The apparent advantage seen during training (0.541 vs 0.557) was 200-sample noise.

**Why**: Both tiny and base start from a poor Hebrew pre-training (untuned WER ~1.0 and ~0.85).
Fine-tuning has to teach them Hebrew nearly from scratch on the same data. In that regime,
base's extra parameters are a liability — more parameters, same data = more underdetermined.
They converge to essentially the same ceiling.

### Finding 2: Small has a qualitatively different starting point

Small (untuned WER 0.503) already speaks Hebrew before any fine-tuning. Tiny and base
(~1.0 and ~0.85) essentially don't. This reflects a pre-training threshold — at ~200M+
params trained on 680k hours, Whisper absorbs enough Hebrew to be genuinely useful as a
starting point. Below that threshold, Hebrew knowledge from pre-training is weak.

**Implication**: Fine-tuning small is *adapting* an already-capable model. Fine-tuning
tiny/base is *teaching* a model Hebrew from scratch. Adaptation is more data-efficient.
This is why small outperforms despite having the worst data-per-parameter ratio.

### Finding 3: Base is still worth pursuing — for inference speed

For keyboard dictation, inference latency matters. Base runs ~3× faster than small on device.
If base can be brought to competitive WER, it is preferable to small for the use case.

The path to better base WER is **distillation from small**, not direct fine-tuning.
See Option 3 below.

### Finding 4: Small is heavily undertrained

`hebrew_small_ft` saw only ~1 epoch with a resume artifact and linear LR schedule —
the same situation as tiny v1 (which improved from 0.636 to 0.581 with a clean run).
Small's true ceiling with 3 clean epochs is likely 0.28–0.32.

---

## Governing principle: match training to inference

Source: [ivrit.ai — "Fine Tune Whisper the Right Way"](https://www.ivrit.ai/en/2025/02/13/training-whisper/)

For keyboard dictation:
- Users speak one sentence or short phrase at a time (~5s clips)
- No 30-second boundaries, no timestamp tokens, no previous-text conditioning needed
- **Short supervised clips are the ideal training format**
- `ivrit-ai/whisper-training` is perfectly aligned with this use case

See `ai_specs/datasets.md` for full dataset analysis.

---

## Option 1: Clean run of small — hebrew_small_ft_v2

**Status: not yet done. Highest priority.**

`hebrew_small_ft` only saw ~1 epoch. A clean 3-epoch cosine run should yield WER ~0.28–0.32,
establishing the true ceiling for small and providing a strong teacher for distillation.

**Config to create** (`configs/hebrew_small_ft_v2.yaml`):
```yaml
run_name: hebrew_small_ft_v2
base_model: openai/whisper-small
output_dir: outputs
language: he
dataset_name: ivrit-ai/whisper-training
text_column: text
audio_column: audio
eval_split: test
streaming: true
per_device_train_batch_size: 4
per_device_eval_batch_size: 1
gradient_accumulation_steps: 4
max_steps: 15000
learning_rate: 6.0e-6
lr_scheduler_type: cosine
warmup_steps: 500
save_steps: 250
save_total_limit: 2
eval_steps: 250
device: mps
fp16: false
seed: 42
```

**Training time**: ~16–18 hours on M4 MPS.

**Why this matters beyond just WER**: The resulting `hebrew_small_ft_v2/final` model
becomes the teacher for distilling into base (Option 3).

---

## Option 2: Distillation from small → base

**This is the most promising path to a fast, accurate model.**

Base (74M) is ~3× faster than small at inference — ideal for keyboard dictation.
Direct fine-tuning hit a ceiling at WER 0.596. Distillation from small can transfer
small's Hebrew knowledge into base without requiring more data.

### Does this require more data?

**No.** Distillation works on the same `ivrit-ai/whisper-training` dataset.
The small model acts as a teacher, generating richer training signal from the same audio.

### Soft-label distillation process

Train base to match small's full output probability distribution at every decoder step,
combined with the ground truth labels:

```
loss = α × CrossEntropy(base_logits, ground_truth_tokens)
     + (1−α) × KL_divergence(base_logits, small_logits)
```

α = 0.5 is a sensible default.

**Step 1**: Fine-tune `hebrew_small_ft_v2` to completion (Option 1).

**Step 2**: Write a distillation training script (`scripts/distill.py`):
- Load small model frozen as teacher
- Load base model as student (initialize from `outputs/hebrew_base_ft/final` or fresh)
- For each batch: run teacher in `torch.no_grad()`, compute KL loss against student
- Train student with combined CE + KL loss

**Step 3**: Run on same dataset, same cosine schedule.

**Implementation notes**:
- Both models in memory simultaneously: small (~900MB fp32) + base (~280MB) = ~1.2GB total, fine on M4
- Cannot use `Seq2SeqTrainer` directly — needs a custom training loop or subclass
- α is an additional hyperparameter; start with 0.5

**Expected WER**: Potentially 0.45–0.52. Distillation from a teacher with WER 0.28–0.30
(after small_ft_v2) could meaningfully improve over base's direct fine-tune ceiling of 0.596.

### Pseudo-label distillation (simpler alternative)

Run small over the training audio, use its transcriptions as hard labels for base.
No custom training loop needed — reuses `finetune.py` directly.
Less powerful than soft-label distillation but much simpler to implement.

---

## Option 3: More training data

**Status: not yet done. Do after Options 1 and 2.**

If the dataset is the bottleneck for tiny/base, more diverse Hebrew speech could push
them below their current WER ceiling. For dictation, speaker and mic diversity matter
more than total hours.

**Best first addition**: `ivrit-ai/crowd-transcribe-v5` (~300h, not gated, diverse speakers).
Filter `noisy=True` and `multiple_speakers=True` before use.

**Implementation needed**: Multi-dataset interleaving loader in `finetune.py`:
```python
from datasets import interleave_datasets
combined = interleave_datasets([ds_primary, ds_secondary], probabilities=[0.7, 0.3])
```

See `ai_specs/datasets.md` for full dataset details and priority ordering.

---

## Option 4: Pseudo-labeling with unlabeled audio

Use small to transcribe `ivrit-ai/audio-v2` (~800h unlabeled), then train
tiny/base on those transcriptions. Expands training data volume significantly.

Requires strict quality filtering of small's outputs before use as training labels.
See previous version of this doc for detailed process steps.

**Do after Options 1–3** — needs a good teacher model and a validated training pipeline first.

---

## Recommended execution order

| Priority | Task | Time | Prerequisite |
|---|---|---|---|
| 1 | `hebrew_small_ft_v2` — clean 3-epoch small run | ~17h | none |
| 2 | Eval small_ft_v2 with `scripts/eval.py` | ~10min | small_ft_v2 done |
| 3 | Write `scripts/distill.py`, distill small→base | ~10h + dev | small_ft_v2 done |
| 4 | Add crowd-transcribe-v5 to finetune.py | dev + ~5h | none |
| 5 | Pseudo-label ivrit-ai/audio-v2 with small | dev + hours | small_ft_v2 done |

---

## WER comparison notes

- All WER numbers are **unnormalized** — not comparable to published English benchmarks
- **Use `scripts/eval.py --samples 2000`** for reliable comparisons; training-time eval (200 samples) is noisy
- WER > 1.0 is possible: it means the model outputs more words than the reference (insertions)
- To compare against other Hebrew models, add Whisper-style text normalization to eval loop
