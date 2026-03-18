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

**Why**: Both tiny and base start from poor Hebrew pre-training (untuned WER ~1.0 and ~0.85).
Fine-tuning has to teach them Hebrew nearly from scratch. In that regime, base's extra
parameters don't help — there isn't enough data to fill the capacity. They converge to
the same ceiling.

**Corrected understanding**: The "data per parameter" argument doesn't explain why small
(244M, worst ratio) beats both. The real differentiator is pre-training quality, not
parameter count during fine-tuning. See Finding 2.

### Finding 2: Pre-training threshold at ~200M params

Small (untuned WER 0.503) already speaks Hebrew before fine-tuning. Tiny and base (~1.0
and ~0.85) essentially don't. At ~200M+ params on 680k hours, Whisper absorbed enough
Hebrew during pre-training to be useful. Below that threshold, Hebrew knowledge is weak.

Fine-tuning small is *adapting* a capable model. Fine-tuning tiny/base is *teaching* Hebrew
from scratch. Adaptation is more data-efficient — this is why small outperforms despite
having the worst data-per-parameter ratio.

### Finding 3: Base is worth pursuing — for inference speed, via distillation

Base runs ~3× faster than small at inference — preferable for keyboard dictation latency.
But direct fine-tuning hit a ceiling at WER 0.596.

**The path to better base WER is distillation**, not more fine-tuning on the same data.
The best available teacher is not our own small model — it's ivrit-ai's Large v3 (see below).

### Finding 4: Small is heavily undertrained

`hebrew_small_ft` saw only ~1 epoch with a resume artifact and linear LR schedule.
Small's true ceiling with 3 clean epochs is likely 0.28–0.32.

---

## Available teacher models for distillation

Source: [ivrit-ai HuggingFace org](https://huggingface.co/ivrit-ai), [Zonos Hebrew issue](https://github.com/Zyphra/Zonos/issues/38)

ivrit-ai trained Whisper models on **5,050 hours** of Hebrew (12× our 400h dataset):
- ~4,700h knesset-plenums (Israeli parliament)
- ~300h crowd-transcribe-v5
- ~50h crowd-recital-whisper-training

| Model | Params | Format | WER (estimated) | Use as teacher? |
|---|---|---|---|---|
| `ivrit-ai/whisper-large-v3` | 2B | HF PyTorch | ~0.05–0.10? | **Best teacher** — highest quality |
| `ivrit-ai/whisper-large-v3-turbo` | 0.8B | HF PyTorch | ~0.08–0.12? | **Good teacher** — lighter, faster |
| `ivrit-ai/whisper-large-v3-ct2` | 2B | CTranslate2 | same | No — can't extract logits |
| `ivrit-ai/whisper-large-v3-turbo-ct2` | 0.8B | CTranslate2 | same | No — wrong format |
| `amitkot/whisper-small-he` | 244M | HF PyTorch | 0.367 | Weaker teacher, but ours |

**WER estimates pending** — running `scripts/eval.py` on ivrit-ai models to get actual
numbers on our eval set. Published WER is in their Interspeech 2025 paper.

**Key insight**: Using ivrit-ai's large-v3 as teacher instead of our small model
dramatically raises the distillation ceiling. A WER ~0.05 teacher could push base
to ~0.25–0.35, potentially matching or beating our fine-tuned small.

**Memory on M4 (64GB unified)**:
- large-v3 (2B, fp32): ~8GB — fits comfortably
- large-v3-turbo (0.8B, fp32): ~3GB — easy
- Teacher + student (base, 74M): total ~8.3–11.3GB — no issues

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

**Status: not yet done. Still valuable even with ivrit-ai teachers available.**

A clean 3-epoch cosine run of small establishes the true ceiling for a model we
fully control and can deploy without external dependencies.

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
**Expected WER**: 0.28–0.32.

---

## Option 2: Distillation from ivrit-ai → base

**This is now the most promising path to a fast, accurate model.**

Using `ivrit-ai/whisper-large-v3` or `ivrit-ai/whisper-large-v3-turbo` as the teacher
instead of our own small model. Their models were trained on 5,050h of Hebrew and
are dramatically stronger than anything we've fine-tuned.

### Does this require more data?

**No.** Distillation works on the same `ivrit-ai/whisper-training` dataset. The teacher
model generates richer training signal (probability distributions) from the same audio.

### Soft-label distillation process

```
loss = α × CrossEntropy(student_logits, ground_truth_tokens)
     + (1−α) × KL_divergence(student_logits, teacher_logits)
```

α = 0.5 is a sensible default.

**Step 1**: Evaluate ivrit-ai teacher models on our eval set to confirm WER.

**Step 2**: Write `scripts/distill.py`:
- Load teacher (`ivrit-ai/whisper-large-v3-turbo`, 0.8B) frozen
- Load student (base, 74M) — initialize fresh from `openai/whisper-base`
- For each batch: run teacher in `torch.no_grad()`, get logits
- Train student with combined CE + KL loss
- Cosine LR schedule, same as other training runs

**Step 3**: Evaluate with `scripts/eval.py --samples 2000`.

**Implementation notes**:
- Teacher (0.8B) + student (74M) ≈ 3.3GB total fp32 — easily fits M4
- If using large-v3 (2B) instead of turbo: ~8.3GB total — still fine
- Cannot use `Seq2SeqTrainer` directly — needs custom training loop
- Consider using turbo as teacher first (faster, simpler), then large-v3 if needed

**Expected WER**: If teacher WER is ~0.05–0.10, base could reach 0.25–0.40.
This would make base competitive with or better than our fine-tuned small (0.367)
at 3× the inference speed.

### Pseudo-label distillation (simpler alternative)

Run ivrit-ai teacher over the training audio, use its transcriptions as hard labels
for base training. No custom training loop needed — reuses `finetune.py` directly.
Less powerful than soft-label but much simpler to implement.

Could also pseudo-label `ivrit-ai/audio-v2` (~800h unlabeled) to expand the
training dataset significantly.

---

## Option 3: More training data

**Status: not yet done. Complements distillation.**

More diverse short Hebrew speech improves robustness across speakers, mics, and accents.
For dictation, speaker and mic diversity matter more than total hours.

**Best first addition**: `ivrit-ai/crowd-transcribe-v5` (~300h, not gated, diverse speakers).
Filter `noisy=True` and `multiple_speakers=True` before use.

**Implementation needed**: Multi-dataset interleaving loader in `finetune.py`:
```python
from datasets import interleave_datasets
combined = interleave_datasets([ds_primary, ds_secondary], probabilities=[0.7, 0.3])
```

See `ai_specs/datasets.md` for full dataset details and priority ordering.

---

## Recommended execution order

| Priority | Task | Time | Prerequisite |
|---|---|---|---|
| 1 | Eval ivrit-ai teacher models on our test set | ~30min | none |
| 2 | `hebrew_small_ft_v2` — clean 3-epoch small run | ~17h | none |
| 3 | Write `scripts/distill.py`, distill ivrit-ai-turbo→base | ~10h + dev | eval confirms teacher quality |
| 4 | Eval distilled base vs fine-tuned small | ~10min | distillation done |
| 5 | Add crowd-transcribe-v5 multi-dataset support | dev + ~5h | none |
| 6 | Pseudo-label ivrit-ai/audio-v2 with large-v3 | dev + hours | teacher model confirmed |

Steps 1 and 2 can run in parallel. Step 1 is already running.

---

## WER comparison notes

- All WER numbers are **unnormalized** — not comparable to published English benchmarks
- **Use `scripts/eval.py --samples 2000`** for reliable comparisons; training-time eval (200 samples) is noisy
- WER > 1.0 is possible: it means the model outputs more words than the reference (insertions)
- To compare against other Hebrew models, add Whisper-style text normalization to eval loop
- ivrit-ai published benchmarks in Interspeech 2025 paper (may use different normalization)
