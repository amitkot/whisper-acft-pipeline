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
| `ivrit-ai/whisper-large-v3` | 2B | HF PyTorch | **0.186** | Best quality (marginally) |
| `ivrit-ai/whisper-large-v3-turbo` | 0.8B | HF PyTorch | **0.189** | **Practical teacher choice** — nearly same WER, 60% less compute |
| `ivrit-ai/whisper-large-v3-ct2` | 2B | CTranslate2 | same | No — can't extract logits |
| `ivrit-ai/whisper-large-v3-turbo-ct2` | 0.8B | CTranslate2 | same | No — wrong format |
| `amitkot/whisper-small-he` | 244M | HF PyTorch | 0.367 | Weaker teacher, but ours |

**Key insight**: ivrit-ai turbo (WER 0.189) is nearly 2× better than our small (0.367).
Using it as teacher dramatically raises the distillation ceiling. A WER 0.189 teacher
could push base to ~0.30–0.40, competitive with our fine-tuned small at 3× inference speed.

Turbo vs large-v3: virtually identical WER (0.189 vs 0.186) but turbo runs in half the time
and uses ~3GB vs ~8GB. **Use turbo as teacher.**

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

## Option 1: Distillation from ivrit-ai turbo → base and tiny

**This is the primary path forward.**

Using `ivrit-ai/whisper-large-v3-turbo` (0.8B, WER 0.189) as teacher.

### Distill into both base AND tiny

Since base didn't beat tiny on direct fine-tuning, distill into both and compare.
If distilled tiny reaches WER ~0.35–0.40, it may be the better choice — faster than
base and "good enough" for keyboard dictation.

### Does this require more data?

**No.** Distillation works on the same `ivrit-ai/whisper-training` dataset. The teacher
model generates richer training signal (probability distributions) from the same audio.

### Soft-label distillation process

```
loss = α × CrossEntropy(student_logits, ground_truth_tokens)
     + (1−α) × KL_divergence(student_logits, teacher_logits)
```

α = 0.5 is a sensible default.

**Step 1**: ~~Evaluate ivrit-ai teacher models~~ — done (turbo=0.189, large-v3=0.186).

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

**Expected WER**: With teacher WER 0.189, base could reach 0.30–0.40.
This would make base competitive with our fine-tuned small (0.367)
at 3× the inference speed.

### Pseudo-label distillation (simpler alternative)

Run ivrit-ai teacher over the training audio, use its transcriptions as hard labels
for base training. No custom training loop needed — reuses `finetune.py` directly.
Less powerful than soft-label but much simpler to implement.

Could also pseudo-label `ivrit-ai/audio-v2` (~800h unlabeled) to expand the
training dataset significantly.

---

## Option 2: More training data

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

## Fallback: Clean run of small — hebrew_small_ft_v2

If distillation doesn't yield good results, fall back to a clean 3-epoch cosine run
of small. Config ready at `configs/hebrew_small_ft_v2.yaml`.

**Training time**: ~16–18 hours on M4 MPS.
**Expected WER**: 0.28–0.32.

Small is too slow for keyboard dictation (~6× slower than tiny at inference), so this
would be a stepping stone — either as a deployment model with latency trade-off, or
as our own teacher for a second distillation attempt.

---

## Recommended execution order

| Priority | Task | Time | Prerequisite |
|---|---|---|---|
| 1 | Write `scripts/distill.py` | dev | none |
| 2 | Distill turbo → base | ~10h | distill.py |
| 3 | Distill turbo → tiny | ~5h | distill.py |
| 4 | Eval distilled models | ~10min | distillation done |
| 5 | ACFT on best distilled model | ~2h | eval shows good WER |
| 6 | Add crowd-transcribe-v5 | dev + training | only if distillation alone isn't enough |

---

## WER comparison notes

- All WER numbers are **unnormalized** — not comparable to published English benchmarks
- **Use `scripts/eval.py --samples 2000`** for reliable comparisons; training-time eval (200 samples) is noisy
- WER > 1.0 is possible: it means the model outputs more words than the reference (insertions)
- To compare against other Hebrew models, add Whisper-style text normalization to eval loop
- ivrit-ai published benchmarks in Interspeech 2025 paper (may use different normalization)
