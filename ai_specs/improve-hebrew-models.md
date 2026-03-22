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

Teacher: `ivrit-ai/whisper-large-v3-turbo` (0.8B, WER 0.189).
Script: `scripts/distill.py` — done.
Configs: `configs/hebrew_base_distill.yaml`, `configs/hebrew_tiny_distill.yaml`.

### Student initialization: start from fine-tuned weights

Students initialize from our fine-tuned models, NOT from original OpenAI weights:
- Base: `outputs/hebrew_base_ft/final` (WER 0.596, HF: `amitkot/whisper-base-he`)
- Tiny: `outputs/hebrew_tiny_ft_v2/final` (WER 0.581, HF: `amitkot/whisper-tiny-he`)

**Why**: Fine-tuned models already map Hebrew phonemes → tokens. Distillation refines
these representations with richer teacher signal. Starting from scratch (WER ~1.0)
forces learning Hebrew AND matching the teacher simultaneously — harder and slower.

**Uncertainty**: This is our best guess, not established fact. Arguments for starting
from original OpenAI weights instead:
- No fine-tuning biases — learns purely from the teacher's distribution
- Fine-tuned representations may resist teacher corrections if they overfit
- DistilWhisper (HuggingFace) starts from original weights, not fine-tuned
- For well-resourced languages, original works fine (pretrained model already capable)

For Hebrew specifically, starting fine-tuned is likely better because the original tiny
is nearly useless (WER 1.004) — too far from functional for the teacher signal alone
to bridge the gap efficiently.

**If results disappoint**: try distilling from `openai/whisper-tiny` instead (one config
line change). Compare WER to determine which initialization is better for Hebrew.

### Why distill into both

Since base didn't beat tiny on direct fine-tuning, distill into both and compare.
If distilled tiny reaches WER ~0.35–0.40, it's the better choice — 2× faster than
base and good enough for keyboard dictation.

### Knowledge distillation approach: online soft-label KD

We use **soft-label distillation** (KL divergence on the teacher's full probability
distribution), not hard pseudo-labeling (teacher's best-guess transcription). Soft labels
preserve the teacher's uncertainty across the full vocabulary — richer signal than
a single transcription string.

**Loss function:**
```
loss = α × CE(student_logits, ground_truth) + (1-α) × KL(student_logits, teacher_logits) × T²
```

| Parameter | Value | Explanation |
|---|---|---|
| α (alpha) | 0.5 | Equal weight to ground truth (hard) and teacher (soft) labels |
| T (temperature) | 2.0 | Softens the teacher's distribution, exposing uncertainty over the full vocab |
| T² scaling | 4.0 | Compensates for magnitude reduction from temperature scaling |
| Padding mask | labels != -100 | KL loss computed only at real token positions |

**Online approach**: The teacher runs a forward pass on every training batch alongside
the student. This means the teacher must be loaded in memory during training — but avoids
the impractical storage cost of pre-computing soft labels (51,865 vocab × ~20 tokens ×
4 bytes × 55k examples ≈ 220GB for full logits).

**Implementation**: `DistillationTrainer` subclasses HF `Seq2SeqTrainer`, overriding only
`compute_loss`. The teacher is stored as an attribute, called under `torch.no_grad()`.
All Trainer infrastructure works unchanged: checkpointing, resume, eval with WER generation,
cosine LR, logging. The optimizer only sees student parameters.

**Cannot share teacher soft labels across runs**: Because the teacher runs live during
training (online KD), base and tiny distillation must run sequentially. Each run loads
the teacher independently. Teacher download is cached after the first run.

### What checkpoints save

HF Trainer checkpoints include:
- `model.safetensors` — student model weights
- `optimizer.pt` — AdamW state (momentum + variance per parameter)
- `scheduler.pt` — LR scheduler state (exact step)
- `trainer_state.json` — step, epoch, loss history, best metric
- `rng_state.pth` — RNG state for reproducibility

**NOT saved** (transient): gradients (discarded after optimizer step), activations
(recomputed with gradient_checkpointing), teacher model (frozen, loaded from HF on resume).

### Memory on M4 (64GB unified)

| Component | Estimated size |
|---|---|
| Teacher (turbo, 0.8B fp32, eval mode, no grad) | ~3.2GB |
| Student weights (base, 74M fp32) | ~0.3GB |
| Optimizer state (AdamW, 2× student params) | ~0.6GB |
| Gradients (student only) | ~0.3GB |
| Activations (with gradient_checkpointing) | ~2–4GB |
| MPS overhead + data pipeline | ~10–20GB |
| **Total estimate** | **~20–30GB** |

Below the ~49GB seen during base fine-tuning. If OOM: reduce
`per_device_train_batch_size` to 2, increase `gradient_accumulation_steps` to 8
(keeps effective batch size at 16).

### Language and task tokens

Labels currently: `<|startoftranscript|><|notimestamps|>text<|endoftext|>`.
Missing: `<|he|>` (Hebrew) and `<|transcribe|>` (task).

This matches standard HF Whisper fine-tuning — language/task enforced during generation
via `generation_config`, not in training labels. Both teacher and student see identical
decoder inputs, so KL comparison is fair.

**Potential improvement** (try if distillation results disappoint): add prefix tokens
by calling `tokenizer.set_prefix_tokens(language="he", task="transcribe")` before data
preparation. This would make labels match how the teacher was likely trained.

### MPS performance: offline beats online by 4×

Benchmarked on M4 Pro 48GB. The 0.8B teacher forward pass dominates online distillation
(13 of 15 seconds per step). Tested optimizations:

| Optimization | s/step | Verdict |
|---|---|---|
| Baseline (fp32 teacher, torch.no_grad) | 18s | — |
| fp16 teacher + torch.inference_mode | 15s | kept |
| + PYTORCH_MPS_FAST_MATH=1 | 15s | no change |
| + PYTORCH_MPS_PREFER_METAL=1 | 20s | **worse** |
| torch.compile | 35s | **much worse**, do not use on MPS |
| Disable gradient_checkpointing | 15s | no change (student too small) |
| batch_size=8 | 17s | worse |
| Student-only (no teacher) | 2s | **target for offline approach** |

**Conclusion**: MPS cannot run the 0.8B teacher faster than ~0.84s per example regardless
of batch size (memory bandwidth bound). The solution is to precompute teacher logits
offline, then train the student without the teacher loaded.

### Execution plan (3 steps)

**Step 1: Precompute teacher logits (~13h, runs unattended)**

```bash
uv run python scripts/precompute_teacher.py --config configs/hebrew_base_distill.yaml
```

- Runs teacher over 55k training examples, saves top-100 logits per token position
- Output: `outputs/hebrew_base_distill/teacher_logits/` (~1.6 GB)
- Resume: Ctrl+C safe, re-run same command to continue from where it stopped
- Log: `outputs/hebrew_base_distill/teacher_logits/precompute_log.jsonl`
- Monitor: `tail -1 outputs/hebrew_base_distill/teacher_logits/precompute_log.jsonl`
- Speed: ~0.84s per example, batch_size=16, fp16 teacher on MPS
- **Logits are reusable for both base and tiny distillation**

**Step 2: Distill base (~8.5h) and tiny (~5.5h)**

```bash
# Base distillation (offline mode, ~2s/step)
uv run python scripts/distill.py --config configs/hebrew_base_distill.yaml

# Tiny distillation (reuses same teacher logits)
uv run python scripts/distill.py --config configs/hebrew_tiny_distill.yaml
```

Stop/resume: Ctrl+C is safe. Re-run the same command to resume from latest checkpoint.
Do not change `max_steps` between runs.

**Step 3: Evaluate**

```bash
uv run python scripts/eval.py outputs/hebrew_base_distill/final outputs/hebrew_tiny_distill/final --samples 2000
```

**Expected WER**: base 0.30–0.40, tiny 0.35–0.45.

**Total time: ~27h** (vs ~116h for two online runs).

### Scripts

| Script | Purpose |
|---|---|
| `scripts/precompute_teacher.py` | Run teacher once, save top-K logits to disk |
| `scripts/distill.py` | Train student with precomputed logits (offline) or live teacher (online) |
| `scripts/eval.py` | Compare models on test set |

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
