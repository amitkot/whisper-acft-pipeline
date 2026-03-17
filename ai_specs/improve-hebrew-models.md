# Improving Hebrew Whisper Models

Comprehensive guide for improving WER on Hebrew ASR for keyboard dictation.
Supersedes `improve-hebrew-tiny-model.md`.

---

## Current state

| Model | Params | Best WER | Step | Config |
|---|---|---|---|---|
| whisper-tiny v1 | 39M | 0.636 | 3000 | `hebrew_tiny_ft` — linear LR, resume artifact |
| whisper-tiny v2 | 39M | 0.557 | 8500 | `hebrew_tiny_ft_v2` — cosine, 3 epochs, clean run |
| whisper-small | 244M | 0.368 | 4000 | `hebrew_small_ft` — only ~1 epoch seen |
| whisper-base | 74M | not yet trained | — | `hebrew_base_ft` — next step |

All WER numbers: `ivrit-ai/whisper-training` test split, 200 samples, `jiwer.wer()`,
**no normalization**. Numbers are comparable to each other but not to published benchmarks
(which use OpenAI's text normalizer and cleaner datasets like LibriSpeech).

---

## Governing principle: match training to inference

Source: [ivrit.ai — "Fine Tune Whisper the Right Way"](https://www.ivrit.ai/en/2025/02/13/training-whisper/)

ivrit.ai's core lesson is that training distribution must match inference format.
Their warning about timestamps and long-form conditioning is **irrelevant for keyboard
dictation** — it only matters when stitching 30-second chunks for long audio.

For keyboard dictation:
- Users speak one sentence or short phrase at a time
- No 30-second boundaries, no timestamp tokens, no previous-text conditioning needed
- **Short supervised clips (~5s) are the ideal training format**
- What ivrit.ai considered a limitation of their short-clip dataset is actually a feature for us

See `ai_specs/datasets.md` for full dataset analysis.

---

## Option 1: Fine-tune a larger model in the tiny→small range

### whisper-base (74M params)

The only official model between tiny (39M) and small (244M).
Config: `configs/hebrew_base_ft.yaml` — see `configs/hebrew_base_ft.md` for rationale.

**Process**: Identical to tiny fine-tuning, just `base_model: openai/whisper-base`.
No code changes needed.

**Expected WER**: ~0.47–0.52 (interpolating between tiny=0.557 and small=0.368,
weighted by capacity ratio).

**Training time**: ~4–5 hours on M4 MPS (~2× tiny, ~0.5× small).

**Recommendation**: Do this first. Lowest effort, likely meaningful gain.

```bash
uv run python scripts/finetune.py --config configs/hebrew_base_ft.yaml
```

---

## Option 2: Continue training small model

`hebrew_small_ft` only saw ~1 epoch of data (stopped at step 4000, best WER 0.368).
The model is undertrained — the same pattern as tiny v1.

**Process**: Create `hebrew_small_ft_v2.yaml` following the same pattern as
`hebrew_tiny_ft_v2.yaml`:
- `base_model: openai/whisper-small`
- `lr_scheduler_type: cosine`
- `max_steps: 15000` (~3 epochs)
- `learning_rate: 6.0e-6` (slightly lower than tiny; larger model is more sensitive)
- `save_total_limit: 2`, `save_steps: 250`, `eval_steps: 250`

**Expected WER**: Potentially 0.30–0.34 (small has more capacity; 3 clean epochs
could push well below current 0.368).

**Training time**: ~16–18 hours on M4 MPS.

---

## Option 3: Add more datasets (multi-dataset training)

### Why it helps

More diverse short Hebrew speech teaches the model acoustic patterns not present in
`ivrit-ai/whisper-training` alone (different speakers, mics, accents, speaking styles).
For dictation, **speaker and mic diversity matters more than total hours**.

### Best first addition: crowd-transcribe-v5

- ~300h, not gated, diverse speakers
- Filter out `noisy=True` and `multiple_speakers=True` examples
- Expected gain: reduces OOV errors on speaker/acoustic variety; WER improvement ~5–10%

### Implementation needed in finetune.py

```python
from datasets import interleave_datasets

combined = interleave_datasets(
    [ds_primary, ds_secondary],
    probabilities=[0.7, 0.3],  # weight cleaner data higher
    seed=cfg.seed,
)
```

The config needs new fields: `dataset_name_2`, `dataset_config_2`, `text_column_2`,
`audio_column_2`, `dataset_weight` (primary proportion).

### Dataset priority for dictation

1. `ivrit-ai/crowd-transcribe-v5` — short clips, diverse, not gated (**do first**)
2. `fsicoli/common_voice_22_0` (he) — community, mobile-like audio
3. `ivrit-ai/crowd-recital-whisper-training` — clean read speech (lower priority)
4. Skip `knesset-plenums` — parliamentary long-form is wrong domain

See `ai_specs/datasets.md` for full details on each.

---

## Option 4: Pseudo-labeling from whisper-small-he

### What it is

Use the stronger model (small, WER 0.368) to transcribe unlabeled Hebrew audio, then
train tiny/base on those transcriptions as supervised data. This is the core of
[DistilWhisper](https://github.com/huggingface/distil-whisper), which achieved
~2–5% absolute WER improvement using this technique.

### Why it works

The small model has seen more of the Hebrew distribution than tiny has "capacity" to
represent directly. Its transcriptions of unlabeled audio encode knowledge that tiny
can absorb via supervised training, without tiny needing to learn it from scratch.

### Process

**Step 1 — Generate pseudo-labels**
```python
# Run whisper-small-he over unlabeled audio (ivrit-ai/audio-v2, ~800h)
from transformers import pipeline
pipe = pipeline("automatic-speech-recognition", model="amitkot/whisper-small-he")
result = pipe(audio_array)  # → {"text": "..."}
```

**Step 2 — Filter for quality**
Discard clips where the output is unreliable:
- Output contains `[BLANK_AUDIO]` or repetition loops
- Output is very short relative to audio length (model failed to transcribe)
- Output has abnormal token repetition (common failure mode)

**Step 3 — Build a training dataset**
Create an HF dataset of `(audio, pseudo_transcript)` pairs and upload or save locally.

**Step 4 — Fine-tune tiny/base on combined data**
Mix real supervised data (weight 70%) + pseudo-labeled (weight 30%).
Real supervised data is ground truth; pseudo-labeled data expands coverage.

### Risk

**Tiny and small models learn noise more readily than large models.**
A bad pseudo-label doesn't "average out" — it trains the model toward the wrong output.
Strict filtering in Step 2 is essential. If filtering is too loose, pseudo-labeling
can make WER *worse*.

### Expected gain

~5–15% relative WER improvement over direct fine-tuning on same data, based on
DistilWhisper results on English. Hebrew may differ.

### Implementation complexity

Medium — needs a separate pseudo-label generation script, then reuses existing
`finetune.py` with the generated dataset.

---

## Option 5: Soft-label distillation

### What it is

Instead of hard pseudo-labels (the small model's best guess), train tiny/base to match
the **full probability distribution** of small's output at every decoder step.

### Loss function

```
total_loss = α × CrossEntropy(student_logits, ground_truth_tokens)
           + (1−α) × KL_divergence(student_logits, teacher_logits)
```

α = 0.5 is a common default. The KL term pulls the student toward the teacher's
uncertainty profile, not just its best guess.

### Why it's better than pseudo-labeling

If small is uncertain between `"הוא"` and `"הם"` (he/they, commonly confused), the
soft label preserves that uncertainty. A hard pseudo-label just picks one and treats
it as fact, potentially misleading tiny.

### Why it's harder

- Both models must be in memory simultaneously (small frozen as teacher, tiny/base training)
  — ~300MB total, fine on M4 but requires code changes
- Cannot use vanilla `Seq2SeqTrainer` — need a custom training loop or subclass
- `α` tuning adds another hyperparameter

### Expected gain vs pseudo-labeling

Marginal in practice for this task. The simpler pseudo-labeling approach is a better
first step. Do Option 4 before considering this.

---

## Recommended execution order

1. **`hebrew_base_ft`** — fine-tune whisper-base, no new code, ~5h *(next step)*
2. **`hebrew_small_ft_v2`** — clean 3-epoch run of small, ~17h
3. **Add crowd-transcribe-v5** to tiny v2 and base — needs multi-dataset loader in finetune.py
4. **Pseudo-label ivrit-ai/audio-v2** using whisper-small-he, train tiny/base on combined data
5. **Soft-label distillation** — only if pseudo-labeling results are promising and the effort is justified

---

## WER comparison notes

All our WER numbers are **unnormalized** (`jiwer.wer()` on raw decoded strings).
Published English benchmarks use OpenAI's text normalizer (removes punctuation,
standardizes numbers, contractions, etc.) — those numbers are **not comparable to ours**.

Our numbers ARE comparable to each other (same eval set, same metric).

If we want to publish comparable results or benchmark against other Hebrew models,
we should add Whisper-style normalization to the eval loop.
