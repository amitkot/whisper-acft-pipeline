# hebrew_base_ft — Parameter Decisions

Companion notes for `hebrew_base_ft.yaml`.

---

## Why whisper-base?

`openai/whisper-base` (74M params) is the only official Whisper model between
tiny (39M) and small (244M). It represents a natural middle point:

- ~2× the capacity of tiny, likely ~12–15% relative WER improvement
- ~0.3× the capacity of small, likely ~25–30% worse than small
- Training time: ~4–5 hours on M4 MPS (~2× tiny, ~0.5× small)

Expected WER: ~0.47–0.52, interpolating between tiny v2 (0.557) and small (0.368).
This is a new data point — no published Hebrew WER for base exists.

---

## Parameter decisions

### `learning_rate: 6.0e-6`

Slightly lower than tiny v2's 8.0e-6. Larger models tend to be more sensitive to
high learning rates — their weight updates affect more complex learned representations.
6e-6 with cosine is a conservative safe choice. If WER is slow to improve early,
consider raising to 8.0e-6 in a follow-up run.

### `lr_scheduler_type: cosine`

Same reasoning as `hebrew_tiny_ft_v2` — cosine stays near peak LR longer, decays
gracefully, and is more forgiving on resume than linear. See `hebrew_tiny_ft_v2.md`
for the full explanation.

### `max_steps: 15000`

Same as tiny v2. At batch_size=16 and ~55k examples, 1 epoch ≈ 5,000 steps.
15,000 steps ≈ 3 epochs. The tiny v2 run showed the model peaked at step 8,500
(epoch 2.1) and plateaued/slightly overfit through epoch 3. Base has more capacity
so may benefit from the full 3 epochs.

**Do not change max_steps on resume** — it recalculates the LR schedule and causes
a jump. See `hebrew_tiny_ft_v2.md` for details.

### `warmup_steps: 500`

Standard ~3% of max_steps warmup. Same as tiny v2.

### `save_steps: 250` / `eval_steps: 250`

Both set to 250 (required to be equal for `load_best_model_at_end=True`).
At ~4s/step for base, 250 steps ≈ ~17 minutes between checkpoints.

### `save_total_limit: 2`

Keeps two most recent checkpoints as a safety net against corrupt mid-write saves.

### `per_device_eval_batch_size: 1`

Evaluation runs in generation mode (full transcription), which uses significantly
more memory than training. Set to 1 to avoid MPS OOM. (This was the cause of the
small model crash at step 500.)

### `per_device_train_batch_size: 4` / `gradient_accumulation_steps: 4`

Effective batch size = 16, same as all other runs. Consistent across models for
fair comparison.

---

## Safe stop and resume

Same rules as `hebrew_tiny_ft_v2.md`:
- Ctrl+C is safe — latest checkpoint is always complete
- Re-run the same command to resume automatically
- Do not change `max_steps` between runs

```bash
uv run python scripts/finetune.py --config configs/hebrew_base_ft.yaml
```

---

## What comes next after base

See `ai_specs/improve-hebrew-models.md` for the full roadmap. Short version:
1. Run ACFT on the fine-tuned base model (same as was done for tiny and small)
2. Consider a v2 run for small (`hebrew_small_ft_v2`) — only ~1 epoch was seen
3. Add `ivrit-ai/crowd-transcribe-v5` as a second dataset
