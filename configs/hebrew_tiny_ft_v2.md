# hebrew_tiny_ft_v2 — Parameter Decisions

Companion notes for `hebrew_tiny_ft_v2.yaml`.
Explains why each non-obvious parameter was chosen, based on lessons from the v1 run.

---

## Background: what went wrong in v1

The first tiny run (`hebrew_tiny_ft`) had two problems:

1. **A mid-run network disconnect** caused a resume around step 2000. The LR schedule
   was properly restored (HF Trainer saves `scheduler.pt`), but the epoch counter
   reset visually due to `ignore_data_skip=True` on a streaming dataset. Training
   continued to step 5000.

2. **Linear LR schedule** decayed to near-zero by step 5000, so the model was
   barely learning in the final steps. The best checkpoint was actually at step 3000
   (WER 0.636), not the end.

3. **Only ~0.6 epochs of data seen total** — the model never got a full pass through
   the training set, let alone multiple epochs.

v2 is a clean run from scratch addressing all three.

---

## Parameter decisions

### `base_model: openai/whisper-tiny`
Fresh start from the original OpenAI weights rather than resuming from the v1
checkpoint. The v1 run had a resume artifact mid-training; starting clean avoids
layering a second training run on top of a compromised first one. The ~3 hours of
saved time from resuming is not worth the uncertainty.

### `max_steps: 15000`
At batch_size=16 and ~55k examples, one epoch ≈ 5,000 steps.
15,000 steps = ~3 epochs, which is the estimated sweet spot for a 39M-parameter model
on this dataset before overfitting risk rises. Set higher rather than lower because:
- You can always stop early if eval WER starts rising
- **Changing `max_steps` on resume recalculates the LR schedule and causes a jump**
  (see resume section below) — so setting it generously upfront avoids ever needing
  to extend it

Expected training time: ~8–9 hours on Apple M4 MPS.

### `learning_rate: 8e-6`
Slightly lower than v1's 1e-5. With a cosine schedule (which stays high longer
than linear), the peak LR can be slightly lower while still learning aggressively
early on. 1e-5 with cosine would be fine too; 8e-6 is a conservative choice.

### `lr_scheduler_type: cosine`
The most important change from v1. The difference between linear and cosine:

- **Linear**: LR drops in a straight line from peak to zero. By the final 20% of
  steps, LR is near zero and the model has stopped learning meaningfully.
- **Cosine**: LR follows a smooth curve — stays near the peak for longer, then
  decays gradually. Never fully reaches zero (typically ~5% of peak at the end).

Cosine is also more forgiving on resume: if you must change `max_steps`, the
cosine curve changes shape less abruptly than linear.

### `warmup_steps: 500`
At the very start of training, model weights are random (for a fresh start from
pretrained, they're not random but are being moved to a new domain). Large LR
steps on an unoriented model cause chaotic, destructive updates.
Warmup ramps LR from 0 → peak over the first 500 steps, letting the model
stabilize before aggressive learning begins.
500 steps = ~3% of 15,000 total steps, which is a standard ratio.

### `per_device_eval_batch_size: 1`
Evaluation runs the model in generation mode (it actually transcribes audio),
which uses much more memory than training (which only does a forward pass).
The v1 small model run crashed with OOM during eval when using the default
eval_batch_size=4. Set to 1 to be safe on MPS.

### `save_steps: 250`
v1 saved every 500 steps. At ~2s/step, that's up to ~8 minutes of lost work if
interrupted between saves. 250 steps ≈ 4 minutes of exposure. The checkpoint
files are small enough (the model is only 39M params) that saving twice as often
has negligible overhead.

### `save_total_limit: 2`
Keeps the two most recent checkpoints instead of one. Protects against a corrupt
checkpoint if the machine crashes mid-write — the previous checkpoint is still
available as a fallback.

### `fp16: false`
HF Trainer's automatic mixed precision (AMP) causes NaN gradients on Apple MPS.
fp32 is required. The script also auto-detects this and overrides if accidentally
set to true.

### `seed: 42`
Changed from v1's seed=0. Minor, but gives a different shuffle of the streaming
dataset, providing slightly different training examples in each epoch.

---

## Safe stop and resume

Stopping mid-run with Ctrl+C is safe. The latest checkpoint is always complete.
To resume, simply re-run the same command:

```bash
uv run python scripts/finetune.py --config configs/hebrew_tiny_ft_v2.yaml
```

The script auto-detects the latest checkpoint and resumes from it (`resume_latest: true`).

**The one rule**: do not change `max_steps` between a run and its resume.
The LR scheduler is saved in `scheduler.pt` inside the checkpoint and is restored
exactly. Changing `max_steps` recalculates the schedule for a different total
duration, placing the current step at a different LR than where it left off.
For example: stopping at step 5000 with max_steps=15000 and resuming with
max_steps=20000 would recalculate LR at step 5000 to be higher than it was,
potentially destabilizing training.

If you genuinely need to extend training beyond 15,000 steps, the right approach
is a **second-pass config** (new `run_name`, lower LR ~3e-6, `warmup_steps: 0`,
`base_model` pointing at the final checkpoint).

---

## Expected results

| Steps | Epochs | WER estimate |
|---|---|---|
| 5,000 | ~1 | ~0.62–0.65 |
| 10,000 | ~2 | ~0.53–0.58 |
| 15,000 | ~3 | ~0.48–0.55 |

These are estimates based on the v1 trajectory. The cosine schedule and clean
data should push toward the lower end. The hard floor for whisper-tiny on this
dataset is likely around 0.40–0.45 (model capacity limit).

For comparison: whisper-small-he achieved WER 0.368 on the same eval set.
