# Distillation Experiment Log

Tracking experiments, results, failures, and lessons learned.

---

## Experiment 1: Online distillation (abandoned — too slow)

**Date**: 2026-03-19
**Config**: `hebrew_base_distill.yaml` (online mode, teacher loaded during training)

**Setup**:
- Teacher: `ivrit-ai/whisper-large-v3-turbo` (0.8B, fp16)
- Student: `outputs/hebrew_base_ft/final` (74M, fp32)
- alpha=0.5, temperature=2.0, lr=6e-6, cosine, batch_size=4

**Result**: ~15s/step on MPS. 15,000 steps would take ~63 hours. Abandoned.

**MPS optimization attempts**:
| Optimization | s/step | Verdict |
|---|---|---|
| Baseline (fp32 teacher, torch.no_grad) | 18s | — |
| fp16 teacher + torch.inference_mode | 15s | kept |
| PYTORCH_MPS_FAST_MATH=1 | 15s | no effect |
| PYTORCH_MPS_PREFER_METAL=1 | 20s | worse |
| torch.compile | 35s | much worse |
| Disable gradient_checkpointing | 15s | no effect |
| batch_size=8 | 17s | worse |
| Student-only (no teacher) | 2s | **target for offline** |

**Lesson**: The 0.8B teacher forward pass dominates (~13 of 15 seconds per step).
MPS cannot be optimized further — the teacher must be removed from the training loop.

---

## Experiment 2: Precomputed teacher logits

**Date**: 2026-03-19 to 2026-03-21
**Script**: `scripts/precompute_teacher.py`

**Setup**:
- Teacher: `ivrit-ai/whisper-large-v3-turbo` (0.8B, fp16)
- Dataset: `ivrit-ai/whisper-training` train split (55,234 valid examples after filtering)
- Top-K: 100 logits per token position
- batch_size=16, streaming mode

**Result**:
- Speed: ~0.95s per example, ~1s/example
- Total time: ~14 hours across two runs (with resume after stop)
- Storage: 1.2 GB (3,453 batch files, ~30 KB per example compressed)
- Format: npz with uint16 token IDs + float16 logit values + int32 labels

**Issues encountered**:
1. **Disk full (first attempt)**: Non-streaming `.map()` cached 44GB of processed features
   to HF cache. Fixed by switching to streaming mode.
2. **Resume overwrote files**: `examples_processed` counter reset on resume, causing
   batch files to overwrite. Fixed by using stream position (`example_idx`) for filenames.
3. **Missing first 17k examples**: Had to rename files and re-run partial range with `--stop-at`.

**Lesson**: Streaming mode is essential to avoid disk caching. File naming must use
absolute stream position, not run-local counters, for safe resume.

---

## Experiment 3: Offline distillation — tiny (FAILED)

**Date**: 2026-03-24
**Config**: `hebrew_tiny_distill.yaml` (offline mode)

**Setup**:
- Teacher logits: precomputed top-100 from Experiment 2
- Student: `outputs/hebrew_tiny_ft_v2/final` (39M, WER 0.581)
- alpha=0.5, temperature=2.0, lr=8e-6, cosine, batch_size=4, grad_accum=4
- Effective batch size: 16

**Result**: Model catastrophically degraded.

| Step | Train loss | Eval WER | Eval loss |
|------|-----------|----------|-----------|
| 0 (start) | — | 0.581 (known) | — |
| 500 | ~10 | 6.06 | 2.05 |
| 1000 | ~12 | 7.77 | 3.37 |
| 1500 | ~14 | 7.03 | 3.32 |
| 2000 | ~16 | 6.79 | 3.53 |
| 2500 | ~20 | 7.90 | 3.72 |
| 3000 | ~26 | 8.03 | 3.78 |
| 3451 | crashed (StopIteration) | — | — |

Training loss INCREASED over time. Eval WER went from 0.581 → 6+ (garbage output).

**Bugs fixed during the run** (but didn't address the core problem):
1. Eval collator crashed on examples without teacher logits → added key existence check
2. Dataset didn't loop for multi-epoch training → added `while True` wrapper

**Analysis — why it failed**:

1. **KL loss scale is enormous**: Combined loss of 17-27 is far above normal CE (~1-2).
   The KL term with T²=4 scaling dominates. The model optimizes for matching the teacher
   distribution at the expense of producing valid Hebrew text.

2. **Top-K renormalization mismatch**: Teacher probabilities are renormalized over top-100
   tokens (softmax on 100 values), but student log-probs are full-vocab softmax (51,865).
   The KL divergence between these differently-normalized distributions may not be
   meaningful — the student is penalized for probability mass it assigns outside top-100.

3. **alpha=0.5 gives equal weight to CE and KL**: With KL loss ~20x larger than CE loss,
   the actual gradient is ~95% from KL. The model abandons correct Hebrew to chase the
   teacher distribution.

4. **Learning rate may be too high**: 8e-6 for a model that's already fine-tuned.
   Distillation from a good starting point may need lower LR to avoid destroying
   learned representations.

5. **Temperature=2.0 spreads the distribution**: Makes the teacher distribution flatter,
   which means more probability mass on unlikely tokens, harder for the student to match.

**Lessons learned**:

- **Monitor eval WER from the first checkpoint.** If it's worse than the starting model,
  something is fundamentally wrong — don't let it run.
- **The KL loss must be comparable in magnitude to the CE loss.** If KL is 10-20x larger,
  it will dominate regardless of alpha.
- **Top-K renormalization creates a distribution mismatch** that may make the KL loss
  meaningless. Need to either: (a) use full-vocab teacher logits, (b) normalize the KL
  loss differently, or (c) compute KL only at the top-K positions without comparing to
  the student's full-vocab distribution.

---

## Proposed fixes for next attempt

### Fix 1: Normalize KL loss properly
The current implementation computes `F.log_softmax(student / T)` over 51,865 tokens
but `F.softmax(teacher_topk / T)` over only 100 tokens. These are different distributions.
Options:
- Compute KL only using the student's probability at the top-K positions, not the full softmax
- Or scale the KL loss by the number of active tokens to match CE scale

### Fix 2: Reduce KL weight drastically
Change alpha from 0.5 to 0.9 or higher (90% CE, 10% KL). Let the model stay grounded
in correct Hebrew, with the teacher providing a gentle nudge.

### Fix 3: Lower temperature
T=1.0 instead of 2.0. Less distribution flattening, more focused on the teacher's
top predictions.

### Fix 4: Lower learning rate
3e-6 or even 1e-6. The fine-tuned model is already good — distillation should be a
gentle refinement, not aggressive retraining.

### Fix 5: Sanity check
Before running 15,000 steps, run 100 steps and check:
- Is the training loss decreasing?
- Is the eval WER staying close to the starting model's WER (0.581)?
If WER jumps above 1.0 in the first 100 steps, abort and adjust hyperparameters.
