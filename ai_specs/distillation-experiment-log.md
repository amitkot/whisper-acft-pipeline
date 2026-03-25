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

---

## Experiment 4: Offline distillation — tiny with fixed labels + α=0.95 (MODEST RESULTS)

**Date**: 2026-03-24 to 2026-03-25
**Config**: `hebrew_tiny_distill.yaml` (offline, α=0.95, T=1.0)

**Setup**:
- Teacher logits: precomputed top-100 from Experiment 2
- Student: `outputs/hebrew_tiny_ft_v2/final` (39M, WER 0.581)
- alpha=0.95, temperature=1.0, lr=8e-6, cosine, batch_size=4, grad_accum=4

**Bugs fixed before this run**:
1. Labels from npz were already processed (padded, shifted) — collator processed
   them again → corrupted labels, CE loss ~13. Fixed: use fresh labels from streaming
   audio text instead of npz labels.
2. IterableDataset didn't loop for multi-epoch → StopIteration at step 3451.
   Fixed: wrap in while-True loop with StopIteration handling.

**Result**: Model improved modestly.

| Step | Eval WER (200 samples) | Notes |
|------|------------------------|-------|
| 0 | 0.581 | Starting point (fine-tuned tiny) |
| 500 | 0.583 | Flat |
| 1500 | 0.563 | Starting to improve |
| 3500 | 0.542 | After epoch boundary |
| 6500 | 0.527 | |
| **10000** | **0.522** | **Best (checkpoint deleted by save_total_limit)** |
| 13500 | 0.536 | |
| 15000 | 0.563 | Final saved model (regressed from best) |

Best WER: 0.522 (10.1% relative improvement from 0.581).
Final saved WER: 0.563 (3.1% improvement — best checkpoint was lost).

**Why results were modest — the key insight**:

We distilled a fine-tuned student on the SAME data it was fine-tuned on. This is the
worst combination for distillation:

1. **CE loss (95% of signal) teaches nothing new.** The student already learned this data
   over 3 epochs of fine-tuning. Further CE on the same data has diminishing returns.
2. **KL loss (5% of signal) is too weak to compensate.** At α=0.95, the teacher's
   distribution provides only a 5% nudge — not enough to meaningfully reshape
   representations that were already optimized by fine-tuning.
3. **Starting from fine-tuned weights anchors the model.** The representations are
   "hardened" from fine-tuning. The teacher signal would be more effective on a fresh
   model whose representations are still forming.

**Comparison with what fine-tuning achieved:**
- Fine-tuning openai/whisper-tiny on this data: 1.004 → 0.581 (42% improvement)
- Distilling on same data with same model: 0.581 → 0.522 (10% improvement)
- The distillation added only a fraction of the value that fine-tuning provided

---

## Key learning: when distillation adds value

Distillation is most valuable when **at least one** of these is true:

1. **The data is new** — the student hasn't been trained on it before.
   The CE loss teaches the data, the KL loss shapes how the student learns it.

2. **The student is fresh** — hasn't been fine-tuned on this data.
   Both CE and KL are informative from step 1. The teacher guides representation
   formation rather than trying to reshape hardened representations.

We violated both: fine-tuned student on the same data. The CE signal was redundant
and the KL signal was too weak (α=0.95) to matter.

**Starting from openai/whisper-tiny (fresh) would likely be better:**
- CE loss is fully informative (model hasn't seen this data)
- KL loss guides learning from the start
- Result should match fine-tuning alone (0.581) PLUS teacher bonus
- Could potentially also use higher alpha (more teacher signal) since the
  representations aren't anchored yet

---

## Recommended next experiments (in priority order)

### Option A: Distill fresh student on same labeled data
- Student: `openai/whisper-tiny` (not fine-tuned)
- Same precomputed teacher logits
- Test multiple alpha values (0.5, 0.7, 0.9) for 1000 steps each
- Expected: should match or beat fine-tuning (WER 0.581), potentially reaching 0.52-0.55
- Time: ~6h for full run after alpha sweep

### Option B: Pseudo-label unlabeled audio, train on expanded data
- Run teacher on `ivrit-ai/audio-v2` (22,000h unlabeled Hebrew audio)
- Generate transcriptions (hard labels, not soft)
- Train student (fine-tuned or fresh) on labeled (400h) + pseudo-labeled data
- Expected: significant WER improvement from 50× more data
- Time: teacher inference on 22,000h + training time
- Challenge: teacher inference at ~1s/example on MPS = ~22,000 hours → needs GPU cloud
  or only use a subset (e.g., 2,000h subset ≈ 5× current data)

### Option C: Distill fresh student on labeled + pseudo-labeled data
- Combine Options A and B
- Best of both worlds: fresh student + more data + teacher guidance
- Requires Option B's pseudo-labeling infrastructure first
