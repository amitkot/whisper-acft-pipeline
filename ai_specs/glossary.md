# Glossary

Terms used across the project documentation.

---

**Alpha (α)** — In distillation, the balance between two learning signals:
- Ground truth labels ("the correct Hebrew text is X")
- Teacher's soft predictions ("I think 90% X, 8% Y, 2% Z")

`loss = α × CE_loss + (1-α) × KL_loss`

α=1.0 means pure fine-tuning (ignore teacher). α=0.0 means pure teacher matching
(ignore ground truth). We found α=0.95 works — ground truth dominates, teacher gives
a gentle 5% nudge.

**ACFT (Audio-Context Fine-Tuning)** — A training method by FUTO that teaches Whisper
to handle short audio clips without repeating. Uses partial encoder with truncated
positional embeddings. Required for FUTO Keyboard deployment.

**CE (Cross-Entropy) loss** — Standard training loss. Measures how far the model's
predictions are from the correct answer. Lower is better. Normal range for fine-tuned
Whisper: ~0.5–2.0.

**Cosine LR scheduler** — Learning rate follows a smooth S-curve from peak to near-zero.
Stays high longer than linear, decays gradually. Preferred for multi-epoch training.
See `configs/hebrew_tiny_ft_v2.md` for detailed explanation.

**Distillation (Knowledge Distillation / KD)** — Training a small fast model (student)
to mimic a large accurate model (teacher). The student learns not just the correct answer
but the teacher's full probability distribution — including what the teacher was uncertain
about. Goal: student approaches teacher quality at a fraction of the inference cost.

**Epoch** — One full pass through all training examples. With 55k examples and batch
size 16: 1 epoch ≈ 3,437 steps.

**KL divergence (Kullback-Leibler)** — Measures how different two probability distributions
are. In distillation, it measures the gap between the student's output distribution and
the teacher's. Training minimizes this gap. KL=0 means the distributions are identical.

**LR (Learning Rate)** — How big each adjustment to the model's weights is per training
step. Too high → overshoots, unstable. Too low → barely learns. Typical values: 1e-6
to 1e-5 (0.000001 to 0.00001).

**MPS (Metal Performance Shaders)** — Apple's GPU compute framework for PyTorch on
Apple Silicon. Used for all training in this project (M4 Pro, 48GB unified memory).
Known limitations: no tensor cores, high kernel launch overhead, fp16 AMP issues.

**Soft labels** — The teacher's full probability distribution over the vocabulary for
each token position (e.g., "90% הוא, 8% הם, 2% היא"). Richer signal than hard labels
(just "הוא"). Preserved via KL divergence loss during distillation.

**Temperature (T)** — Controls how "spread out" the teacher's predictions are before
comparing with the student. T=1.0 keeps predictions as-is. T>1 flattens them (less
confident). We found T=1.0 works best — T=2.0 inflated the KL loss ~4x.

**Top-K logits** — Instead of saving the teacher's full vocabulary distribution (51,865
values per token), we save only the K most probable tokens and their logit values.
K=100 captures essentially all meaningful probability mass. Storage: ~30KB per example
vs ~4MB for full logits.

**WER (Word Error Rate)** — Primary metric for speech recognition quality.
WER = (insertions + deletions + substitutions) / reference_words. Lower is better.
WER=0 is perfect. WER>1.0 is possible (more errors than reference words, usually from
insertion of garbage text). Our WER is unnormalized — not comparable to published
benchmarks that use text normalization.
