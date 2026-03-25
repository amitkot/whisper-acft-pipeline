# Distillation Research Findings

Research into why our distillation approach underperformed, and what the literature
recommends for Whisper distillation specifically.

---

## What DistilWhisper (HuggingFace) actually does

Source: [huggingface/distil-whisper](https://github.com/huggingface/distil-whisper),
[Gandhi et al. 2023](https://arxiv.org/abs/2311.00430)

| Aspect | DistilWhisper | Our approach |
|---|---|---|
| Student init | **Copy layers from teacher** (maximally spaced decoder layers) | Load separate pretrained model |
| Encoder | **Frozen, copied from teacher** | Trained from scratch |
| CE labels | **Pseudo-labels** (teacher transcriptions) | Ground truth labels |
| KL logits | **Full vocabulary** (51,865 tokens) | Top-K=100 (0.19% of vocab) |
| Data | **22,000 hours** (pseudo-labeled) | 400 hours (labeled) |
| CE weight | 0.8 (fixed) | α (configurable) |
| KL weight | 1.0 (fixed, includes T² scaling) | (1-α) × T² |
| Temperature | 2.0 | Tested 1.0 and 2.0 |
| LR | 1e-4 with constant_with_warmup | 8e-6 with cosine |

**Key difference**: DistilWhisper initializes the student by copying decoder layers
directly from the teacher. This is only possible when teacher and student share the
same architecture dimensions. We cannot do this — whisper-tiny (384 hidden dim) and
whisper-large-v3-turbo (1280 hidden dim) are incompatible.

---

## Four compounding problems in our setup

### 1. Top-K logit bias (most impactful)

Source: [Sparse Logit Sampling, ACL 2025](https://arxiv.org/abs/2503.16870),
[Don't Ignore the Tail, 2025](https://arxiv.org/html/2602.20816)

We save only 100 of 51,865 token logits per position (0.19% of vocabulary).
The teacher's softmax is renormalized over these 100 tokens, creating an artificially
concentrated distribution that doesn't match the teacher's true output.

**The problem**:
- The student computes log_softmax over all 51,865 tokens
- The teacher distribution is renormalized over only 100 tokens
- The KL divergence between these differently-normalized distributions is biased
- The student is penalized for any probability mass on the 51,765 tokens outside top-100,
  but the teacher provides NO signal about what those probabilities should be
- With T=2.0, both distributions spread further, amplifying the mismatch

Published research confirms: "Caching Top-K probabilities provides biased estimates
of teacher probability distribution to the student, resulting in suboptimal performance
and calibration." (Sparse Logit Sampling paper)

**Solutions from the literature**:
- Use full-vocab logits (~100GB storage for our dataset — impractical on local disk)
- Use top-K=500+ to reduce bias (5-10× more storage, ~6-12GB — feasible)
- Importance sampling correction (complex to implement)
- **Drop KL entirely, use pseudo-labels** (teacher transcriptions as CE labels)

### 2. Fine-tuned student on same data

Source: [Born-Again Networks, Furlanello et al. 2018](https://arxiv.org/abs/1805.04770)

Self-distillation (training on same data) CAN work, but requires the student to start
from scratch. A fine-tuned student's representations are "hardened" — already converged
to a local minimum for this data. The KL loss tries to reshape these, but:

- CE gradient (dominant at α=0.95) pulls back toward the same minimum
- Only 5% teacher signal is too weak to escape the local minimum
- Fresh student avoids this — both CE and KL are informative from step 1

### 3. Capacity gap (20×)

Source: [Cho & Hariharan, ICCV 2019](https://arxiv.org/abs/1910.01348)

"Larger, more accurate teachers don't necessarily make better teachers." When the
capacity gap is too large, the student cannot mimic the teacher distribution, resulting
in high KL divergence that doesn't decrease during training.

Our gap: 39M (tiny) vs 809M (turbo) = 20× ratio. DistilWhisper avoids this by
initializing from teacher layers (0 gap at start).

### 4. Architecture mismatch

Whisper-tiny has 384 hidden dim, 4 encoder/decoder layers.
Whisper-large-v3-turbo has 1280 hidden dim, 32 encoder / 4 decoder layers.

We cannot copy teacher layers into the student (dimension mismatch).
We cannot freeze the encoder from the teacher (different encoder architecture).
The student must learn entirely different internal representations that produce
similar outputs — fundamentally harder than refining copied weights.

---

## What Hinton actually recommends

Source: [Hinton et al. 2015](https://arxiv.org/abs/1503.02531),
[Intel Distiller docs](https://intellabs.github.io/distiller/knowledge_distillation.html)

- Alpha should be **small** (more weight on KL, not less) — but this assumes
  CE and KL magnitudes are comparable. Our top-K bias breaks this assumption.
- Temperature 2-20 is task-dependent. **Lower T for small students** (T=2.5-4).
- T² scaling is required to make soft and hard target gradients comparable.

The standard α=0.5, T=2.0 recipe is NOT universal. Different tasks need different values.

---

## Multilingual DistilWhisper (alternative approach)

Source: [Ferraz et al. 2024](https://arxiv.org/html/2311.01070v2)

Uses a different recipe:
- T=1.0 (not 2.0)
- Jensen-Shannon divergence instead of KL
- KD weight = 2 (loss = CE + gate_loss + 2×JS)
- Student initialized from whisper-small (same family, compatible dimensions)

---

## Recommended path forward

Given our constraints (local M4 MPS, no matching architecture for layer copying,
400h labeled data + 22,000h unlabeled audio available):

### Option A: Pseudo-labeling (recommended)

Drop soft-label KL entirely. Use teacher to generate transcriptions (hard labels)
for unlabeled audio. Train student with standard CE on pseudo-labeled data.

**Why this works**:
- Sidesteps ALL four problems (no KL, no top-K bias, no capacity gap in loss, no arch mismatch)
- Proven by DistilWhisper as the primary method
- Simple implementation: reuses `finetune.py` directly
- The main gain comes from data scale (50× more data), not loss function tricks

**Steps**:
1. Run teacher on subset of `ivrit-ai/audio-v2` (2,000-5,000h)
2. Save transcriptions as a HuggingFace dataset
3. Train student on labeled (400h) + pseudo-labeled data with `finetune.py`

### Option B: Soft-label distillation with fixes

If we want to keep trying soft labels:
- Increase top-K to 500+ (reduces bias, ~6GB storage)
- Use fresh student (`openai/whisper-tiny`)
- α=0.5, T=1.0 (Multilingual DistilWhisper recipe)
- Consider Jensen-Shannon divergence instead of KL

### Option C: Combine A and B

Pseudo-label for data expansion, then soft-label distillation on the expanded dataset.
Requires both infrastructure pieces.

---

## Sources

- [HuggingFace distil-whisper](https://github.com/huggingface/distil-whisper)
- [Distil-Whisper paper (Gandhi et al. 2023)](https://arxiv.org/abs/2311.00430)
- [Multilingual DistilWhisper (Ferraz et al. 2024)](https://arxiv.org/html/2311.01070v2)
- [Hinton et al. — Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Sparse Logit Sampling (ACL 2025)](https://arxiv.org/abs/2503.16870)
- [Don't Ignore the Tail (2025)](https://arxiv.org/html/2602.20816)
- [On the Efficacy of KD (Cho & Hariharan, ICCV 2019)](https://arxiv.org/abs/1910.01348)
- [Born-Again Networks (Furlanello et al. 2018)](https://arxiv.org/abs/1805.04770)
- [Whisper-KDQ (Shao et al. 2023)](https://arxiv.org/abs/2305.10788)
