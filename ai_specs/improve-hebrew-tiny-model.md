# Improving whisper-tiny-he

## Context

`amitkot/whisper-tiny-he` was fine-tuned for 5000 steps on `ivrit-ai/whisper-training` (~400h).
The v1 run had a mid-run resume and used a linear LR schedule; best checkpoint was step 3000
(WER 0.636), not the final. Only ~0.6 epochs of data were seen. Several directions can improve it.

## Options

### 1. Continue Fine-Tuning on Same Dataset

**Decision: fresh start with better hyperparameters → `configs/hebrew_tiny_ft_v2.yaml`**

The v1 run had a mid-run resume artifact and a linear LR schedule that decayed to near-zero
before seeing 2 epochs of data. v2 fixes this with a cosine schedule, 15,000 steps (~3 epochs),
and safe checkpointing. See `configs/hebrew_tiny_ft_v2.md` for full parameter rationale.

Run with:
```bash
uv run python scripts/finetune.py --config configs/hebrew_tiny_ft_v2.yaml
```

Expected WER after 15,000 steps: ~0.48–0.55. Hard floor for tiny: ~0.40–0.45.

### 2. Add More Datasets

Combine `ivrit-ai/whisper-training` with additional Hebrew speech data for better generalization.

**Candidate datasets (all on HuggingFace):**

| Dataset | Notes |
|---|---|
| `ivrit-ai/crowd-transcribe-v5` | ~300h, not gated, diverse speakers/quality |
| `ivrit-ai/crowd-recital-whisper-training` | ivrit-ai, whisper-training format, read speech |
| `ivrit-ai/knesset-plenums-whisper-training` | Israeli parliament, formal Hebrew, different domain |
| `fsicoli/common_voice_22_0` (he split) | Mozilla CV, community speakers, most downloaded |
| `imvladikon/hebrew_speech_kan` | KAN public broadcaster, news/broadcast quality |

Best combination for diversity: **knesset + common_voice** adds formal speech and community
recordings on top of the existing read-speech corpus.

Each dataset needs a format check (column names, sample rate, splits) before adding to the pipeline.
The `finetune.py` config already supports `text_column`, `audio_column`, `dataset_config` overrides.
A multi-dataset interleaving loader would need to be added.

### 3. Knowledge Distillation from whisper-small-he

Use `amitkot/whisper-small-he` (WER 0.368) as a teacher to generate pseudo-labels for
unlabeled Hebrew audio, then fine-tune tiny on those labels.

- Tiny learns from a stronger teacher signal
- Can leverage large unlabeled Hebrew audio corpora
- More complex to implement (inference pipeline + new training loop or soft-label loss)

## Recommended Order

1. **Option 1** — run more steps, cheap and immediate
2. **Option 2** — add `crowd-transcribe-v5` first (same format as current dataset, low friction),
   then knesset + common_voice
3. **Option 3** — tackle after dataset expansion, as distillation is most complex
