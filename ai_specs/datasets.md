# Hebrew ASR Dataset Research

Reference for dataset selection across all training runs.
Last updated based on HuggingFace search + ivrit.ai article analysis.

---

## Key principle: match training distribution to inference

Source: [ivrit.ai blog — "Fine Tune Whisper the Right Way"](https://www.ivrit.ai/en/2025/02/13/training-whisper/)

The ivrit.ai team discovered that their original short-clip dataset was perfectly fine for
teaching Hebrew, but caused failures in long-form transcription because the model never saw
the timestamp and previous-text conditioning tokens that Whisper uses across 30-second
boundaries.

**For keyboard dictation (our use case), this warning inverts:**
- Users speak short clips — one sentence, a short phrase
- There are no 30-second boundaries to stitch
- Timestamp tokens and previous-text conditioning are irrelevant
- The dataset format that ivrit.ai considered a limitation is actually ideal for us

**Concrete implication:** Prefer short supervised clips over long-window data.
Average clip length is more important than total hours.

---

## Datasets

### `ivrit-ai/whisper-training` — PRIMARY (currently in use)

- **Size**: ~400h, ~55k examples
- **Access**: Gated (requires HF account approval)
- **Format**: `uuid`, `audio`, `orig_text`, `text` columns. Split: `train`, `test`
- **Sample rate**: 44100 Hz (script resamples to 16kHz)
- **Avg clip length**: Short, non-consecutive clips averaging ~5 seconds (ivrit.ai)
- **Label quality**: Supervised, clean
- **Why it's the best primary source**: Directly training-oriented, matches
  dictation-style short utterances, clean supervised labels
- **Used in**: `hebrew_tiny_ft`, `hebrew_tiny_ft_v2`, `hebrew_small_ft`

### `ivrit-ai/crowd-transcribe-v5` — NEXT PRIORITY

- **Size**: ~300h
- **Access**: Not gated
- **Format**: `uuid`, `audio`, `orig_sentence`, `sentence` + quality metadata columns:
  `noisy`, `multiple_speakers`, etc.
- **Split**: Train only (no test split)
- **Why useful**: More short Hebrew speech, diverse speakers, different recording conditions
- **Caution**: Quality varies — **must filter** using the quality metadata columns:
  exclude `noisy=True` and `multiple_speakers=True` examples for best results
- **Compatibility**: Similar format to whisper-training; needs `text_column` config override
  to `sentence`
- **Status**: Not yet used. First candidate for multi-dataset training.

### `ivrit-ai/crowd-recital-whisper-training`

- **Size**: Unknown (smaller, ~40 downloads)
- **Access**: Check HF page
- **Format**: Likely whisper-training format (name suggests it)
- **Content**: Read speech (recitation-style)
- **Why useful**: Clean labels, whisper-training compatible format
- **Caution**: Read speech is less representative of dictation than conversational speech.
  Lower priority than crowd-transcribe-v5.
- **Status**: Not yet inspected

### `fsicoli/common_voice_22_0` (Hebrew split)

- **Size**: Unknown Hebrew subset size (full dataset is large, Hebrew is a minority)
- **Access**: Not gated
- **Format**: Mozilla Common Voice standard format
- **Content**: Community-recorded short clips, diverse speakers, mobile-like audio
- **Why useful**: Short utterances, diverse speakers, closest to mobile mic conditions
- **Caution**: Needs format normalization (different column names). Hebrew subset may be
  small. Community recordings can have inconsistent quality.
- **Status**: Not yet inspected

### `imvladikon/hebrew_speech_kan`

- **Size**: Unknown
- **Content**: KAN public broadcaster (Israeli news/radio)
- **Why useful**: Broadcast-quality audio, formal Hebrew, different acoustic conditions
- **Caution**: Broadcast news style ≠ casual dictation. Clips may be longer.
  Lower priority for dictation use case.
- **Status**: Not yet inspected

### `ivrit-ai/knesset-plenums-whisper-training`

- **Size**: Unknown
- **Content**: Israeli parliament (Knesset) plenary session transcripts
- **Why useful**: Formal Hebrew, high-quality transcripts, domain diversity
- **Caution**: **Long-form parliamentary speech is the wrong domain for keyboard dictation.**
  Speaking style, vocabulary, and clip length all diverge from dictation.
  Skip for now unless testing domain robustness specifically.
- **Status**: Not recommended for primary use

---

## Datasets for pseudo-labeling (unlabeled audio)

### `ivrit-ai/audio-v2`

- **Size**: ~22,000 hours of Hebrew audio, 80+ content sources, 6,000+ speakers
- **Access**: HF (806 downloads), ivrit.ai license (research + commercial AI training OK)
- **Content**: Podcasts, YouTube channels — politics, sports, medicine, Gemara lessons, etc.
  Diverse ages and accents including Hebrew as second language.
- **Why useful**: Massive unlabeled pool for pseudo-labeling pipeline —
  run `ivrit-ai/whisper-large-v3-turbo` (not our small model) over it to generate
  training pairs for tiny/base. Using the strongest available teacher minimizes error propagation.
- **Caution**: Machine-generated transcriptions inherit teacher model errors.
  Must filter clips where the model is uncertain (repetition loops, [BLANK_AUDIO], very short output)
- **Status**: Not yet used. Relevant for pseudo-labeling after distillation pipeline is built.

## ivrit-ai pre-trained models (for distillation, not training data)

Source: [ivrit-ai HuggingFace](https://huggingface.co/ivrit-ai), [Interspeech 2025 paper](https://www.isca-archive.org/interspeech_2025/marmor25_interspeech.pdf)

ivrit-ai trained Whisper models on **5,050 hours** of Hebrew:
- ~4,700h `knesset-plenums-whisper-training` (Israeli parliament)
- ~300h `crowd-transcribe-v5`
- ~50h `crowd-recital-whisper-training`

| Model | Params | Format | Notes |
|---|---|---|---|
| `ivrit-ai/whisper-large-v3` | 2B | HF PyTorch | Best quality, 2-phase training (3 epochs knesset + 1 epoch mixed) |
| `ivrit-ai/whisper-large-v3-turbo` | 0.8B | HF PyTorch | Distilled from large-v3, 50% fewer params |
| `ivrit-ai/whisper-large-v3-ct2` | 2B | CTranslate2 | Production inference only |
| `ivrit-ai/whisper-large-v3-turbo-ct2` | 0.8B | CTranslate2 | Most popular (16.5k downloads/month) |

For distillation: **use HF PyTorch format** (need to extract logits).
ct2 format is for fast inference only, not suitable as distillation teacher.

---

## What to skip

| Dataset | Reason to skip |
|---|---|
| `ivrit-ai/knesset-plenums-whisper-training` | Long-form parliamentary, wrong domain |
| `ivrit-ai/audio-v2-40s` | 40-second chunks, too long for dictation training |
| `verbit/hebrew_medical_audio` | Domain too narrow, tiny dataset |
| `akiva-skolnik/hebrew-impairment-speech-v1` | Impairment speech, wrong population |
| Any timestamp-rich dataset | Timestamp conditioning irrelevant for short-utterance keyboard use |

---

## Multi-dataset implementation notes

`finetune.py` currently supports one dataset per run. To combine datasets, the script
needs a multi-dataset interleaving loader. The HuggingFace `datasets` library supports
this via `interleave_datasets()` with configurable sampling probabilities:

```python
from datasets import interleave_datasets

combined = interleave_datasets(
    [ds_primary, ds_secondary],
    probabilities=[0.7, 0.3],  # weight primary higher
    seed=42,
)
```

Weighting: **weight cleaner supervised data (whisper-training) higher** than noisier
crowd-sourced data. A 70/30 or 60/40 split is reasonable as a starting point.
