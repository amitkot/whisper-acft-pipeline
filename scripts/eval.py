#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.1",
#   "transformers>=4.41",
#   "datasets[audio]==3.6.0",
#   "jiwer>=3.0",
#   "numpy>=1.24",
#   "pyyaml>=6.0",
#   "soundfile>=0.12",
#   "tqdm>=4.0",
# ]
# ///
"""Evaluate one or more Whisper models on a Hebrew ASR dataset.

Runs greedy generation on a fixed eval set and reports WER per model.
Designed for apples-to-apples comparison across checkpoints.

Usage examples:
  # Compare all fine-tuned models
  uv run python scripts/eval.py \\
      outputs/hebrew_tiny_ft_v2/final \\
      outputs/hebrew_base_ft/final \\
      outputs/hebrew_small_ft/final

  # Include baseline (original OpenAI weights, no fine-tuning)
  uv run python scripts/eval.py \\
      openai/whisper-tiny \\
      openai/whisper-base \\
      outputs/hebrew_tiny_ft_v2/final

  # Limit samples
  uv run python scripts/eval.py outputs/hebrew_base_ft/final --samples 500

  # Full test split (8426 examples, slow)
  uv run python scripts/eval.py outputs/hebrew_base_ft/final --samples 0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jiwer
import torch
from datasets import Audio, load_dataset
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor


DATASET_NAME = "ivrit-ai/whisper-training"
DATASET_SPLIT = "test"
TEXT_COLUMN = "text"
AUDIO_COLUMN = "audio"
DEFAULT_SAMPLES = 2000


def load_eval_dataset(n_samples: int):
    print(f"Loading eval dataset: {DATASET_NAME} split={DATASET_SPLIT}", flush=True)
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=False, trust_remote_code=True)
    ds = ds.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))
    ds = ds.filter(lambda x: bool(x[TEXT_COLUMN] and str(x[TEXT_COLUMN]).strip()))

    if n_samples and n_samples < len(ds):
        ds = ds.select(range(n_samples))

    print(f"Eval set: {len(ds)} examples", flush=True)
    return ds


def run_model(model_id: str, ds, device: str, batch_size: int) -> dict:
    print(f"\n{'─'*60}", flush=True)
    print(f"Model: {model_id}", flush=True)

    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.to(device)
    model.train(False)

    model.generation_config.language = "he"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    refs, hyps = [], []
    t0 = time.time()

    for i in tqdm(range(0, len(ds), batch_size), desc="  batches"):
        batch = ds[i : i + batch_size]

        audio_arrays = [a["array"] for a in batch[AUDIO_COLUMN]]
        inputs = processor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_features = inputs.input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(input_features)

        transcriptions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        refs.extend([str(t) for t in batch[TEXT_COLUMN]])
        hyps.extend(transcriptions)

    elapsed = time.time() - t0

    # Filter pairs where reference is empty (jiwer raises on empty strings)
    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    clean_refs, clean_hyps = zip(*pairs)
    wer = jiwer.wer(list(clean_refs), list(clean_hyps))

    del model
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": model_id,
        "samples": len(pairs),
        "wer": wer,
        "elapsed_s": elapsed,
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate Whisper models on Hebrew ASR")
    ap.add_argument(
        "models",
        nargs="+",
        help="Model IDs or local paths (e.g. openai/whisper-tiny, outputs/hebrew_base_ft/final)",
    )
    ap.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"Number of test examples (default: {DEFAULT_SAMPLES}, 0 = full split)",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Inference batch size (default: 4)",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: mps, cuda, cpu (default: auto-detect)",
    )
    args = ap.parse_args()

    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}", flush=True)

    ds = load_eval_dataset(args.samples)

    results = []
    for model_id in args.models:
        result = run_model(model_id, ds, device, args.batch_size)
        results.append(result)
        print(f"  WER: {result['wer']:.4f}  ({result['elapsed_s']:.0f}s elapsed)", flush=True)

    # Summary table
    print(f"\n{'═'*60}")
    print(f"{'Model':<45} {'WER':>7}  {'Samples':>8}")
    print(f"{'─'*60}")
    for r in sorted(results, key=lambda x: x["wer"]):
        p = Path(r["model"])
        label = str(p.parent.name / p.name) if p.exists() else r["model"]
        print(f"  {label:<43} {r['wer']:>7.4f}  {r['samples']:>8,}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
