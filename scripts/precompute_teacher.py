#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.1",
#   "transformers>=4.41",
#   "datasets[audio]==3.6.0",
#   "numpy>=1.24",
#   "pyyaml>=6.0",
#   "soundfile>=0.12",
#   "tqdm>=4.0",
# ]
# ///
"""Precompute teacher logits for offline knowledge distillation.

Runs a frozen teacher model over the training dataset and saves top-K
logits per token position to disk. The saved logits can then be used
by distill.py in offline mode, eliminating the teacher forward pass
during student training (~7x speedup on MPS).

Outputs a directory of .npz files (one per batch of examples), plus
a manifest.json with metadata.

Usage:
  uv run python scripts/precompute_teacher.py --config configs/hebrew_base_distill.yaml
  uv run python scripts/precompute_teacher.py --config configs/hebrew_base_distill.yaml --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from datasets import Audio, load_dataset
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor


DATASET_NAME = "ivrit-ai/whisper-training"
TEXT_COLUMN = "text"
AUDIO_COLUMN = "audio"
TOP_K = 100


def load_and_prepare_dataset(
    dataset_name: str,
    split: str,
    processor: WhisperProcessor,
    text_column: str = TEXT_COLUMN,
    audio_column: str = AUDIO_COLUMN,
    max_samples: Optional[int] = None,
):
    """Load dataset non-streaming, prepare features."""
    print(f"Loading dataset: {dataset_name} split={split}")
    ds = load_dataset(dataset_name, split=split, streaming=False, trust_remote_code=True)
    ds = ds.cast_column(audio_column, Audio(sampling_rate=16000))

    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer
    max_label_length = 448

    def is_valid(example):
        text = example.get(text_column)
        audio = example.get(audio_column)
        if not text or not str(text).strip():
            return False
        if not audio or not isinstance(audio, dict) or "array" not in audio:
            return False
        return True

    def prepare(example):
        audio = example[audio_column]
        text = str(example[text_column])
        example["input_features"] = feature_extractor(
            audio["array"], sampling_rate=16000
        ).input_features[0]
        labels = tokenizer(text).input_ids
        example["labels"] = labels[:max_label_length]
        return example

    print(f"Filtering and preparing {len(ds)} examples...")
    ds = ds.filter(is_valid)
    ds = ds.map(prepare, remove_columns=ds.column_names)
    print(f"Prepared {len(ds)} examples")
    return ds


def collate_batch(batch_examples: list, pad_token_id: int, decoder_start_token_id: int):
    """Collate a list of examples into padded tensors."""
    # Pad input features
    max_feat_len = max(len(ex["input_features"][0]) for ex in batch_examples)
    input_features = torch.stack([
        torch.tensor(ex["input_features"]) for ex in batch_examples
    ])

    # Pad labels
    max_label_len = max(len(ex["labels"]) for ex in batch_examples)
    labels_list = []
    for ex in batch_examples:
        lab = ex["labels"]
        padded = lab + [pad_token_id] * (max_label_len - len(lab))
        labels_list.append(padded)

    labels = torch.tensor(labels_list)

    # Mask padding with -100
    attention_mask = (labels != pad_token_id).long()
    labels = labels.masked_fill(attention_mask.ne(1), -100)

    if (labels[:, 0] == decoder_start_token_id).all().item():
        labels = labels[:, 1:]

    return input_features, labels


def main():
    ap = argparse.ArgumentParser(description="Precompute teacher logits for offline distillation")
    ap.add_argument("--config", type=str, default="configs/hebrew_base_distill.yaml",
                    help="Distillation config (reads teacher_model, dataset settings)")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Inference batch size (default: 16, no gradients so can be large)")
    ap.add_argument("--top-k", type=int, default=TOP_K,
                    help=f"Number of top logits to save per token position (default: {TOP_K})")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Output directory (default: outputs/<run_name>/teacher_logits)")
    ap.add_argument("--split", type=str, default="train",
                    help="Dataset split to process (default: train)")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Max samples to process (default: all)")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    # Load config for teacher model info
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    teacher_model_id = cfg.get("teacher_model", "ivrit-ai/whisper-large-v3-turbo")
    output_base = cfg.get("output_dir", "outputs")
    run_name = cfg.get("run_name", "hebrew_base_distill")

    out_dir = Path(args.output_dir or f"{output_base}/{run_name}/teacher_logits")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing manifest (resume support)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("complete"):
            print(f"Teacher logits already precomputed at {out_dir}, skipping.")
            return
        start_idx = manifest.get("next_idx", 0)
        print(f"Resuming from example {start_idx}")
    else:
        start_idx = 0

    # Device
    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Load teacher
    print(f"Loading teacher: {teacher_model_id}")
    processor = WhisperProcessor.from_pretrained(teacher_model_id)
    teacher = WhisperForConditionalGeneration.from_pretrained(
        teacher_model_id, torch_dtype=torch.float16,
    )
    teacher.generation_config.language = cfg.get("language", "he")
    teacher.generation_config.task = cfg.get("task", "transcribe")
    teacher.generation_config.forced_decoder_ids = None
    teacher.requires_grad_(False)
    teacher.train(False)
    teacher = teacher.to(device)

    pad_token_id = processor.tokenizer.pad_token_id
    decoder_start_token_id = teacher.config.decoder_start_token_id

    # Load dataset
    ds = load_and_prepare_dataset(
        dataset_name=cfg.get("dataset_name", DATASET_NAME),
        split=args.split,
        processor=processor,
        text_column=cfg.get("text_column", TEXT_COLUMN),
        audio_column=cfg.get("audio_column", AUDIO_COLUMN),
        max_samples=args.max_samples,
    )

    total = len(ds)
    batch_size = args.batch_size
    top_k = args.top_k
    t0 = time.time()

    log_path = out_dir / "precompute_log.jsonl"
    print(f"\nPrecomputing top-{top_k} logits for {total} examples, batch_size={batch_size}")
    print(f"Output: {out_dir}")
    print(f"Log: {log_path}\n")

    for batch_start in tqdm(range(start_idx, total, batch_size), desc="Precomputing"):
        batch_end = min(batch_start + batch_size, total)
        batch_examples = [ds[i] for i in range(batch_start, batch_end)]

        input_features, labels = collate_batch(
            batch_examples, pad_token_id, decoder_start_token_id
        )
        input_features = input_features.to(device=device, dtype=torch.float16)
        labels = labels.to(device)

        with torch.inference_mode():
            outputs = teacher(input_features=input_features, labels=labels)
            logits = outputs.logits.float()  # (B, seq_len, vocab)

        # Extract top-K per token position
        topk_vals, topk_ids = torch.topk(logits, top_k, dim=-1)  # (B, seq_len, K)

        # Save as compressed npz: token IDs as uint16, values as float16
        batch_file = out_dir / f"batch_{batch_start:06d}.npz"
        np.savez_compressed(
            batch_file,
            topk_ids=topk_ids.cpu().numpy().astype(np.uint16),
            topk_vals=topk_vals.cpu().to(torch.float16).numpy(),
            labels=labels.cpu().numpy().astype(np.int32),
        )

        # Log progress
        elapsed_so_far = time.time() - t0
        examples_done = batch_end - start_idx
        s_per_example = elapsed_so_far / examples_done if examples_done else 0
        remaining = (total - batch_end) * s_per_example
        log_entry = {
            "batch_start": batch_start,
            "batch_end": batch_end,
            "examples_done": batch_end,
            "total": total,
            "pct": round(100 * batch_end / total, 1),
            "s_per_example": round(s_per_example, 2),
            "elapsed_min": round(elapsed_so_far / 60, 1),
            "remaining_min": round(remaining / 60, 1),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Update manifest for resume
        manifest = {
            "teacher_model": teacher_model_id,
            "top_k": top_k,
            "total_examples": total,
            "batch_size": batch_size,
            "next_idx": batch_end,
            "complete": batch_end >= total,
            "split": args.split,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))

    elapsed = time.time() - t0
    size_mb = sum(f.stat().st_size for f in out_dir.glob("*.npz")) / 1024 / 1024

    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Saved {total} examples to {out_dir} ({size_mb:.0f} MB)")
    print(f"Speed: {elapsed/total:.2f}s per example, {elapsed/(total/batch_size):.2f}s per batch")


if __name__ == "__main__":
    main()
