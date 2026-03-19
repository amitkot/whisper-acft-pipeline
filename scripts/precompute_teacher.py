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

Runs a frozen teacher model over the training dataset (streaming) and saves
top-K logits per token position to disk. The saved logits can then be used
by distill.py in offline mode, eliminating the teacher forward pass during
student training (~7x speedup on MPS).

Uses streaming to avoid caching the full dataset to disk.

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


def collate_batch(
    batch_examples: list,
    feature_extractor,
    tokenizer,
    pad_token_id: int,
    decoder_start_token_id: int,
    text_column: str,
    audio_column: str,
):
    """Prepare raw streaming examples into padded tensors."""
    max_label_length = 448

    input_features_list = []
    labels_list = []

    for ex in batch_examples:
        audio = ex[audio_column]
        text = str(ex[text_column])
        feats = feature_extractor(audio["array"], sampling_rate=16000).input_features[0]
        input_features_list.append(torch.tensor(feats))
        labels = tokenizer(text).input_ids[:max_label_length]
        labels_list.append(labels)

    input_features = torch.stack(input_features_list)

    # Pad labels
    max_label_len = max(len(lab) for lab in labels_list)
    padded_labels = []
    for lab in labels_list:
        padded = lab + [pad_token_id] * (max_label_len - len(lab))
        padded_labels.append(padded)

    labels = torch.tensor(padded_labels)
    attention_mask = (labels != pad_token_id).long()
    labels = labels.masked_fill(attention_mask.ne(1), -100)

    if (labels[:, 0] == decoder_start_token_id).all().item():
        labels = labels[:, 1:]

    return input_features, labels


def main():
    ap = argparse.ArgumentParser(description="Precompute teacher logits for offline distillation")
    ap.add_argument("--config", type=str, default="configs/hebrew_base_distill.yaml")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    teacher_model_id = cfg.get("teacher_model", "ivrit-ai/whisper-large-v3-turbo")
    output_base = cfg.get("output_dir", "outputs")
    run_name = cfg.get("run_name", "hebrew_base_distill")
    text_column = cfg.get("text_column", TEXT_COLUMN)
    audio_column = cfg.get("audio_column", AUDIO_COLUMN)

    out_dir = Path(args.output_dir or f"{output_base}/{run_name}/teacher_logits")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume support
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

    # Load teacher (fp16 for speed)
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

    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id
    decoder_start_token_id = teacher.config.decoder_start_token_id

    # Load dataset as streaming — no disk caching
    print(f"Loading dataset: {cfg.get('dataset_name', DATASET_NAME)} split={args.split} (streaming)")
    ds = load_dataset(
        cfg.get("dataset_name", DATASET_NAME),
        split=args.split,
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.cast_column(audio_column, Audio(sampling_rate=16000))

    batch_size = args.batch_size
    top_k = args.top_k
    t0 = time.time()
    log_path = out_dir / "precompute_log.jsonl"
    example_idx = 0
    examples_processed = 0
    batch_buffer: list = []

    print(f"\nPrecomputing top-{top_k} logits, batch_size={batch_size}")
    print(f"Output: {out_dir}")
    print(f"Log: {log_path}")
    if start_idx > 0:
        print(f"Skipping first {start_idx} examples (already processed)\n")
    else:
        print()

    for example in tqdm(ds, desc="Precomputing"):
        # Skip already-processed examples (resume support)
        if example_idx < start_idx:
            example_idx += 1
            continue

        # Filter invalid
        text = example.get(text_column)
        audio = example.get(audio_column)
        if not text or not str(text).strip():
            example_idx += 1
            continue
        if not audio or not isinstance(audio, dict) or "array" not in audio:
            example_idx += 1
            continue

        batch_buffer.append(example)
        example_idx += 1

        if len(batch_buffer) < batch_size:
            continue

        # Process batch
        input_features, labels = collate_batch(
            batch_buffer, feature_extractor, tokenizer,
            pad_token_id, decoder_start_token_id,
            text_column, audio_column,
        )
        input_features = input_features.to(device=device, dtype=torch.float16)
        labels = labels.to(device)

        with torch.inference_mode():
            outputs = teacher(input_features=input_features, labels=labels)
            logits = outputs.logits.float()

        topk_vals, topk_ids = torch.topk(logits, top_k, dim=-1)

        batch_file = out_dir / f"batch_{examples_processed:06d}.npz"
        np.savez_compressed(
            batch_file,
            topk_ids=topk_ids.cpu().numpy().astype(np.uint16),
            topk_vals=topk_vals.cpu().to(torch.float16).numpy(),
            labels=labels.cpu().numpy().astype(np.int32),
        )

        examples_processed += len(batch_buffer)
        batch_buffer = []

        # Log
        elapsed = time.time() - t0
        s_per_ex = elapsed / examples_processed if examples_processed else 0
        log_entry = {
            "examples_processed": examples_processed,
            "example_idx": example_idx,
            "pct_stream": "unknown (streaming)",
            "s_per_example": round(s_per_ex, 2),
            "elapsed_min": round(elapsed / 60, 1),
            "throughput_ex_per_min": round(examples_processed / (elapsed / 60), 1) if elapsed > 0 else 0,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Update manifest
        manifest = {
            "teacher_model": teacher_model_id,
            "top_k": top_k,
            "batch_size": batch_size,
            "next_idx": example_idx,
            "examples_processed": examples_processed,
            "complete": False,
            "split": args.split,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))

    # Handle remaining examples in buffer
    if batch_buffer:
        input_features, labels = collate_batch(
            batch_buffer, feature_extractor, tokenizer,
            pad_token_id, decoder_start_token_id,
            text_column, audio_column,
        )
        input_features = input_features.to(device=device, dtype=torch.float16)
        labels = labels.to(device)

        with torch.inference_mode():
            outputs = teacher(input_features=input_features, labels=labels)
            logits = outputs.logits.float()

        topk_vals, topk_ids = torch.topk(logits, top_k, dim=-1)

        batch_file = out_dir / f"batch_{examples_processed:06d}.npz"
        np.savez_compressed(
            batch_file,
            topk_ids=topk_ids.cpu().numpy().astype(np.uint16),
            topk_vals=topk_vals.cpu().to(torch.float16).numpy(),
            labels=labels.cpu().numpy().astype(np.int32),
        )
        examples_processed += len(batch_buffer)

    # Final manifest
    elapsed = time.time() - t0
    manifest = {
        "teacher_model": teacher_model_id,
        "top_k": top_k,
        "batch_size": batch_size,
        "next_idx": example_idx,
        "examples_processed": examples_processed,
        "complete": True,
        "split": args.split,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    size_mb = sum(f.stat().st_size for f in out_dir.glob("*.npz")) / 1024 / 1024
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Saved {examples_processed} examples to {out_dir} ({size_mb:.0f} MB)")
    print(f"Speed: {elapsed/examples_processed:.2f}s per example")


if __name__ == "__main__":
    main()
