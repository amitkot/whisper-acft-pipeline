#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.1",
#   "transformers>=4.41",
#   "datasets[audio]==3.6.0",
#   "accelerate>=0.30",
#   "jiwer>=3.0",
#   "numpy>=1.24",
#   "pyyaml>=6.0",
#   "soundfile>=0.12",
# ]
# ///
"""Supervised fine-tuning for Whisper using Seq2SeqTrainer.

Trains a Whisper model on a speech-to-text dataset to improve recognition
quality (e.g. Hebrew WER). Supports streaming datasets for large corpora
that don't fit on disk.

Inputs
- A YAML config file (see configs/hebrew_tiny_finetune.yaml)

Outputs
- Hugging Face style checkpoint in <output_dir>/<run_name>/final/
"""

from __future__ import annotations

import argparse
import dataclasses
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import jiwer
import numpy as np
import torch
import yaml
from datasets import Audio, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


# ------------------------------------
# Config
# ------------------------------------

@dataclasses.dataclass
class FinetuneConfig:
    run_name: str = "hebrew_tiny_ft"
    base_model: str = "openai/whisper-tiny"
    output_dir: str = "outputs"
    language: str = "he"
    task: str = "transcribe"

    # Dataset
    dataset_name: str = "ivrit-ai/whisper-training"
    dataset_config: Optional[str] = None
    train_split: str = "train"
    eval_split: str = "validation"
    text_column: str = "text"
    audio_column: str = "audio"
    streaming: bool = True
    shuffle_buffer: int = 500

    # Training (max_steps required for streaming)
    max_steps: int = 5000
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    fp16: bool = True
    gradient_checkpointing: bool = True

    # Eval & Save
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 1
    logging_steps: int = 25
    max_eval_samples: Optional[int] = 200

    # Generation (for WER eval)
    generation_max_length: int = 225
    predict_with_generate: bool = True

    # Runtime
    device: str = "mps"
    seed: int = 0
    dataloader_num_workers: int = 0  # MPS + macOS = 0 workers

    # Resume
    resume_from: Optional[str] = None
    resume_latest: bool = True


def _deep_update(dc: FinetuneConfig, overrides: Dict[str, Any]) -> FinetuneConfig:
    d = dataclasses.asdict(dc)
    for k, v in overrides.items():
        if k not in d:
            raise KeyError(f"Unknown config field: {k}")
        d[k] = v
    return FinetuneConfig(**d)


def load_config(path: Path) -> FinetuneConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = FinetuneConfig()
    return _deep_update(cfg, raw)


# ------------------------------------
# Data
# ------------------------------------

@dataclasses.dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def load_datasets(cfg: FinetuneConfig, processor: WhisperProcessor):
    """Load train (streaming) and eval (non-streaming, capped) datasets."""
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer

    max_label_length = 448  # Whisper max target positions

    def is_valid(example):
        """Filter out examples with empty text or missing audio."""
        text = example.get(cfg.text_column)
        audio = example.get(cfg.audio_column)
        if not text or not str(text).strip():
            return False
        if not audio or not isinstance(audio, dict) or "array" not in audio:
            return False
        return True

    def prepare(example):
        audio = example[cfg.audio_column]
        text = str(example[cfg.text_column])
        example["input_features"] = feature_extractor(
            audio["array"], sampling_rate=16000
        ).input_features[0]
        labels = tokenizer(text).input_ids
        example["labels"] = labels[:max_label_length]
        return example

    # Train: streaming by default
    train_ds = load_dataset(
        cfg.dataset_name,
        cfg.dataset_config,
        split=cfg.train_split,
        streaming=cfg.streaming,
        trust_remote_code=True,
    )
    if cfg.streaming:
        train_ds = train_ds.shuffle(buffer_size=cfg.shuffle_buffer, seed=cfg.seed)
    train_ds = train_ds.cast_column(cfg.audio_column, Audio(sampling_rate=16000))
    train_ds = train_ds.filter(is_valid)

    # Remove original columns to avoid carrying raw audio in memory
    col_names = train_ds.column_names
    if col_names:
        train_ds = train_ds.map(prepare, remove_columns=col_names)
    else:
        train_ds = train_ds.map(prepare)

    # Eval: non-streaming, capped at max_eval_samples
    eval_ds = None
    if cfg.eval_split:
        try:
            eval_ds = load_dataset(
                cfg.dataset_name,
                cfg.dataset_config,
                split=cfg.eval_split,
                streaming=False,
                trust_remote_code=True,
            )
        except (ValueError, KeyError):
            print(f"WARNING: eval split '{cfg.eval_split}' not found, skipping eval")

    if eval_ds is not None:
        if cfg.max_eval_samples:
            eval_ds = eval_ds.select(range(min(cfg.max_eval_samples, len(eval_ds))))
        eval_ds = eval_ds.cast_column(cfg.audio_column, Audio(sampling_rate=16000))
        eval_ds = eval_ds.filter(is_valid)
        eval_ds = eval_ds.map(prepare, remove_columns=eval_ds.column_names)

    return train_ds, eval_ds


# ------------------------------------
# Metrics
# ------------------------------------

def make_compute_metrics(processor: WhisperProcessor):
    tokenizer = processor.tokenizer

    def compute_metrics(pred):
        pred_ids = pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Filter out empty references (jiwer raises on empty strings)
        pairs = [(r, h) for r, h in zip(label_str, pred_str) if r.strip()]
        if not pairs:
            return {"wer": 0.0}
        refs, hyps = zip(*pairs)

        return {"wer": jiwer.wer(list(refs), list(hyps))}

    return compute_metrics


# ------------------------------------
# Resume
# ------------------------------------

def find_checkpoint(
    out_dir: Path,
    resume_from: Optional[str] = None,
    resume_latest: bool = True,
) -> Optional[str]:
    """Find checkpoint to resume from (Seq2SeqTrainer uses checkpoint-{step} dirs)."""
    if resume_from:
        p = Path(resume_from)
        if p.exists():
            return str(p)
        print(f"WARNING: resume_from={resume_from} not found, starting fresh")
        return None

    if not resume_latest:
        return None

    if not out_dir.exists():
        return None

    best_step = -1
    best_path: Optional[Path] = None
    for p in out_dir.iterdir():
        if not p.is_dir():
            continue
        m = re.match(r"checkpoint-(\d+)$", p.name)
        if not m:
            continue
        s = int(m.group(1))
        if s > best_step:
            best_step = s
            best_path = p

    if best_path:
        print(f"Found checkpoint to resume from: {best_path}")
        return str(best_path)
    return None


# ------------------------------------
# Training
# ------------------------------------

def train(cfg: FinetuneConfig) -> None:
    out_dir = Path(cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if training already completed
    final_dir = out_dir / "final"
    required = ["config.json", "model.safetensors", "tokenizer.json"]
    if final_dir.exists() and all((final_dir / f).exists() for f in required):
        print(f"Final model already exists at {final_dir}, skipping training.")
        return

    # Load processor & model
    processor = WhisperProcessor.from_pretrained(cfg.base_model)
    model = WhisperForConditionalGeneration.from_pretrained(cfg.base_model)

    # Configure generation for target language
    model.generation_config.language = cfg.language
    model.generation_config.task = cfg.task
    model.generation_config.forced_decoder_ids = None

    # Load data
    print(f"Loading dataset: {cfg.dataset_name} (streaming={cfg.streaming})")
    train_ds, eval_ds = load_datasets(cfg, processor)

    # Collator
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Disable fp16 on MPS — HF Trainer AMP doesn't work correctly on MPS
    use_fp16 = cfg.fp16
    if use_fp16 and cfg.device == "mps":
        print("WARNING: fp16 disabled — HF Trainer AMP is not supported on MPS, using fp32")
        use_fp16 = False

    # Training args
    has_eval = eval_ds is not None
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        fp16=use_fp16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=cfg.eval_steps if has_eval else None,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        predict_with_generate=cfg.predict_with_generate,
        generation_max_length=cfg.generation_max_length,
        logging_steps=cfg.logging_steps,
        load_best_model_at_end=has_eval,
        metric_for_best_model="wer" if has_eval else None,
        greater_is_better=False if has_eval else None,
        seed=cfg.seed,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=cfg.device != "mps",
        ignore_data_skip=cfg.streaming,
        report_to="none",
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=make_compute_metrics(processor) if has_eval else None,
        processing_class=processor.feature_extractor,
    )

    # Resume
    checkpoint = find_checkpoint(out_dir, cfg.resume_from, cfg.resume_latest)
    trainer.train(resume_from_checkpoint=checkpoint)

    # Save final model
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"Saved fine-tuned model to: {final_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune Whisper for speech recognition")
    ap.add_argument("--config", type=str, default="configs/hebrew_tiny_finetune.yaml")
    ap.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint dir to resume from (e.g. outputs/run/checkpoint-500)",
    )
    ap.add_argument(
        "--no-resume-latest", action="store_true",
        help="Disable auto-resume from latest checkpoint",
    )
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    if args.resume is not None:
        cfg = dataclasses.replace(cfg, resume_from=args.resume)
    if args.no_resume_latest:
        cfg = dataclasses.replace(cfg, resume_latest=False)

    train(cfg)


if __name__ == "__main__":
    main()
