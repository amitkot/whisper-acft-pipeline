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
"""Soft-label knowledge distillation for Whisper.

Trains a small student model (e.g. whisper-base) to match the output
distribution of a larger teacher model (e.g. ivrit-ai/whisper-large-v3-turbo)
while also learning from ground-truth labels.

Loss = alpha * CE(student, ground_truth) + (1-alpha) * KL(student, teacher) * T^2

Inputs
- A YAML config file (see configs/hebrew_base_distill.yaml)

Outputs
- Hugging Face style checkpoint in <output_dir>/<run_name>/final/
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import jiwer
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import Audio, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


# ------------------------------------
# Config
# ------------------------------------

@dataclasses.dataclass
class DistillConfig:
    run_name: str = "hebrew_base_distill"
    student_model: str = "openai/whisper-base"
    teacher_model: str = "ivrit-ai/whisper-large-v3-turbo"
    output_dir: str = "outputs"
    language: str = "he"
    task: str = "transcribe"

    # Distillation
    temperature: float = 2.0
    alpha: float = 0.5  # weight for CE loss; (1-alpha) for KL loss

    # Dataset
    dataset_name: str = "ivrit-ai/whisper-training"
    dataset_config: Optional[str] = None
    train_split: str = "train"
    eval_split: str = "validation"
    text_column: str = "text"
    audio_column: str = "audio"
    streaming: bool = True
    shuffle_buffer: int = 500

    # Training
    max_steps: int = 15000
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 6e-6
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 500
    weight_decay: float = 0.01
    fp16: bool = False
    gradient_checkpointing: bool = True

    # Save & Logging
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 2
    logging_steps: int = 25
    max_eval_samples: Optional[int] = 200

    # Generation (for WER during eval)
    generation_max_length: int = 225
    predict_with_generate: bool = True

    # Runtime
    device: str = "mps"
    seed: int = 42
    dataloader_num_workers: int = 0

    # Offline distillation (precomputed teacher logits)
    teacher_logits_dir: Optional[str] = None  # path to precomputed logits

    # Resume
    resume_from: Optional[str] = None
    resume_latest: bool = True


def _deep_update(overrides: Dict[str, Any]) -> DistillConfig:
    dc = DistillConfig()
    d = dataclasses.asdict(dc)
    for k, v in overrides.items():
        if k not in d:
            raise KeyError(f"Unknown config field: {k}")
        d[k] = v
    return DistillConfig(**d)


def load_config(path: Path) -> DistillConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _deep_update(raw)


# ------------------------------------
# Data (same as finetune.py)
# ------------------------------------

@dataclasses.dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    teacher_processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Student features
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Teacher features (may have different mel bins, e.g. 128 vs 80)
        teacher_features = [{"input_features": f["teacher_input_features"]} for f in features]
        teacher_batch = self.teacher_processor.feature_extractor.pad(teacher_features, return_tensors="pt")
        batch["teacher_input_features"] = teacher_batch["input_features"]

        # Labels (shared)
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def load_datasets(cfg: DistillConfig, processor: WhisperProcessor, teacher_processor: WhisperProcessor):
    student_fe = processor.feature_extractor
    teacher_fe = teacher_processor.feature_extractor
    tokenizer = processor.tokenizer
    max_label_length = 448

    def is_valid(example):
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
        audio_array = audio["array"]
        example["input_features"] = student_fe(
            audio_array, sampling_rate=16000
        ).input_features[0]
        example["teacher_input_features"] = teacher_fe(
            audio_array, sampling_rate=16000
        ).input_features[0]
        labels = tokenizer(text).input_ids
        example["labels"] = labels[:max_label_length]
        return example

    train_ds = load_dataset(
        cfg.dataset_name, cfg.dataset_config,
        split=cfg.train_split, streaming=cfg.streaming, trust_remote_code=True,
    )
    if cfg.streaming:
        train_ds = train_ds.shuffle(buffer_size=cfg.shuffle_buffer, seed=cfg.seed)
    train_ds = train_ds.cast_column(cfg.audio_column, Audio(sampling_rate=16000))
    train_ds = train_ds.filter(is_valid)

    col_names = train_ds.column_names
    if col_names:
        train_ds = train_ds.map(prepare, remove_columns=col_names)
    else:
        train_ds = train_ds.map(prepare)

    ds_for_eval = None
    if cfg.eval_split:
        try:
            ds_for_eval = load_dataset(
                cfg.dataset_name, cfg.dataset_config,
                split=cfg.eval_split, streaming=False, trust_remote_code=True,
            )
        except (ValueError, KeyError):
            print(f"WARNING: eval split '{cfg.eval_split}' not found, skipping")

    if ds_for_eval is not None:
        if cfg.max_eval_samples:
            ds_for_eval = ds_for_eval.select(range(min(cfg.max_eval_samples, len(ds_for_eval))))
        ds_for_eval = ds_for_eval.cast_column(cfg.audio_column, Audio(sampling_rate=16000))
        ds_for_eval = ds_for_eval.filter(is_valid)
        ds_for_eval = ds_for_eval.map(prepare, remove_columns=ds_for_eval.column_names)

    return train_ds, ds_for_eval


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
    if resume_from:
        p = Path(resume_from)
        if p.exists():
            return str(p)
        print(f"WARNING: resume_from={resume_from} not found, starting fresh")
        return None

    if not resume_latest or not out_dir.exists():
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
# Distillation Trainer
# ------------------------------------

class JsonLogCallback(TrainerCallback):
    """Appends every log entry to a JSON-lines file for live monitoring."""

    def __init__(self, log_path: Path):
        self.log_path = log_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        entry = {"step": state.global_step, "epoch": state.epoch, **logs}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")



class DistillationTrainer(Seq2SeqTrainer):
    def __init__(self, teacher_model, temperature, alpha, **kwargs):
        super().__init__(**kwargs)
        self.teacher = teacher_model
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Pop teacher features before student forward (student doesn't expect them)
        teacher_features = inputs.pop("teacher_input_features")

        # Student forward pass (with CE loss from labels)
        outputs = model(**inputs)
        ce_loss = outputs.loss
        student_logits = outputs.logits
        # Teacher forward (frozen, fp16 — cast inputs to match, logits back to fp32)
        with torch.inference_mode():
            teacher_outputs = self.teacher(
                input_features=teacher_features.to(dtype=self.teacher.dtype),
                labels=inputs["labels"],
            )
            teacher_logits = teacher_outputs.logits.float()

        # Align vocab sizes (large-v3-turbo has 51866 tokens, base/tiny have 51865)
        min_vocab = min(student_logits.size(-1), teacher_logits.size(-1))
        student_logits = student_logits[..., :min_vocab]
        teacher_logits = teacher_logits[..., :min_vocab]

        # KL divergence on temperature-scaled logits, masked at padding positions
        T = self.temperature
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)

        mask = (inputs["labels"] != -100).unsqueeze(-1)  # (B, seq_len, 1)
        kl_per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none")
        kl_loss = (kl_per_token * mask).sum() / mask.sum()
        kl_loss = kl_loss * (T ** 2)

        loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss

        return (loss, outputs) if return_outputs else loss


# ------------------------------------
# Offline distillation (precomputed logits)
# ------------------------------------

class OfflineDistillDataset(torch.utils.data.IterableDataset):
    """Yields examples with student audio features + precomputed teacher top-K logits.

    Iterates the streaming audio dataset and the precomputed npz files in lockstep.
    Both were produced in the same stream order, so they align naturally.
    """

    def __init__(
        self,
        logits_dir: Path,
        dataset_name: str,
        dataset_config: Optional[str],
        split: str,
        audio_column: str,
        text_column: str,
        feature_extractor,
        tokenizer,
        seed: int = 42,
    ):
        self.logits_dir = logits_dir
        self.batch_files = sorted(logits_dir.glob("batch_*.npz"))
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.audio_column = audio_column
        self.text_column = text_column
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.seed = seed
        self.max_label_length = 448

    def __iter__(self):
        # Loop forever — Trainer stops at max_steps
        while True:
            yield from self._iter_one_epoch()

    def _iter_one_epoch(self):
        ds = load_dataset(
            self.dataset_name, self.dataset_config,
            split=self.split, streaming=True, trust_remote_code=True,
        )
        ds = ds.cast_column(self.audio_column, Audio(sampling_rate=16000))
        stream = iter(ds)
        example_idx = 0

        def next_stream():
            """Advance stream, return (example, new_idx) or None if exhausted."""
            nonlocal example_idx
            try:
                ex = next(stream)
                example_idx += 1
                return ex
            except StopIteration:
                return None

        for batch_file in self.batch_files:
            # The filename encodes the stream position: batch_NNNNNN.npz
            batch_start = int(batch_file.stem.split("_")[1])

            # Skip stream to the right position
            while example_idx < batch_start:
                if next_stream() is None:
                    return

            data = np.load(batch_file)
            n_examples = data["labels"].shape[0]

            for i in range(n_examples):
                # Advance stream, skipping filtered examples
                while True:
                    example = next_stream()
                    if example is None:
                        return
                    text = example.get(self.text_column)
                    audio = example.get(self.audio_column)
                    if (text and str(text).strip()
                            and audio and isinstance(audio, dict)
                            and "array" in audio):
                        break

                # Compute student features from audio + fresh labels from text
                input_features = self.feature_extractor(
                    audio["array"], sampling_rate=16000
                ).input_features[0]
                labels = self.tokenizer(str(text)).input_ids[:self.max_label_length]

                yield {
                    "input_features": input_features,
                    "labels": labels,
                    "teacher_topk_ids": data["topk_ids"][i],  # (seq, K) uint16
                    "teacher_topk_vals": data["topk_vals"][i],  # (seq, K) float16
                }


@dataclasses.dataclass
class OfflineDataCollator:
    """Collates offline distillation examples with student features + teacher top-K."""
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Student input features
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels

        # Teacher top-K (only present in training examples, not eval)
        if "teacher_topk_ids" in features[0]:
            seq_len = labels.shape[1]
            topk_ids_list = []
            topk_vals_list = []
            for f in features:
                ids = torch.tensor(f["teacher_topk_ids"].astype(np.int64))
                vals = torch.tensor(f["teacher_topk_vals"].astype(np.float32))
                cur_len = ids.shape[0]
                if cur_len < seq_len:
                    pad = torch.zeros(seq_len - cur_len, ids.shape[1], dtype=ids.dtype)
                    ids = torch.cat([ids, pad], dim=0)
                    vals = torch.cat([vals, torch.zeros_like(pad, dtype=vals.dtype)], dim=0)
                else:
                    ids = ids[:seq_len]
                    vals = vals[:seq_len]
                topk_ids_list.append(ids)
                topk_vals_list.append(vals)
            batch["teacher_topk_ids"] = torch.stack(topk_ids_list)
            batch["teacher_topk_vals"] = torch.stack(topk_vals_list)
        return batch


class OfflineDistillationTrainer(Seq2SeqTrainer):
    """Distillation using precomputed top-K teacher logits (no teacher model loaded)."""

    def __init__(self, temperature, alpha, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        teacher_topk_ids = inputs.pop("teacher_topk_ids")  # (B, seq, K)
        teacher_topk_vals = inputs.pop("teacher_topk_vals")  # (B, seq, K)

        outputs = model(**inputs)
        ce_loss = outputs.loss
        student_logits = outputs.logits  # (B, seq, V)

        T = self.temperature

        # Full student log-probs over vocabulary
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)  # (B, seq, V)

        # Gather student log-probs at teacher's top-K positions
        student_log_probs_topk = student_log_probs.gather(
            -1, teacher_topk_ids.long()
        )  # (B, seq, K)

        # Teacher probs (renormalized softmax over top-K logits)
        teacher_probs_topk = F.softmax(teacher_topk_vals / T, dim=-1)  # (B, seq, K)

        # KL divergence over top-K, masked at padding positions
        mask = (inputs["labels"] != -100).unsqueeze(-1)  # (B, seq, 1)
        kl_per_token = (teacher_probs_topk * (
            teacher_probs_topk.log() - student_log_probs_topk
        ))  # (B, seq, K)
        kl_loss = (kl_per_token * mask).sum() / mask.sum()
        kl_loss = kl_loss * (T ** 2)

        loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss

        return (loss, outputs) if return_outputs else loss


# ------------------------------------
# Training
# ------------------------------------

def train(cfg: DistillConfig) -> None:
    out_dir = Path(cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    final_dir = out_dir / "final"
    required = ["config.json", "model.safetensors", "tokenizer.json"]
    if final_dir.exists() and all((final_dir / f).exists() for f in required):
        print(f"Final model already exists at {final_dir}, skipping training.")
        return

    # Determine offline vs online mode
    logits_dir = None
    if cfg.teacher_logits_dir:
        logits_dir = Path(cfg.teacher_logits_dir)
    else:
        # Auto-detect: check if logits exist at the default location
        default_logits = out_dir / "teacher_logits"
        if default_logits.exists() and list(default_logits.glob("batch_*.npz")):
            logits_dir = default_logits

    offline = logits_dir is not None
    if offline:
        print(f"Offline mode: using precomputed logits from {logits_dir}")
    else:
        print("Online mode: teacher loaded during training (slow)")

    # Load processor & student
    processor = WhisperProcessor.from_pretrained(cfg.student_model)
    student = WhisperForConditionalGeneration.from_pretrained(cfg.student_model)
    student.generation_config.language = cfg.language
    student.generation_config.task = cfg.task
    student.generation_config.forced_decoder_ids = None

    # Disable fp16 on MPS
    use_fp16 = cfg.fp16
    if use_fp16 and cfg.device == "mps":
        print("WARNING: fp16 disabled on MPS, using fp32")
        use_fp16 = False

    if offline:
        # Offline: no teacher model needed, load logits from disk
        train_ds = OfflineDistillDataset(
            logits_dir=logits_dir,
            dataset_name=cfg.dataset_name,
            dataset_config=cfg.dataset_config,
            split=cfg.train_split,
            audio_column=cfg.audio_column,
            text_column=cfg.text_column,
            feature_extractor=processor.feature_extractor,
            tokenizer=processor.tokenizer,
            seed=cfg.seed,
        )
        collator = OfflineDataCollator(
            processor=processor,
            decoder_start_token_id=student.config.decoder_start_token_id,
        )
    else:
        # Online: load teacher model
        teacher_processor = WhisperProcessor.from_pretrained(cfg.teacher_model)
        print(f"Loading teacher model: {cfg.teacher_model}")
        teacher = WhisperForConditionalGeneration.from_pretrained(
            cfg.teacher_model, torch_dtype=torch.float16,
        )
        teacher.generation_config.language = cfg.language
        teacher.generation_config.task = cfg.task
        teacher.generation_config.forced_decoder_ids = None
        teacher.requires_grad_(False)
        teacher.train(False)
        if cfg.device == "mps":
            teacher = teacher.to("mps")
        elif cfg.device == "cuda":
            teacher = teacher.to("cuda")

        print(f"Loading dataset: {cfg.dataset_name} (streaming={cfg.streaming})")
        train_ds, _ = load_datasets(cfg, processor, teacher_processor)
        collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            teacher_processor=teacher_processor,
            decoder_start_token_id=student.config.decoder_start_token_id,
        )

    # Eval dataset (same for both modes)
    ds_for_eval = None
    if cfg.eval_split:
        try:
            ds_for_eval = load_dataset(
                cfg.dataset_name, cfg.dataset_config,
                split=cfg.eval_split, streaming=False, trust_remote_code=True,
            )
        except (ValueError, KeyError):
            print(f"WARNING: eval split '{cfg.eval_split}' not found")

    if ds_for_eval is not None:
        fe = processor.feature_extractor
        tok = processor.tokenizer
        if cfg.max_eval_samples:
            ds_for_eval = ds_for_eval.select(range(min(cfg.max_eval_samples, len(ds_for_eval))))
        ds_for_eval = ds_for_eval.cast_column(cfg.audio_column, Audio(sampling_rate=16000))
        ds_for_eval = ds_for_eval.filter(
            lambda x: bool(x.get(cfg.text_column) and str(x[cfg.text_column]).strip()
                          and x.get(cfg.audio_column) and isinstance(x[cfg.audio_column], dict)
                          and "array" in x[cfg.audio_column])
        )

        def prepare_eval(example):
            example["input_features"] = fe(
                example[cfg.audio_column]["array"], sampling_rate=16000
            ).input_features[0]
            labels = tok(str(example[cfg.text_column])).input_ids
            example["labels"] = labels[:448]
            return example

        ds_for_eval = ds_for_eval.map(prepare_eval, remove_columns=ds_for_eval.column_names)

    has_eval = ds_for_eval is not None
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
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
        ignore_data_skip=True,
        remove_unused_columns=False,
        report_to="none",
    )

    log_path = out_dir / "training_log.jsonl"
    print(f"Live training log: {log_path}")

    # Use eval collator for eval dataset (no teacher features needed)
    eval_collator = OfflineDataCollator(
        processor=processor,
        decoder_start_token_id=student.config.decoder_start_token_id,
    ) if offline else collator

    if offline:
        trainer = OfflineDistillationTrainer(
            temperature=cfg.temperature,
            alpha=cfg.alpha,
            model=student,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=ds_for_eval,
            data_collator=collator,
            compute_metrics=make_compute_metrics(processor) if has_eval else None,
            processing_class=processor.feature_extractor,
            callbacks=[JsonLogCallback(log_path)],
        )
    else:
        trainer = DistillationTrainer(
            teacher_model=teacher,
            temperature=cfg.temperature,
            alpha=cfg.alpha,
            model=student,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=ds_for_eval,
            data_collator=collator,
            compute_metrics=make_compute_metrics(processor) if has_eval else None,
            processing_class=processor.feature_extractor,
            callbacks=[JsonLogCallback(log_path)],
        )

    checkpoint = find_checkpoint(out_dir, cfg.resume_from, cfg.resume_latest)
    trainer.train(resume_from_checkpoint=checkpoint)

    # Save final (with load_best_model_at_end, this is the best WER model)
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"Saved distilled model to: {final_dir}")


# ------------------------------------
# Network retry
# ------------------------------------

_NETWORK_ERRORS: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)
try:
    import requests as _requests
    _NETWORK_ERRORS = _NETWORK_ERRORS + (
        _requests.exceptions.ConnectionError,
        _requests.exceptions.ChunkedEncodingError,
        _requests.exceptions.Timeout,
    )
except ImportError:
    pass


def train_with_retry(cfg: DistillConfig, max_retries: int = 20, base_delay: int = 30) -> None:
    for attempt in range(max_retries):
        try:
            train(cfg)
            return
        except _NETWORK_ERRORS as exc:
            delay = min(base_delay * (2 ** attempt), 300)
            print(f"\nNetwork error on attempt {attempt + 1}: {exc}")
            print(f"Retrying in {delay}s (will auto-resume from latest checkpoint)...")
            time.sleep(delay)
    print(f"ERROR: Training failed after {max_retries} network retries.")
    raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Distill a Whisper teacher into a smaller student")
    ap.add_argument("--config", type=str, default="configs/hebrew_base_distill.yaml")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--no-resume-latest", action="store_true")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    if args.resume is not None:
        cfg = dataclasses.replace(cfg, resume_from=args.resume)
    if args.no_resume_latest:
        cfg = dataclasses.replace(cfg, resume_latest=False)

    train_with_retry(cfg)


if __name__ == "__main__":
    main()
