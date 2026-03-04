#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.1",
#   "transformers>=4.41",
#   "datasets[audio]==3.6.0",
#   "accelerate>=0.30",
#   "safetensors>=0.4",
#   "numpy>=1.24",
#   "pyyaml>=6.0",
#   "soundfile>=0.12",
# ]
# ///
"""ACFT finetuning for Whisper, packaged as a single uv-run script.

Speed-focused defaults for Apple Silicon (MPS)
- Uses MPS autocast when fp16=true
- Uses feature_extractor directly (less overhead than full processor call)
- Optional: precompute mel features once via dataset.map (biggest speed win)
- DataLoader tuned for macOS (persistent workers, prefetch)

Inputs
- A YAML config file (see configs/hebrew_tiny_acft.yaml)

Outputs
- Hugging Face style checkpoint in <output_dir>/<run_name>/
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import random
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from datasets import Audio, Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_cosine_schedule_with_warmup
import torch.nn.functional as F


# -----------------------------
# Config
# -----------------------------

@dataclasses.dataclass
class TrainConfig:
    run_name: str = "hebrew_tiny_acft"
    base_model: str = "mike249/whisper-tiny-he-2"
    output_dir: str = "outputs"

    # Data
    dataset_name: str = "google/fleurs"
    dataset_config: str = "he_il"
    train_split: str = "train"
    eval_split: str = "validation"

    text_column: str = "transcription"
    audio_column: str = "audio"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = 200

    # ACFT: short-context crop length (seconds)
    min_audio_seconds: float = 1.5
    max_audio_seconds: float = 12.0

    # Optimization
    seed: int = 1337
    device: str = "auto"   # auto|mps|cpu|cuda
    fp16: bool = True      # for MPS: uses autocast float16
    batch_size: int = 8
    grad_accum_steps: int = 2
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_steps: int = 2000
    eval_every_steps: int = 250
    save_every_steps: int = 500

    # DataLoader
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True

    # Precompute features (big speed win). If true, we compute input_features once
    # and remove raw audio from the hot path.
    precompute_features: bool = True

    # Resume
    resume_from: Optional[str] = None
    resume_latest: bool = True

    # Misc
    log_every_steps: int = 25
    max_text_len: int = 256


def _deep_update(dc: TrainConfig, overrides: Dict[str, Any]) -> TrainConfig:
    d = dataclasses.asdict(dc)
    for k, v in overrides.items():
        if k not in d:
            raise KeyError(f"Unknown config field: {k}")
        d[k] = v
    return TrainConfig(**d)


def load_config(path: Path) -> TrainConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = TrainConfig()
    return _deep_update(cfg, raw)


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seconds_to_mel_frames(seconds: float) -> int:
    # Whisper mel features are effectively 100 frames per second.
    return max(1, int(round(seconds * 100)))


def crop_mels(mels: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Randomly crop mel frames to a shorter segment, then pad back to original length."""
    frames = mels.shape[-1]
    if target_frames >= frames:
        return mels
    max_start = frames - target_frames
    start = torch.randint(0, max_start + 1, (1,), device=mels.device).item()
    crop = mels[..., start : start + target_frames]
    return F.pad(crop, (0, frames - target_frames))


def pad_or_trim_mels(x: torch.Tensor, expected: int = 3000) -> torch.Tensor:
    t = x.shape[-1]
    if t < expected:
        return F.pad(x, (0, expected - t))
    if t > expected:
        return x[..., :expected]
    return x


def _find_latest_checkpoint(out_root: Path) -> Optional[Path]:
    if not out_root.exists():
        return None
    best_step = -1
    best_path: Optional[Path] = None
    for p in out_root.iterdir():
        if not p.is_dir():
            continue
        m = re.match(r"step-(\d+)$", p.name)
        if not m:
            continue
        s = int(m.group(1))
        if s > best_step:
            best_step = s
            best_path = p
    return best_path


def _save_training_state(out_dir: Path, optim: torch.optim.Optimizer, sched: torch.optim.lr_scheduler._LRScheduler, step: int) -> None:
    state = {
        "step": step,
        "optim": optim.state_dict(),
        "sched": sched.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(state, out_dir / "training_state.pt")


def _load_training_state(path: Path, optim: torch.optim.Optimizer, sched: torch.optim.lr_scheduler._LRScheduler) -> int:
    ts = path / "training_state.pt"
    if not ts.exists():
        m = re.match(r"step-(\d+)$", path.name)
        return int(m.group(1)) if m else 0

    state = torch.load(ts, map_location="cpu", weights_only=False)
    step = int(state.get("step", 0))

    try:
        optim.load_state_dict(state["optim"])
    except Exception:
        pass
    try:
        sched.load_state_dict(state["sched"])
    except Exception:
        try:
            sched.last_epoch = step - 1
        except Exception:
            pass

    rng = state.get("rng") or {}
    try:
        random.setstate(rng.get("python"))
    except Exception:
        pass
    try:
        np.random.set_state(rng.get("numpy"))
    except Exception:
        pass
    try:
        torch.random.set_rng_state(rng.get("torch"))
    except Exception:
        pass

    return step


# -----------------------------
# Data
# -----------------------------

def load_speech_dataset(cfg: TrainConfig) -> Tuple[Dataset, Optional[Dataset]]:
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, trust_remote_code=True)
    train = ds[cfg.train_split]
    ev = ds[cfg.eval_split] if cfg.eval_split in ds else None

    train = train.cast_column(cfg.audio_column, Audio(sampling_rate=16000))
    if ev is not None:
        ev = ev.cast_column(cfg.audio_column, Audio(sampling_rate=16000))

    if cfg.max_train_samples is not None:
        train = train.select(range(min(cfg.max_train_samples, len(train))))
    if ev is not None and cfg.max_eval_samples is not None:
        ev = ev.select(range(min(cfg.max_eval_samples, len(ev))))
    return train, ev


def maybe_precompute_features(ds: Dataset, processor: WhisperProcessor, cfg: TrainConfig) -> Dataset:
    if not cfg.precompute_features:
        return ds

    fe = processor.feature_extractor

    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        arr = ex[cfg.audio_column]["array"]
        feats = fe(arr, sampling_rate=16000, return_tensors="pt").input_features[0]  # (80, T)
        feats = pad_or_trim_mels(feats, 3000)
        ex["input_features"] = feats.to(torch.float32).numpy()
        return ex

    num_proc = max(1, min(cfg.num_workers, os.cpu_count() or 1))
    ds2 = ds.map(_map, num_proc=num_proc, desc="Precomputing mel features (one-time)")
    keep = [cfg.text_column, "input_features"]
    drop = [c for c in ds2.column_names if c not in keep]
    return ds2.remove_columns(drop)


class Collator:
    def __init__(self, processor: WhisperProcessor, cfg: TrainConfig):
        self.cfg = cfg
        self.fe = processor.feature_extractor
        self.tok = processor.tokenizer

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [str(b[self.cfg.text_column]) for b in batch]

        if "input_features" in batch[0]:
            feats = torch.tensor([b["input_features"] for b in batch], dtype=torch.float32)
        else:
            audio = [b[self.cfg.audio_column]["array"] for b in batch]
            inputs = self.fe(audio, sampling_rate=16000, return_tensors="pt")
            feats = pad_or_trim_mels(inputs.input_features, 3000)

        labels = self.tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_text_len,
        ).input_ids
        labels[labels == self.tok.pad_token_id] = -100

        return {"input_features": feats, "labels": labels}


# -----------------------------
# ACFT objective
# -----------------------------

def forward_hidden_states(
    model: WhisperForConditionalGeneration,
    input_features: torch.Tensor,
    decoder_input_ids: torch.Tensor,
) -> torch.Tensor:
    out = model.model(
        input_features=input_features,
        decoder_input_ids=decoder_input_ids,
        output_hidden_states=True,
        return_dict=True,
    )
    return out.decoder_hidden_states[-1]


def make_decoder_inputs(labels: torch.Tensor, bos_token_id: int) -> torch.Tensor:
    x = labels.clone()
    x[x == -100] = 0
    bos = torch.full((x.size(0), 1), bos_token_id, dtype=x.dtype, device=x.device)
    return torch.cat([bos, x[:, :-1]], dim=1)


# -----------------------------
# Training
# -----------------------------

def save_checkpoint(model: WhisperForConditionalGeneration, processor: WhisperProcessor, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)


def save_checkpoint_full(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    out_dir: Path,
    optim: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler._LRScheduler,
    step: int,
) -> None:
    save_checkpoint(model, processor, out_dir)
    _save_training_state(out_dir, optim, sched, step)


@torch.no_grad()
def eval_loss(
    model_ref: WhisperForConditionalGeneration,
    model_train: WhisperForConditionalGeneration,
    dl: DataLoader,
    device: torch.device,
    processor: WhisperProcessor,
    cfg: TrainConfig,
) -> float:
    model_ref.eval()
    model_train.eval()
    losses: List[float] = []

    for batch in dl:
        feats = batch["input_features"].to(device)
        labels = batch["labels"].to(device)
        dec_in = make_decoder_inputs(labels, processor.tokenizer.bos_token_id)

        h_full = forward_hidden_states(model_ref, feats, dec_in)

        min_frames = seconds_to_mel_frames(cfg.min_audio_seconds)
        max_frames = seconds_to_mel_frames(cfg.max_audio_seconds)
        target_frames = int(torch.randint(min_frames, max_frames + 1, (1,), device=device).item())
        feats_short = crop_mels(feats, target_frames)

        h_short = forward_hidden_states(model_train, feats_short, dec_in)
        losses.append(torch.mean((h_short - h_full) ** 2).item())

    model_ref.train()
    model_train.train()
    return float(np.mean(losses)) if losses else float("nan")


def _autocast_ctx(device: torch.device, fp16: bool):
    if not fp16:
        return torch.autocast(device_type=device.type, enabled=False)
    if device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=torch.float16)
    if device.type == "mps":
        return torch.autocast(device_type="mps", dtype=torch.float16)
    return torch.autocast(device_type=device.type, enabled=False)


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    device = pick_device(cfg.device)
    print(f"Using device: {device} (type={device.type})")

    out_root = Path(cfg.output_dir) / cfg.run_name
    out_root.mkdir(parents=True, exist_ok=True)

    # Resume logic
    resume_path: Optional[Path] = None
    if cfg.resume_from:
        resume_path = Path(cfg.resume_from)
    elif cfg.resume_latest:
        resume_path = _find_latest_checkpoint(out_root)

    resume_step = 0
    if resume_path is not None and resume_path.exists():
        print(f"Resuming from checkpoint: {resume_path}")
        processor = WhisperProcessor.from_pretrained(str(resume_path))
        model_train = WhisperForConditionalGeneration.from_pretrained(str(resume_path))
    else:
        processor = WhisperProcessor.from_pretrained(cfg.base_model)
        model_train = WhisperForConditionalGeneration.from_pretrained(cfg.base_model)

    # Reference model always from base (frozen)
    model_ref = WhisperForConditionalGeneration.from_pretrained(cfg.base_model)

    model_ref.to(device)
    model_train.to(device)

    for p in model_ref.parameters():
        p.requires_grad_(False)
    model_ref.eval()

    train_ds, eval_ds = load_speech_dataset(cfg)
    train_ds = maybe_precompute_features(train_ds, processor, cfg)
    if eval_ds is not None:
        eval_ds = maybe_precompute_features(eval_ds, processor, cfg)

    collate = Collator(processor, cfg)

    # If precompute_features=true, collator is cheap and workers add overhead on macOS.
    nw = 0 if cfg.precompute_features else max(0, int(cfg.num_workers))

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=nw,
        collate_fn=collate,
        pin_memory=False,
        persistent_workers=(cfg.persistent_workers and nw > 0),
        prefetch_factor=(cfg.prefetch_factor if nw > 0 else None),
    )

    eval_dl = None
    if eval_ds is not None:
        eval_dl = DataLoader(
            eval_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=nw,
            collate_fn=collate,
            pin_memory=False,
            persistent_workers=(cfg.persistent_workers and nw > 0),
            prefetch_factor=(cfg.prefetch_factor if nw > 0 else None),
        )

    step = 0

    params = [p for p in model_train.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    sched = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.max_steps,
    )

    # Restore optimizer/scheduler state if available
    if resume_path is not None and resume_path.exists():
        resume_step = _load_training_state(resume_path, optim, sched)
        if resume_step > 0:
            step = resume_step
            print(f"Resumed training state at step={step}")

    accum = 0
    running: List[float] = []
    start = time.time()

    model_train.train()

    it = iter(train_dl)
    while step < cfg.max_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_dl)
            batch = next(it)

        feats = batch["input_features"].to(device)
        labels = batch["labels"].to(device)
        dec_in = make_decoder_inputs(labels, processor.tokenizer.bos_token_id)

        with torch.no_grad():
            with _autocast_ctx(device, cfg.fp16):
                h_full = forward_hidden_states(model_ref, feats, dec_in)

        min_frames = seconds_to_mel_frames(cfg.min_audio_seconds)
        max_frames = seconds_to_mel_frames(cfg.max_audio_seconds)
        target_frames = int(torch.randint(min_frames, max_frames + 1, (1,), device=device).item())
        feats_short = crop_mels(feats, target_frames)

        with _autocast_ctx(device, cfg.fp16):
            h_short = forward_hidden_states(model_train, feats_short, dec_in)
            loss = torch.mean((h_short - h_full) ** 2)

        (loss / cfg.grad_accum_steps).backward()

        running.append(float(loss.item()))
        accum += 1

        if accum >= cfg.grad_accum_steps:
            optim.step()
            optim.zero_grad(set_to_none=True)
            sched.step()

            step += 1
            accum = 0

            if step % cfg.log_every_steps == 0:
                dt = time.time() - start
                avg = float(np.mean(running[-cfg.log_every_steps :]))
                lr = sched.get_last_lr()[0]
                print(f"[{step}/{cfg.max_steps}] loss={avg:.6f} lr={lr:.2e} time={dt:.1f}s")

            if eval_dl is not None and step % cfg.eval_every_steps == 0:
                ev = eval_loss(model_ref, model_train, eval_dl, device, processor, cfg)
                print(f"[{step}] eval_mse={ev:.6f}")

            if step % cfg.save_every_steps == 0:
                ckpt = out_root / f"step-{step}"
                save_checkpoint_full(model_train, processor, ckpt, optim, sched, step)
                print(f"Saved checkpoint: {ckpt}")

    final_dir = out_root / "final"
    save_checkpoint_full(model_train, processor, final_dir, optim, sched, step)
    print(f"Saved final checkpoint: {final_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/hebrew_tiny_acft.yaml")
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint dir to resume from (e.g. outputs/run/step-500)")
    ap.add_argument("--no-resume-latest", action="store_true", help="Disable auto-resume from latest step-* checkpoint")
    args = ap.parse_args()
    cfg = load_config(Path(args.config))
    if args.resume is not None:
        cfg = dataclasses.replace(cfg, resume_from=args.resume)
    if args.no_resume_latest:
        cfg = dataclasses.replace(cfg, resume_latest=False)
    train(cfg)


if __name__ == "__main__":
    main()
