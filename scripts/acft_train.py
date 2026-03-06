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
"""ACFT finetuning for Whisper, aligned with the FUTO reference implementation.

Key differences from vanilla approach:
- Partial encoder with truncated positional embeddings (not crop+pad)
- Compares ALL decoder hidden states, not just the last
- Audio context proportional to actual duration with +/-jitter
- Epoch-based training (8 epochs, batch_size=1)

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
import shutil
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

    # ACFT (FUTO-aligned defaults)
    max_audio_duration: float = 29.0     # skip audio longer than this
    acft_jitter_frames: int = 64         # +/-jitter on audio context

    # Optimization (FUTO-aligned defaults)
    seed: int = 1337
    device: str = "auto"   # auto|mps|cpu|cuda
    fp16: bool = True      # for MPS: uses autocast float16
    batch_size: int = 1                  # FUTO: per-example processing
    grad_accum_steps: int = 1            # FUTO: immediate updates
    lr: float = 1.0e-6                   # FUTO: 1e-6 for all model sizes
    weight_decay: float = 0.01
    warmup_steps: int = 200
    num_epochs: int = 8                  # FUTO: 8 epochs
    max_steps: int = 0                   # 0 = use num_epochs only
    eval_every_steps: int = 250
    save_every_steps: int = 500

    # DataLoader
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True

    # Precompute features (big speed win)
    precompute_features: bool = True

    # Resume
    resume_from: Optional[str] = None
    resume_latest: bool = True

    # Misc
    log_every_steps: int = 25
    max_text_len: int = 256

    # DEPRECATED -- kept for backward compat, unused in new code
    min_audio_seconds: float = 1.5
    max_audio_seconds: float = 12.0


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


def _save_training_state(out_dir: Path, step: int, epoch: int = 0) -> None:
    state = {
        "step": step,
        "epoch": epoch,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
        },
    }
    torch.save(state, out_dir / "training_state.pt")


def _load_training_state(path: Path, steps_per_epoch: int = 0) -> Tuple[int, int]:
    """Returns (step, epoch). Reconstructs from dir name if state file is missing/corrupt."""
    ts = path / "training_state.pt"

    def _from_dir_name() -> Tuple[int, int]:
        m = re.match(r"step-(\d+)$", path.name)
        s = int(m.group(1)) if m else 0
        e = s // steps_per_epoch if steps_per_epoch > 0 else 0
        return s, e

    if not ts.exists():
        return _from_dir_name()

    try:
        state = torch.load(ts, map_location="cpu", weights_only=False)
    except RuntimeError:
        print("WARNING: training_state.pt is corrupt, recovering from dir name")
        return _from_dir_name()

    step = int(state.get("step", 0))
    epoch = int(state.get("epoch", 0))

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

    return step, epoch


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
        ex["audio_duration"] = len(arr) / 16000.0
        feats = fe(arr, sampling_rate=16000, return_tensors="pt").input_features[0]  # (80, T)
        feats = pad_or_trim_mels(feats, 3000)
        ex["input_features"] = feats.to(torch.float32).numpy()
        return ex

    num_proc = max(1, min(cfg.num_workers, os.cpu_count() or 1))
    ds2 = ds.map(_map, num_proc=num_proc, desc="Precomputing mel features (one-time)")
    keep = [cfg.text_column, "input_features", "audio_duration"]
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
            durations = torch.tensor([b["audio_duration"] for b in batch], dtype=torch.float32)
        else:
            audio_arrays = [b[self.cfg.audio_column]["array"] for b in batch]
            durations = torch.tensor([len(a) / 16000.0 for a in audio_arrays], dtype=torch.float32)
            inputs = self.fe(audio_arrays, sampling_rate=16000, return_tensors="pt")
            feats = pad_or_trim_mels(inputs.input_features, 3000)

        labels = self.tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_text_len,
        ).input_ids
        labels[labels == self.tok.pad_token_id] = -100

        return {"input_features": feats, "labels": labels, "audio_durations": durations}


# -----------------------------
# ACFT objective (FUTO-aligned)
# -----------------------------

def compute_n_ctx(audio_duration: float, cfg: TrainConfig, jitter: bool = True) -> int:
    """Compute audio context frames proportional to actual duration, with optional jitter."""
    n_ctx = round(50.0 * audio_duration)  # 50 frames/sec = 1500/30s
    n_ctx = max(1, min(n_ctx, 1500))
    if jitter:
        max_j = min(cfg.acft_jitter_frames, n_ctx // 3)
        if max_j > 0:
            n_ctx += random.randint(-max_j, max_j)
            n_ctx = max(1, min(n_ctx, 1500))
    return n_ctx


def compute_partial_encoder(
    model: WhisperForConditionalGeneration,
    input_features: torch.Tensor,
    n_audio_ctx: int,
) -> torch.Tensor:
    """Run encoder with truncated positional embeddings for shorter context.

    Returns encoder hidden states of shape (B, n_audio_ctx, d_model).
    Falls back to standard encoder when n_audio_ctx >= 1500.
    """
    encoder = model.model.encoder

    if n_audio_ctx >= 1500:
        return encoder(input_features=input_features).last_hidden_state

    mel_frames = 2 * n_audio_ctx  # conv2 stride=2
    mel_trimmed = input_features[:, :, :mel_frames]

    hidden_states = F.gelu(encoder.conv1(mel_trimmed))
    hidden_states = F.gelu(encoder.conv2(hidden_states))
    hidden_states = hidden_states.permute(0, 2, 1)  # (B, n_audio_ctx, d_model)

    hidden_states = hidden_states + encoder.embed_positions.weight[:n_audio_ctx]

    hidden_states = F.dropout(hidden_states, p=encoder.dropout, training=encoder.training)

    for layer in encoder.layers:
        if encoder.training:
            dropout_probability = torch.rand([])
            if dropout_probability < encoder.layerdrop:
                continue
        hidden_states = layer(hidden_states, attention_mask=None)[0]

    hidden_states = encoder.layer_norm(hidden_states)
    return hidden_states


def forward_decoder_all_hidden_states(
    model: WhisperForConditionalGeneration,
    encoder_hidden_states: torch.Tensor,
    decoder_input_ids: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """Run decoder and return ALL hidden states (embedding + each layer)."""
    out = model.model.decoder(
        input_ids=decoder_input_ids,
        encoder_hidden_states=encoder_hidden_states,
        output_hidden_states=True,
        return_dict=True,
    )
    return out.hidden_states


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
    step: int,
    epoch: int = 0,
) -> None:
    save_checkpoint(model, processor, out_dir)
    _save_training_state(out_dir, step, epoch)


def _cleanup_old_checkpoints(out_root: Path, keep: Path) -> None:
    """Delete all step-* checkpoint dirs except `keep`."""
    for p in out_root.iterdir():
        if not p.is_dir() or not re.match(r"step-\d+$", p.name):
            continue
        if p.resolve() == keep.resolve():
            continue
        shutil.rmtree(p)
        print(f"Removed old checkpoint: {p}")


@torch.no_grad()
def eval_loss(
    model_ref: WhisperForConditionalGeneration,
    model_train: WhisperForConditionalGeneration,
    dl: DataLoader,
    device: torch.device,
    processor: WhisperProcessor,
    cfg: TrainConfig,
) -> float:
    model_train.eval()
    losses: List[float] = []

    for batch in dl:
        feats = batch["input_features"].to(device)
        labels = batch["labels"].to(device)
        durations = batch["audio_durations"]
        dec_in = make_decoder_inputs(labels, processor.tokenizer.bos_token_id)

        for i in range(feats.size(0)):
            duration = durations[i].item()
            if duration > cfg.max_audio_duration:
                continue

            n_ctx = compute_n_ctx(duration, cfg, jitter=False)
            f_i = feats[i : i + 1]
            d_i = dec_in[i : i + 1]

            enc_full = model_ref.model.encoder(input_features=f_i).last_hidden_state
            h_full = forward_decoder_all_hidden_states(model_ref, enc_full, d_i)

            enc_partial = compute_partial_encoder(model_train, f_i, n_ctx)
            h_partial = forward_decoder_all_hidden_states(model_train, enc_partial, d_i)

            loss = F.mse_loss(torch.cat(h_partial, 0), torch.cat(h_full, 0))
            losses.append(loss.item())

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
    if cfg.min_audio_seconds != 1.5 or cfg.max_audio_seconds != 12.0:
        print("WARNING: min_audio_seconds/max_audio_seconds are deprecated and ignored. "
              "Audio context is now computed from actual duration.")

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

    # Compute total training steps
    steps_per_epoch = len(train_ds) // cfg.grad_accum_steps
    total_steps = steps_per_epoch * cfg.num_epochs
    if cfg.max_steps > 0:
        total_steps = cfg.max_steps

    print(f"Dataset size: {len(train_ds)}, epochs: {cfg.num_epochs}, "
          f"est. total steps: {total_steps}")

    params = [p for p in model_train.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    sched = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )

    step = 0
    start_epoch = 0

    if resume_path is not None and resume_path.exists():
        resume_step, resume_epoch = _load_training_state(resume_path, steps_per_epoch)
        if resume_step > 0:
            step = resume_step
            start_epoch = resume_epoch
            print(f"Resumed training state at step={step}, epoch={start_epoch}")

    running: List[float] = []
    start = time.time()
    accum = 0
    done = False
    epoch = start_epoch

    model_train.train()

    for epoch in range(start_epoch, cfg.num_epochs):
        if done:
            break
        print(f"--- Epoch {epoch + 1}/{cfg.num_epochs} ---")

        for batch in train_dl:
            if cfg.max_steps > 0 and step >= cfg.max_steps:
                done = True
                break

            duration = batch["audio_durations"][0].item()
            if duration > cfg.max_audio_duration:
                continue

            feats = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            dec_in = make_decoder_inputs(labels, processor.tokenizer.bos_token_id)

            n_ctx = compute_n_ctx(duration, cfg, jitter=True)

            with torch.no_grad():
                with _autocast_ctx(device, cfg.fp16):
                    enc_full = model_ref.model.encoder(input_features=feats).last_hidden_state
                    h_full = forward_decoder_all_hidden_states(model_ref, enc_full, dec_in)

            with _autocast_ctx(device, cfg.fp16):
                enc_partial = compute_partial_encoder(model_train, feats, n_ctx)
                h_partial = forward_decoder_all_hidden_states(model_train, enc_partial, dec_in)
                loss = F.mse_loss(torch.cat(h_partial, 0), torch.cat(h_full, 0))

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
                    avg = float(np.mean(running[-cfg.log_every_steps:]))
                    lr = sched.get_last_lr()[0]
                    print(f"[step {step}] loss={avg:.6f} lr={lr:.2e} epoch={epoch+1} time={dt:.1f}s")

                if eval_dl is not None and step % cfg.eval_every_steps == 0:
                    ev = eval_loss(model_ref, model_train, eval_dl, device, processor, cfg)
                    print(f"[step {step}] eval_mse={ev:.6f}")

                if step % cfg.save_every_steps == 0:
                    ckpt = out_root / f"step-{step}"
                    save_checkpoint_full(model_train, processor, ckpt, step, epoch)
                    print(f"Saved checkpoint: {ckpt}")
                    _cleanup_old_checkpoints(out_root, keep=ckpt)

    final_dir = out_root / "final"
    save_checkpoint_full(model_train, processor, final_dir, step, epoch)
    _cleanup_old_checkpoints(out_root, keep=final_dir)
    print(f"Saved final checkpoint: {final_dir} (step={step})")


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
