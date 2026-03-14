# Fix ACFT Training to Match FUTO Reference

## Context

The ACFT-trained model repeats sentences indefinitely on short audio input. This happens because the current training implementation **crops mel spectrograms then pads them back to full length** (3000 frames), so the encoder always sees full-length positional embeddings. The model never learns to handle genuinely shorter encoder context — which is the entire point of ACFT.

The FUTO reference implementation passes truly shorter input through the encoder with truncated positional embeddings, teaching the model to produce correct decoder behavior regardless of audio context length.

### Root Causes (6 differences from FUTO reference)

| # | Current | FUTO Reference |
|---|---------|----------------|
| 1 | Crop mels, pad back to 3000 frames — encoder always sees full positional embeddings | Custom partial encoder with truly shorter input and truncated positional embeddings |
| 2 | Compares only **last** decoder hidden state | Compares **all** decoder hidden states (all layers concatenated) |
| 3 | Random crop length between `min_audio_seconds` and `max_audio_seconds` | Audio context proportional to **actual duration**: `n_ctx = round(50 * seconds)` |
| 4 | No context jitter | ±64 frame random jitter (capped at n_ctx//3) |
| 5 | Step-based: `max_steps=2000` | Epoch-based: 8 epochs over full dataset |
| 6 | `batch_size=8`, `grad_accum=2` | `batch_size=1` (each example has unique n_ctx) |

### What doesn't need changing

- **Inference params** (compression_ratio_threshold, condition_on_previous_text, entropy_thold) are runtime whisper.cpp settings, not embedded in model files. Can't be controlled per-model in FUTO keyboard. The fix is training-side.
- **Hyperparams per model size**: FUTO uses identical params for tiny/small/base — no model-specific tuning needed.
- **pipeline.py**: No changes needed (config changes are backward-compatible).

## Files to Modify

- `scripts/acft_train.py` — all training logic changes
- `configs/hebrew_tiny_acft.yaml` — remove deprecated fields
- `configs/hebrew_small_acft_mike_v4.yaml` — same
- `configs/hebrew_small_acft_eeizenman.yaml` — same

## Changes

### Step 1: Make TrainConfig defaults match FUTO values

The correct FUTO-aligned values become the **defaults in TrainConfig**, so any new config automatically gets them right without specifying every field.

```python
@dataclasses.dataclass
class TrainConfig:
    # ... existing fields ...

    # ACFT (FUTO-aligned defaults)
    max_audio_duration: float = 29.0     # skip audio longer than this
    acft_jitter_frames: int = 64         # ±jitter on audio context

    # Optimization (FUTO-aligned defaults)
    batch_size: int = 1                  # FUTO: per-example processing
    grad_accum_steps: int = 1            # FUTO: immediate updates
    lr: float = 1.0e-6                   # FUTO: 1e-6 for all model sizes
    num_epochs: int = 8                  # FUTO: 8 epochs
    max_steps: int = 0                   # 0 = use num_epochs only

    # DEPRECATED — kept for backward compat, unused in new code
    min_audio_seconds: float = 1.5
    max_audio_seconds: float = 12.0
```

Add validation in `train()` that warns if deprecated fields are set:

```python
if cfg.min_audio_seconds != 1.5 or cfg.max_audio_seconds != 12.0:
    print("WARNING: min_audio_seconds/max_audio_seconds are deprecated and ignored. "
          "Audio context is now computed from actual duration.")
```

### Step 2: Preserve audio duration during feature precomputation

**`maybe_precompute_features()`** (line 258): store `audio_duration` alongside mel features.

- In `_map()`: add `ex["audio_duration"] = len(arr) / 16000.0`
- Keep `audio_duration` column when dropping others
- In `Collator.__call__()`: collect durations into tensor, include in batch dict
- Non-precomputed path: compute `len(audio_array) / 16000.0`

### Step 3: Add `compute_partial_encoder()` — the core fix

New function replacing `crop_mels()`. Runs the HF WhisperEncoder sub-modules manually with shorter input:

1. Compute mel input length: `mel_frames = 2 * n_audio_ctx` (conv2 stride=2)
2. Trim mel to `mel_frames`
3. `F.gelu(encoder.conv1(mel))` → `F.gelu(encoder.conv2(...))` → permute to `(B, n_ctx, d_model)`
4. **Truncated positional embeddings**: `encoder.embed_positions.weight[:n_audio_ctx]`
5. Dropout → iterate through `encoder.layers` → `encoder.layer_norm`
6. Return `(B, n_audio_ctx, d_model)`

When `n_audio_ctx >= 1500` (full context), fall back to standard `encoder(input_features)` for efficiency.

Access pattern through HF model: `model.model.encoder.conv1`, `.conv2`, `.embed_positions`, `.layers`, `.layer_norm`, `.dropout`, `.layerdrop`.

### Step 4: Replace hidden state comparison

Replace `forward_hidden_states()` with `forward_decoder_all_hidden_states()`:

- Takes encoder hidden states + decoder input IDs
- Calls `model.model.decoder(encoder_hidden_states=..., output_hidden_states=True)`
- Returns tuple of ALL hidden states (embedding + each decoder layer)

Loss computation:
```python
loss = F.mse_loss(torch.cat(h_partial, 0), torch.cat(h_full, 0))
```

### Step 5: Compute audio context from actual duration with jitter

New `compute_n_ctx()` function:
```python
n_ctx = round(50.0 * audio_duration)  # 50 frames/sec = 1500/30
n_ctx = clamp(n_ctx, 1, 1500)
if jitter:
    max_j = min(cfg.acft_jitter_frames, n_ctx // 3)
    n_ctx += randint(-max_j, max_j)
    n_ctx = clamp(n_ctx, 1, 1500)
```

### Step 6: Rewrite training loop to epoch-based

- Outer loop: `for epoch in range(cfg.num_epochs)`
- Inner loop: iterate DataLoader batches
- For each batch with `batch_size=1`: compute `n_ctx` from `audio_durations[0]`, skip if `> max_audio_duration`
- Full encoder on reference model → partial encoder on training model → decoder hidden states comparison
- Compute total steps for scheduler: `(len(train_ds) // grad_accum_steps) * num_epochs`
- Keep existing checkpoint save/eval/log logic at step intervals

### Step 7: Update `eval_loss()` to use new approach

Same new functions, but with `jitter=False` during evaluation.

### Step 8: Remove deprecated code

- Remove `crop_mels()` function
- Remove `forward_hidden_states()` function
- Remove `seconds_to_mel_frames()` (no longer needed)

### Step 9: Slim down config files

Since TrainConfig defaults now match FUTO, configs only need model-specific overrides:

```yaml
# hebrew_tiny_acft.yaml
run_name: hebrew_tiny_acft
base_model: mike249/whisper-tiny-he-2
output_dir: outputs
dataset_name: google/fleurs
dataset_config: he_il
train_split: train
eval_split: validation
text_column: raw_transcription
audio_column: audio
device: mps
fp16: true
precompute_features: true
num_workers: 4
seed: 0
```

All ACFT-critical params (batch_size, lr, num_epochs, etc.) come from defaults. New configs created by copying this template will automatically be correct.

## Verification

1. Delete existing checkpoints (`outputs/`, `runs/`) — trained with wrong approach
2. Train tiny model first: `uv run python scripts/pipeline.py --config configs/hebrew_tiny_acft.yaml`
3. Check training logs — loss should decrease across 8 epochs
4. Pipeline auto-converts to ggml and quantizes
5. Load quantized model in FUTO keyboard
6. Test with short (~2-3s) and long (~15-20s) Hebrew utterances — short audio should no longer loop
