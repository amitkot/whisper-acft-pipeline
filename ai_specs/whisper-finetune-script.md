# Whisper Fine-Tuning Script

## Context

We have an ACFT training script (`acft_train.py`) that teaches Whisper to handle short audio contexts, but no standard fine-tuning script to improve the model's actual Hebrew recognition. The goal is: fine-tune `openai/whisper-tiny` on `ivrit-ai/whisper-training` (~400h Hebrew) to improve Hebrew WER, then optionally run ACFT as a second pass for FUTO keyboard optimization.

Constraints: 10GB disk limit (streaming required), macOS M4 48GB MPS, resumable.

## New Files

- `scripts/finetune.py` — supervised fine-tuning with Seq2SeqTrainer
- `configs/hebrew_tiny_finetune.yaml` — config for whisper-tiny + ivrit.ai

## Modified Files

- `scripts/pipeline.py` — add `--finetune-config` flag for finetune → ACFT → convert → quantize

## Design

### 1. `scripts/finetune.py`

**Config** (dataclass + YAML, same pattern as acft_train.py):

```python
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
    text_column: str = "sentence"       # verify actual column name
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
    max_eval_samples: Optional[int] = 200  # cap streaming eval

    # Generation (for WER eval)
    generation_max_length: int = 225
    predict_with_generate: bool = True

    # Runtime
    device: str = "mps"
    seed: int = 0
    dataloader_num_workers: int = 0     # MPS + macOS = 0 workers
```

**Data pipeline**:

```python
def load_streaming_dataset(cfg):
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config,
                      split=cfg.train_split, streaming=True, trust_remote_code=True)
    ds = ds.shuffle(buffer_size=cfg.shuffle_buffer, seed=cfg.seed)
    # eval: load non-streaming, cap at max_eval_samples
    eval_ds = load_dataset(..., split=cfg.eval_split, streaming=False)
    if cfg.max_eval_samples:
        eval_ds = eval_ds.select(range(min(cfg.max_eval_samples, len(eval_ds))))
    return ds, eval_ds

def prepare_dataset(example, processor, cfg):
    audio = example[cfg.audio_column]
    input_features = processor.feature_extractor(
        audio["array"], sampling_rate=16000, return_tensors="pt"
    ).input_features[0]
    labels = processor.tokenizer(example[cfg.text_column]).input_ids
    return {"input_features": input_features, "labels": labels}
```

- Apply via `ds.map(prepare_dataset)` — works with streaming
- Collator: `DataCollatorSpeechSeq2SeqWithPadding` — pads labels, masks with -100

**Training**:

```python
processor = WhisperProcessor.from_pretrained(cfg.base_model)
model = WhisperForConditionalGeneration.from_pretrained(cfg.base_model)
model.generation_config.language = cfg.language
model.generation_config.task = cfg.task
model.generation_config.forced_decoder_ids = None

training_args = Seq2SeqTrainingArguments(
    output_dir=f"{cfg.output_dir}/{cfg.run_name}",
    max_steps=cfg.max_steps,
    per_device_train_batch_size=cfg.per_device_train_batch_size,
    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    learning_rate=cfg.learning_rate,
    fp16=cfg.fp16,
    gradient_checkpointing=cfg.gradient_checkpointing,
    eval_strategy="steps",
    eval_steps=cfg.eval_steps,
    save_steps=cfg.save_steps,
    save_total_limit=cfg.save_total_limit,
    predict_with_generate=True,
    generation_max_length=cfg.generation_max_length,
    logging_steps=cfg.logging_steps,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
    compute_metrics=compute_wer,
    processing_class=processor.feature_extractor,
)
trainer.train(resume_from_checkpoint=find_checkpoint(cfg))
```

**WER metric**:

```python
def compute_wer(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": jiwer.wer(label_str, pred_str)}
```

**CLI**: Same pattern as acft_train.py — `--config`, `--resume`, `--no-resume-latest`.

### 2. Config: `configs/hebrew_tiny_finetune.yaml`

```yaml
run_name: hebrew_tiny_ft
base_model: openai/whisper-tiny
output_dir: outputs
language: he

dataset_name: ivrit-ai/whisper-training
text_column: sentence          # verify actual column name
audio_column: audio
streaming: true

device: mps
fp16: true
seed: 0
```

All training hyperparams (max_steps, lr, batch_size, etc.) come from FinetuneConfig defaults.

### 3. Pipeline integration (`scripts/pipeline.py`)

Add optional `--finetune-config` flag:

```
uv run python scripts/pipeline.py \
  --finetune-config configs/hebrew_tiny_finetune.yaml \
  --config configs/hebrew_tiny_acft.yaml
```

Flow:
1. If `--finetune-config` given: run finetune.py first
2. Update ACFT config's `base_model` to point at fine-tuned checkpoint
3. Run ACFT (existing acft_train.py)
4. Convert → quantize (existing logic)

If `--finetune-config` is omitted, pipeline works exactly as before (ACFT only).

### 4. Disk budget estimate

| Item | Size | Notes |
|------|------|-------|
| whisper-tiny model | ~150 MB | Weights + tokenizer |
| 1 checkpoint (save_total_limit=1) | ~150 MB | Trainer auto-cleans |
| HF streaming cache | ~200 MB | Temporary shard buffer |
| Final model | ~150 MB | Trainer saves best |
| GGML output | ~100 MB | After convert + quantize |
| **Total** | **~750 MB** | Well within 10GB |

## Implementation Steps

1. Create `scripts/finetune.py` with FinetuneConfig, data pipeline, training setup
2. Create `configs/hebrew_tiny_finetune.yaml`
3. Test: verify dataset loads in streaming mode, confirm column names
4. Test: run ~50 steps to verify training loop works on MPS
5. Update `scripts/pipeline.py` with `--finetune-config` support
6. Update README with fine-tuning docs

## Verification

1. `uv run python scripts/finetune.py --config configs/hebrew_tiny_finetune.yaml` — should start training, show loss decreasing, eval WER
2. Stop and re-run — should resume from checkpoint
3. Full pipeline: `uv run python scripts/pipeline.py --finetune-config configs/hebrew_tiny_finetune.yaml --config configs/hebrew_tiny_acft.yaml`

## Open Questions

- Verify `ivrit-ai/whisper-training` column names (text_column). Need to check actual dataset schema.
- Dataset may be gated — user needs to `huggingface-cli login` and accept terms first.
