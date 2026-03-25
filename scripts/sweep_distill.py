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
"""Sweep distillation hyperparameters to find the best alpha/temperature.

Runs short distillation experiments (1000 steps each) with different
alpha and temperature values, evaluates WER, and prints a summary table.
Checkpoints are kept for continuing the best run.

Usage:
  uv run python scripts/sweep_distill.py --config configs/hebrew_tiny_distill.yaml
  uv run python scripts/sweep_distill.py --config configs/hebrew_tiny_distill.yaml --steps 500
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


DEFAULT_GRID = {
    "alpha": [0.5, 0.7, 0.9],
    "temperature": [1.0, 2.0],
}


def run_one(config_path: Path, alpha: float, temp: float, steps: int, output_base: Path):
    """Run a single distillation experiment and return eval WER."""
    run_name = f"_sweep_a{alpha}_t{temp}"
    run_dir = output_base / run_name

    # Create temporary config — just find the best alpha/temperature.
    # No checkpoint reuse (LR schedule wouldn't match the full run).
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cfg["alpha"] = alpha
    cfg["temperature"] = temp
    cfg["max_steps"] = steps
    cfg["run_name"] = run_name
    cfg["output_dir"] = str(output_base)
    cfg["save_steps"] = steps
    cfg["eval_steps"] = steps
    cfg["save_total_limit"] = 1
    cfg["logging_steps"] = 100
    cfg["warmup_steps"] = min(cfg.get("warmup_steps", 500), steps // 4)

    tmp_config = output_base / f"{run_name}_config.yaml"
    tmp_config.parent.mkdir(parents=True, exist_ok=True)
    tmp_config.write_text(yaml.dump(cfg))

    print(f"\n{'='*60}")
    print(f"  alpha={alpha}, temperature={temp}, steps={steps}")
    print(f"  run_dir: {run_dir}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "scripts/distill.py", "--config", str(tmp_config)],
        capture_output=False,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        return {"alpha": alpha, "temperature": temp, "wer": None, "elapsed": elapsed, "error": True}

    # Read eval WER from training log
    log_path = run_dir / "training_log.jsonl"
    wer = None
    if log_path.exists():
        for line in log_path.read_text().strip().split("\n"):
            entry = json.loads(line)
            if "eval_wer" in entry:
                wer = entry["eval_wer"]

    print(f"  WER: {wer:.4f}" if wer else "  WER: not available")
    print(f"  Time: {elapsed/60:.1f} min")

    return {"alpha": alpha, "temperature": temp, "wer": wer, "elapsed": elapsed, "error": False}


def main():
    ap = argparse.ArgumentParser(description="Sweep distillation hyperparameters")
    ap.add_argument("--config", type=str, required=True,
                    help="Base distillation config (student, teacher, dataset settings)")
    ap.add_argument("--steps", type=int, default=1000,
                    help="Steps per sweep run (default: 1000)")
    ap.add_argument("--output-dir", type=str, default="outputs/sweep",
                    help="Output directory for sweep runs (default: outputs/sweep)")
    ap.add_argument("--alphas", type=str, default="0.5,0.7,0.9",
                    help="Comma-separated alpha values to test")
    ap.add_argument("--temperatures", type=str, default="1.0,2.0",
                    help="Comma-separated temperature values to test")
    args = ap.parse_args()

    config_path = Path(args.config)
    output_base = Path(args.output_dir)
    alphas = [float(x) for x in args.alphas.split(",")]
    temps = [float(x) for x in args.temperatures.split(",")]

    print(f"Sweep: {len(alphas)} alphas × {len(temps)} temperatures = {len(alphas)*len(temps)} runs")
    print(f"Steps per run: {args.steps}")
    print(f"Base config: {config_path}")
    print(f"Output: {output_base}")

    results = []
    for alpha in alphas:
        for temp in temps:
            r = run_one(config_path, alpha, temp, args.steps, output_base)
            results.append(r)

    # Summary table
    print(f"\n{'='*60}")
    print(f"  SWEEP RESULTS ({args.steps} steps each)")
    print(f"{'='*60}")
    print(f"  {'Alpha':>6}  {'Temp':>5}  {'WER':>8}  {'Time':>8}")
    print(f"  {'-'*35}")

    valid = [r for r in results if r["wer"] is not None]
    for r in sorted(valid, key=lambda x: x["wer"]):
        marker = " <-- best" if r == valid[0] else ""
        print(f"  {r['alpha']:>6.2f}  {r['temperature']:>5.1f}  {r['wer']:>8.4f}  {r['elapsed']/60:>7.1f}m{marker}")

    for r in results:
        if r["error"]:
            print(f"  {r['alpha']:>6.2f}  {r['temperature']:>5.1f}  {'FAILED':>8}  {r['elapsed']/60:>7.1f}m")

    if valid:
        best = min(valid, key=lambda x: x["wer"])
        print(f"\n  Best: alpha={best['alpha']}, temperature={best['temperature']}, WER={best['wer']:.4f}")
        print(f"  Checkpoint: {output_base}/_sweep_a{best['alpha']}_t{best['temperature']}/")
        print(f"\n  To continue this run for the full training:")
        print(f"  1. Copy checkpoint to the target output dir")
        print(f"  2. Update config with alpha={best['alpha']}, temperature={best['temperature']}")
        print(f"  3. Run: uv run python scripts/distill.py --config <updated_config>")

    # Save results
    results_path = output_base / "sweep_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Full results saved to: {results_path}")


if __name__ == "__main__":
    main()
