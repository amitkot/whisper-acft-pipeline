#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import yaml


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ensure_tokenizer_legacy_files(hf_dir: Path) -> None:
    """
    whisper.cpp HF converter expects:
      - vocab.json
      - merges.txt
      - added_tokens.json (sometimes)
    Your HF checkpoint has tokenizer.json, so we derive legacy files if missing.
    """
    tok_path = hf_dir / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"Missing {tok_path}")

    tok = json.loads(tok_path.read_text(encoding="utf-8"))
    model = tok.get("model", {})
    vocab = model.get("vocab")
    merges = model.get("merges")

    # vocab.json
    vocab_path = hf_dir / "vocab.json"
    if not vocab_path.exists():
        if not isinstance(vocab, dict):
            raise RuntimeError("tokenizer.json missing model.vocab in expected format")
        vocab_path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {vocab_path}")

    # merges.txt
    merges_path = hf_dir / "merges.txt"
    if not merges_path.exists():
        if not isinstance(merges, list):
            raise RuntimeError("tokenizer.json missing model.merges in expected format")

        lines: list[str] = ["#version: 0.2"]
        for m in merges:
            if isinstance(m, str):
                lines.append(m)
            elif isinstance(m, (list, tuple)) and len(m) == 2:
                lines.append(f"{m[0]} {m[1]}")
            else:
                raise RuntimeError(f"Unexpected merge entry: {m!r}")

        merges_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {merges_path}")

    # added_tokens.json
    added_path = hf_dir / "added_tokens.json"
    if not added_path.exists():
        added_path.write_text("{}", encoding="utf-8")
        print(f"Wrote {added_path}")


def resolve_repo_root() -> Path:
    # pipeline.py is expected to be in <repo>/scripts/pipeline.py
    return Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description="ACFT -> ggml -> quantize pipeline")
    ap.add_argument("--config", default="configs/hebrew_tiny_acft.yaml", help="ACFT config yaml")
    ap.add_argument(
        "--final-dir",
        default=None,
        help="HF final checkpoint directory (default: <output_dir>/<run_name>/final from config)",
    )
    ap.add_argument("--out-dir", default=None, help="Output directory for ggml bins (default: out/<run_name>)")
    ap.add_argument(
        "--base-name",
        default=None,
        help="Base filename (without extension) for produced ggml files (default: ggml-<run_name>)",
    )
    ap.add_argument(
        "--quants",
        default="q5_0,q8_0",
        help="Comma-separated quant targets (e.g. q5_0,q4_1,q8_0). Avoid q6_k if FUTO rejects it.",
    )
    ap.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip ACFT training step (assumes final checkpoint already exists).",
    )
    ap.add_argument(
        "--skip-quant",
        action="store_true",
        help="Skip quantization step.",
    )
    ap.add_argument(
        "--resume",
        default=None,
        help="Resume from specific checkpoint dir (passed through to acft_train.py --resume).",
    )
    ap.add_argument(
        "--no-resume-latest",
        action="store_true",
        help="Disable auto-resume behavior (passed through to acft_train.py).",
    )
    args = ap.parse_args()

    repo = resolve_repo_root()
    cfg_path = (repo / args.config).resolve()

    # Load config to derive defaults
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    run_name = cfg.get("run_name", "default")
    output_dir = cfg.get("output_dir", "outputs")

    final_dir = (repo / (args.final_dir or f"{output_dir}/{run_name}/final")).resolve()
    out_dir = (repo / (args.out_dir or f"out/{run_name}")).resolve()
    base_name = args.base_name or f"ggml-{run_name}"

    whisper_cpp = (repo / "external" / "whisper.cpp").resolve()
    openai_whisper = (repo / "external" / "whisper").resolve()
    converter = whisper_cpp / "models" / "convert-h5-to-ggml.py"
    quantizer = whisper_cpp / "build" / "bin" / "whisper-quantize"

    # 1) Train
    if not args.skip_train:
        cmd = ["uv", "run", "python", str(repo / "scripts" / "acft_train.py"), "--config", str(cfg_path)]
        if args.resume:
            cmd += ["--resume", args.resume]
        if args.no_resume_latest:
            cmd += ["--no-resume-latest"]
        run(cmd, cwd=repo)

    # 2) Validate HF final checkpoint
    if not final_dir.exists():
        raise FileNotFoundError(f"Final checkpoint dir not found: {final_dir}")

    required = ["config.json", "model.safetensors", "tokenizer.json"]
    missing = [x for x in required if not (final_dir / x).exists()]
    if missing:
        raise FileNotFoundError(f"Final dir missing files: {missing} in {final_dir}")

    # 3) Tokenizer bridge files for whisper.cpp
    ensure_tokenizer_legacy_files(final_dir)

    # 4) Convert HF -> ggml
    out_dir.mkdir(parents=True, exist_ok=True)
    run(
        ["uv", "run", "python", str(converter), str(final_dir), str(openai_whisper), str(out_dir)],
        cwd=repo,
    )

    # whisper.cpp converter produces out/ggml-model.bin
    ggml_model = out_dir / "ggml-model.bin"
    if not ggml_model.exists():
        raise FileNotFoundError(f"Converter did not produce {ggml_model}")

    base_bin = out_dir / f"{base_name}.bin"
    shutil.move(str(ggml_model), str(base_bin))
    print(f"\nCreated base ggml: {base_bin} ({base_bin.stat().st_size / (1024*1024):.1f} MB)")

    # 5) Quantize
    produced: list[Path] = [base_bin]
    if not args.skip_quant:
        if not quantizer.exists():
            raise FileNotFoundError(f"Quantizer not found: {quantizer}. Build whisper.cpp first.")
        quants = [q.strip() for q in args.quants.split(",") if q.strip()]
        for q in quants:
            out_q = out_dir / f"{base_name}-{q}.bin"
            run([str(quantizer), str(base_bin), str(out_q), q], cwd=repo)
            produced.append(out_q)

    print("\nDone. Artifacts:")
    for p in produced:
        print(f"  - {p}  ({p.stat().st_size / (1024*1024):.1f} MB)")


if __name__ == "__main__":
    main()
