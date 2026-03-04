#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Initializing submodules..."
git submodule update --init --recursive

echo "==> Building whisper.cpp..."
cmake -S external/whisper.cpp -B external/whisper.cpp/build
cmake --build external/whisper.cpp/build -j

echo "==> Installing Python dependencies..."
uv sync

echo "==> Done! Run training with:"
echo "    uv run python scripts/acft_train.py --config configs/hebrew_tiny_acft.yaml"
