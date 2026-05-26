#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest

python -m ar_gstd.generate_corruptions \
  --input data/meeting_summaries_seed.jsonl \
  --output artifacts/corruptions.jsonl \
  --beta 0.55 \
  --seed 7

python -m ar_gstd.generate_corruptions \
  --input data/meeting_summaries_seed.jsonl \
  --output artifacts/corruption_table.md \
  --format markdown \
  --limit 8 \
  --beta 0.55 \
  --seed 7

python -m ar_gstd.evaluate_corruptions \
  --input artifacts/corruptions.jsonl \
  --output artifacts/corruption_metrics.md

printf '\nWrote artifacts/corruptions.jsonl\n'
printf 'Wrote artifacts/corruption_table.md\n'
printf 'Wrote artifacts/corruption_metrics.md\n'
