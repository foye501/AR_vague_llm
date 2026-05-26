#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
DENOISER_MODEL="${DENOISER_MODEL:-google/flan-t5-small}"
TOP_K="${TOP_K:-8}"
BETA="${BETA:-0.35}"
VARIANTS_PER_EXAMPLE="${VARIANTS_PER_EXAMPLE:-8}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"

python -m ar_gstd.build_transition_cache \
  --input data/meeting_summaries_seed.jsonl \
  --output artifacts/ar_transition_cache.jsonl \
  --teacher-model "$TEACHER_MODEL" \
  --top-k "$TOP_K" \
  --max-examples "$MAX_EXAMPLES" \
  --device auto

python -m ar_gstd.make_fixed_transition_cache \
  --input artifacts/ar_transition_cache.jsonl \
  --output artifacts/fixed_transition_cache.jsonl \
  --top-k "$TOP_K"

python -m ar_gstd.materialize_training_data \
  --cache artifacts/ar_transition_cache.jsonl \
  --output artifacts/train_pairs_ar.jsonl \
  --beta "$BETA" \
  --variants-per-example "$VARIANTS_PER_EXAMPLE" \
  --seed 7

python -m ar_gstd.materialize_training_data \
  --cache artifacts/fixed_transition_cache.jsonl \
  --output artifacts/train_pairs_fixed.jsonl \
  --beta "$BETA" \
  --variants-per-example "$VARIANTS_PER_EXAMPLE" \
  --seed 7

python -m ar_gstd.train_seq2seq_denoiser \
  --train-file artifacts/train_pairs_ar.jsonl \
  --output-dir artifacts/denoiser_ar \
  --model-name "$DENOISER_MODEL" \
  --epochs 3 \
  --batch-size 4

printf '\nFinished AR-top-k denoiser run.\n'
printf 'Transition cache: artifacts/ar_transition_cache.jsonl\n'
printf 'Fixed baseline cache: artifacts/fixed_transition_cache.jsonl\n'
printf 'AR training pairs: artifacts/train_pairs_ar.jsonl\n'
printf 'Fixed training pairs: artifacts/train_pairs_fixed.jsonl\n'
printf 'Model output: artifacts/denoiser_ar/final\n'
