#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
DENOISER_MODEL="${DENOISER_MODEL:-google/flan-t5-small}"
TOP_K="${TOP_K:-8}"
BETA="${BETA:-0.35}"
VARIANTS_PER_EXAMPLE="${VARIANTS_PER_EXAMPLE:-4}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
DATA_FILE="${DATA_FILE:-artifacts/sql_create_context_subset.jsonl}"
DATASET_NAME="${DATASET_NAME:-b-mc2/sql-create-context}"
DATASET_MAX_EXAMPLES="${DATASET_MAX_EXAMPLES:-200}"
MAX_TARGET_TOKENS="${MAX_TARGET_TOKENS:-160}"
EVAL_RATIO="${EVAL_RATIO:-0.2}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"

python -m ar_gstd.prepare_sql_create_context \
  --dataset "$DATASET_NAME" \
  --output "$DATA_FILE" \
  --max-examples "$DATASET_MAX_EXAMPLES" \
  --seed 7

python -m ar_gstd.build_transition_cache \
  --input "$DATA_FILE" \
  --output artifacts/ar_transition_cache.jsonl \
  --teacher-model "$TEACHER_MODEL" \
  --top-k "$TOP_K" \
  --max-examples "$MAX_EXAMPLES" \
  --max-target-tokens "$MAX_TARGET_TOKENS" \
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
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --eval-ratio "$EVAL_RATIO" \
  --train-split-output artifacts/train_pairs_ar_train.jsonl \
  --eval-split-output artifacts/train_pairs_ar_eval.jsonl

python -m ar_gstd.train_seq2seq_denoiser \
  --train-file artifacts/train_pairs_fixed.jsonl \
  --output-dir artifacts/denoiser_fixed \
  --model-name "$DENOISER_MODEL" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --eval-ratio "$EVAL_RATIO" \
  --train-split-output artifacts/train_pairs_fixed_train.jsonl \
  --eval-split-output artifacts/train_pairs_fixed_eval.jsonl

python -m ar_gstd.evaluate_denoiser \
  --model-dir artifacts/denoiser_ar/final \
  --eval-file artifacts/train_pairs_ar_eval.jsonl \
  --output-predictions artifacts/predictions_ar_on_ar.jsonl \
  --output-metrics artifacts/metrics_ar_on_ar.json \
  --batch-size "$BATCH_SIZE"

python -m ar_gstd.evaluate_denoiser \
  --model-dir artifacts/denoiser_fixed/final \
  --eval-file artifacts/train_pairs_ar_eval.jsonl \
  --output-predictions artifacts/predictions_fixed_on_ar.jsonl \
  --output-metrics artifacts/metrics_fixed_on_ar.json \
  --batch-size "$BATCH_SIZE"

python -m ar_gstd.evaluate_denoiser \
  --model-dir artifacts/denoiser_ar/final \
  --eval-file artifacts/train_pairs_fixed_eval.jsonl \
  --output-predictions artifacts/predictions_ar_on_fixed.jsonl \
  --output-metrics artifacts/metrics_ar_on_fixed.json \
  --batch-size "$BATCH_SIZE"

python -m ar_gstd.evaluate_denoiser \
  --model-dir artifacts/denoiser_fixed/final \
  --eval-file artifacts/train_pairs_fixed_eval.jsonl \
  --output-predictions artifacts/predictions_fixed_on_fixed.jsonl \
  --output-metrics artifacts/metrics_fixed_on_fixed.json \
  --batch-size "$BATCH_SIZE"

python -m ar_gstd.summarize_metrics \
  --output artifacts/metrics_summary.md \
  artifacts/metrics_ar_on_ar.json \
  artifacts/metrics_fixed_on_ar.json \
  artifacts/metrics_ar_on_fixed.json \
  artifacts/metrics_fixed_on_fixed.json

printf '\nFinished AR-top-k and fixed-top-k denoiser run.\n'
printf 'Dataset file: %s\n' "$DATA_FILE"
printf 'Transition cache: artifacts/ar_transition_cache.jsonl\n'
printf 'Fixed baseline cache: artifacts/fixed_transition_cache.jsonl\n'
printf 'AR training pairs: artifacts/train_pairs_ar.jsonl\n'
printf 'Fixed training pairs: artifacts/train_pairs_fixed.jsonl\n'
printf 'AR model output: artifacts/denoiser_ar/final\n'
printf 'Fixed model output: artifacts/denoiser_fixed/final\n'
printf 'Main comparison: artifacts/metrics_ar_on_ar.json vs artifacts/metrics_fixed_on_ar.json\n'
printf 'Summary table: artifacts/metrics_summary.md\n'
