#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
TEACHER_DTYPE="${TEACHER_DTYPE:-bfloat16}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen2.5-0.5B}"
TOKENIZER_NAME="${TOKENIZER_NAME:-$TEACHER_MODEL}"
STUDENT_FROM_SCRATCH="${STUDENT_FROM_SCRATCH:-0}"
TOP_K="${TOP_K:-8}"
DATA_FILE="${DATA_FILE:-artifacts/sql_create_context_subset.jsonl}"
DATASET_NAME="${DATASET_NAME:-b-mc2/sql-create-context}"
DATASET_MAX_EXAMPLES="${DATASET_MAX_EXAMPLES:-200}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
MAX_TARGET_TOKENS="${MAX_TARGET_TOKENS:-160}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-1024}"
NUM_STEPS="${NUM_STEPS:-10}"
TIMESTEPS="${TIMESTEPS:-1,2,4,6,8,10}"
VARIANTS_PER_TIMESTEP="${VARIANTS_PER_TIMESTEP:-1}"
AR_STRENGTH="${AR_STRENGTH:-0.65}"
EVAL_RATIO="${EVAL_RATIO:-0.2}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
TRAIN_BF16="${TRAIN_BF16:-0}"
TRAIN_FP16="${TRAIN_FP16:-0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
MASK_TOKEN="${MASK_TOKEN:-[MASK]}"
PAD_TOKEN="${PAD_TOKEN:-[PAD]}"

TRAIN_EXTRA_ARGS=(
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --save-strategy "$SAVE_STRATEGY"
  --save-total-limit "$SAVE_TOTAL_LIMIT"
  --save-steps "$SAVE_STEPS"
)
if [[ "$TRAIN_BF16" == "1" ]]; then
  TRAIN_EXTRA_ARGS+=(--bf16)
fi
if [[ "$TRAIN_FP16" == "1" ]]; then
  TRAIN_EXTRA_ARGS+=(--fp16)
fi
if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
  TRAIN_EXTRA_ARGS+=(--gradient-checkpointing)
fi
if [[ "$STUDENT_FROM_SCRATCH" == "1" ]]; then
  TRAIN_EXTRA_ARGS+=(--from-scratch)
fi

python -m ar_gstd.prepare_sql_create_context \
  --dataset "$DATASET_NAME" \
  --output "$DATA_FILE" \
  --max-examples "$DATASET_MAX_EXAMPLES" \
  --seed 7

python -m ar_gstd.build_transition_cache \
  --input "$DATA_FILE" \
  --output artifacts/ar_transition_cache.jsonl \
  --teacher-model "$TEACHER_MODEL" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --student-tokenizer-name "$TOKENIZER_NAME" \
  --top-k "$TOP_K" \
  --max-examples "$MAX_EXAMPLES" \
  --max-target-tokens "$MAX_TARGET_TOKENS" \
  --dtype "$TEACHER_DTYPE" \
  --device auto

python -m ar_gstd.make_fixed_transition_cache \
  --input artifacts/ar_transition_cache.jsonl \
  --output artifacts/fixed_transition_cache.jsonl \
  --top-k "$TOP_K"

python -m ar_gstd.analyze_transition_cache \
  --conditional-cache artifacts/ar_transition_cache.jsonl \
  --fixed-cache artifacts/fixed_transition_cache.jsonl \
  --output-json artifacts/transition_cache_analysis.json \
  --output-markdown artifacts/transition_cache_analysis.md

python -m ar_gstd.materialize_diffusion_training_data \
  --cache artifacts/ar_transition_cache.jsonl \
  --output artifacts/train_pairs_bidir_absorb.jsonl \
  --noise-kind absorbing \
  --num-steps "$NUM_STEPS" \
  --timesteps "$TIMESTEPS" \
  --variants-per-timestep "$VARIANTS_PER_TIMESTEP" \
  --mask-token "$MASK_TOKEN" \
  --pad-token "$PAD_TOKEN" \
  --seed 7

python -m ar_gstd.materialize_diffusion_training_data \
  --cache artifacts/ar_transition_cache.jsonl \
  --output artifacts/train_pairs_bidir_ar_absorb.jsonl \
  --noise-kind ar_absorb \
  --num-steps "$NUM_STEPS" \
  --timesteps "$TIMESTEPS" \
  --variants-per-timestep "$VARIANTS_PER_TIMESTEP" \
  --ar-strength "$AR_STRENGTH" \
  --mask-token "$MASK_TOKEN" \
  --pad-token "$PAD_TOKEN" \
  --seed 7

python -m ar_gstd.materialize_diffusion_training_data \
  --cache artifacts/fixed_transition_cache.jsonl \
  --output artifacts/train_pairs_bidir_fixed_absorb.jsonl \
  --noise-kind ar_absorb \
  --num-steps "$NUM_STEPS" \
  --timesteps "$TIMESTEPS" \
  --variants-per-timestep "$VARIANTS_PER_TIMESTEP" \
  --ar-strength "$AR_STRENGTH" \
  --mask-token "$MASK_TOKEN" \
  --pad-token "$PAD_TOKEN" \
  --seed 7

python -m ar_gstd.train_bidirectional_qwen_denoiser \
  --train-file artifacts/train_pairs_bidir_absorb.jsonl \
  --output-dir artifacts/denoiser_bidir_absorb \
  --model-name "$STUDENT_MODEL" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --mask-token "$MASK_TOKEN" \
  --pad-token "$PAD_TOKEN" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  "${TRAIN_EXTRA_ARGS[@]}" \
  --eval-ratio "$EVAL_RATIO" \
  --max-sequence-length "$MAX_SEQUENCE_LENGTH" \
  --max-target-length "$MAX_TARGET_TOKENS" \
  --train-split-output artifacts/train_pairs_bidir_absorb_train.jsonl \
  --eval-split-output artifacts/train_pairs_bidir_absorb_eval.jsonl

python -m ar_gstd.train_bidirectional_qwen_denoiser \
  --train-file artifacts/train_pairs_bidir_ar_absorb.jsonl \
  --output-dir artifacts/denoiser_bidir_ar_absorb \
  --model-name "$STUDENT_MODEL" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --mask-token "$MASK_TOKEN" \
  --pad-token "$PAD_TOKEN" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  "${TRAIN_EXTRA_ARGS[@]}" \
  --eval-ratio "$EVAL_RATIO" \
  --max-sequence-length "$MAX_SEQUENCE_LENGTH" \
  --max-target-length "$MAX_TARGET_TOKENS" \
  --train-split-output artifacts/train_pairs_bidir_ar_absorb_train.jsonl \
  --eval-split-output artifacts/train_pairs_bidir_ar_absorb_eval.jsonl

python -m ar_gstd.train_bidirectional_qwen_denoiser \
  --train-file artifacts/train_pairs_bidir_fixed_absorb.jsonl \
  --output-dir artifacts/denoiser_bidir_fixed_absorb \
  --model-name "$STUDENT_MODEL" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --mask-token "$MASK_TOKEN" \
  --pad-token "$PAD_TOKEN" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  "${TRAIN_EXTRA_ARGS[@]}" \
  --eval-ratio "$EVAL_RATIO" \
  --max-sequence-length "$MAX_SEQUENCE_LENGTH" \
  --max-target-length "$MAX_TARGET_TOKENS" \
  --train-split-output artifacts/train_pairs_bidir_fixed_absorb_train.jsonl \
  --eval-split-output artifacts/train_pairs_bidir_fixed_absorb_eval.jsonl

python -m ar_gstd.evaluate_bidirectional_qwen_denoiser \
  --model-dir artifacts/denoiser_bidir_absorb/final \
  --eval-file artifacts/train_pairs_bidir_absorb_eval.jsonl \
  --only-timestep "$NUM_STEPS" \
  --output-predictions artifacts/predictions_bidir_absorb_tT.jsonl \
  --output-metrics artifacts/metrics_bidir_absorb_tT.json \
  --batch-size "$BATCH_SIZE" \
  --max-sequence-length "$MAX_SEQUENCE_LENGTH" \
  --max-target-length "$MAX_TARGET_TOKENS"

python -m ar_gstd.evaluate_bidirectional_qwen_denoiser \
  --model-dir artifacts/denoiser_bidir_ar_absorb/final \
  --eval-file artifacts/train_pairs_bidir_ar_absorb_eval.jsonl \
  --only-timestep "$NUM_STEPS" \
  --output-predictions artifacts/predictions_bidir_ar_absorb_tT.jsonl \
  --output-metrics artifacts/metrics_bidir_ar_absorb_tT.json \
  --batch-size "$BATCH_SIZE" \
  --max-sequence-length "$MAX_SEQUENCE_LENGTH" \
  --max-target-length "$MAX_TARGET_TOKENS"

python -m ar_gstd.evaluate_bidirectional_qwen_denoiser \
  --model-dir artifacts/denoiser_bidir_fixed_absorb/final \
  --eval-file artifacts/train_pairs_bidir_fixed_absorb_eval.jsonl \
  --only-timestep "$NUM_STEPS" \
  --output-predictions artifacts/predictions_bidir_fixed_absorb_tT.jsonl \
  --output-metrics artifacts/metrics_bidir_fixed_absorb_tT.json \
  --batch-size "$BATCH_SIZE" \
  --max-sequence-length "$MAX_SEQUENCE_LENGTH" \
  --max-target-length "$MAX_TARGET_TOKENS"

python -m ar_gstd.summarize_metrics \
  --output artifacts/metrics_bidirectional_qwen_tT_summary.md \
  artifacts/metrics_bidir_absorb_tT.json \
  artifacts/metrics_bidir_ar_absorb_tT.json \
  artifacts/metrics_bidir_fixed_absorb_tT.json

printf '\nFinished bidirectional Qwen masked-denoiser run.\n'
printf 'Main comparison: artifacts/metrics_bidirectional_qwen_tT_summary.md\n'
