#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
TEACHER_DTYPE="${TEACHER_DTYPE:-bfloat16}"
DENOISER_MODEL="${DENOISER_MODEL:-google/flan-t5-small}"
DENOISER_TOKENIZER_NAME="${DENOISER_TOKENIZER_NAME:-$DENOISER_MODEL}"
STUDENT_TOKENIZER_NAME="${STUDENT_TOKENIZER_NAME:-$DENOISER_TOKENIZER_NAME}"
TOP_K="${TOP_K:-8}"
DATA_FILE="${DATA_FILE:-artifacts/sql_create_context_subset.jsonl}"
DATASET_NAME="${DATASET_NAME:-b-mc2/sql-create-context}"
DATASET_MAX_EXAMPLES="${DATASET_MAX_EXAMPLES:-200}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
MAX_TARGET_TOKENS="${MAX_TARGET_TOKENS:-160}"
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
RESUME_CLEAN_SFT="${RESUME_CLEAN_SFT:-}"
RESUME_DIFF_ABSORB="${RESUME_DIFF_ABSORB:-}"
RESUME_DIFF_AR_ABSORB="${RESUME_DIFF_AR_ABSORB:-}"
RESUME_DIFF_FIXED_ABSORB="${RESUME_DIFF_FIXED_ABSORB:-}"
TRAIN_FROM_SCRATCH="${TRAIN_FROM_SCRATCH:-0}"
ADD_MASK_TOKEN="${ADD_MASK_TOKEN:-0}"
MASK_TOKEN="${MASK_TOKEN:-[MASK]}"
PAD_TOKEN="${PAD_TOKEN:-[PAD]}"
RUN_BASE_ZERO_SHOT="${RUN_BASE_ZERO_SHOT:-1}"

if [[ "$TRAIN_FROM_SCRATCH" == "1" || "$DENOISER_TOKENIZER_NAME" != "$DENOISER_MODEL" ]]; then
  RUN_BASE_ZERO_SHOT=0
fi
if [[ "$DENOISER_TOKENIZER_NAME" != "$DENOISER_MODEL" && "$TRAIN_FROM_SCRATCH" != "1" ]]; then
  printf 'DENOISER_TOKENIZER_NAME differs from DENOISER_MODEL; set TRAIN_FROM_SCRATCH=1 to avoid mismatched pretrained embeddings.\n' >&2
  exit 1
fi

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

TRAIN_MODEL_ARGS=(
  --model-name "$DENOISER_MODEL"
  --tokenizer-name "$DENOISER_TOKENIZER_NAME"
  --mask-token "$MASK_TOKEN"
  --pad-token "$PAD_TOKEN"
)
if [[ "$TRAIN_FROM_SCRATCH" == "1" ]]; then
  TRAIN_MODEL_ARGS+=(--from-scratch)
fi
if [[ "$ADD_MASK_TOKEN" == "1" ]]; then
  TRAIN_MODEL_ARGS+=(--add-mask-token)
fi

resume_args() {
  local checkpoint_path="$1"
  if [[ -n "$checkpoint_path" ]]; then
    printf '%s\n' --resume-from-checkpoint "$checkpoint_path"
  fi
}

python -m ar_gstd.prepare_sql_create_context \
  --dataset "$DATASET_NAME" \
  --output "$DATA_FILE" \
  --max-examples "$DATASET_MAX_EXAMPLES" \
  --seed 7

python -m ar_gstd.materialize_clean_training_data \
  --input "$DATA_FILE" \
  --output artifacts/train_pairs_clean_sft.jsonl

python -m ar_gstd.train_seq2seq_denoiser \
  --train-file artifacts/train_pairs_clean_sft.jsonl \
  --output-dir artifacts/denoiser_clean_sft \
  "${TRAIN_MODEL_ARGS[@]}" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  "${TRAIN_EXTRA_ARGS[@]}" \
  $(resume_args "$RESUME_CLEAN_SFT") \
  --eval-ratio "$EVAL_RATIO" \
  --train-split-output artifacts/train_pairs_clean_sft_train.jsonl \
  --eval-split-output artifacts/train_pairs_clean_sft_eval.jsonl

if [[ "$RUN_BASE_ZERO_SHOT" == "1" ]]; then
  python -m ar_gstd.evaluate_denoiser \
    --model-dir "$DENOISER_MODEL" \
    --eval-file artifacts/train_pairs_clean_sft_eval.jsonl \
    --output-predictions artifacts/predictions_base_zero_shot.jsonl \
    --output-metrics artifacts/metrics_base_zero_shot.json \
    --batch-size "$BATCH_SIZE"
fi

python -m ar_gstd.evaluate_denoiser \
  --model-dir artifacts/denoiser_clean_sft/final \
  --eval-file artifacts/train_pairs_clean_sft_eval.jsonl \
  --output-predictions artifacts/predictions_clean_sft.jsonl \
  --output-metrics artifacts/metrics_clean_sft.json \
  --batch-size "$BATCH_SIZE"

python -m ar_gstd.build_transition_cache \
  --input "$DATA_FILE" \
  --output artifacts/ar_transition_cache.jsonl \
  --teacher-model "$TEACHER_MODEL" \
  --student-tokenizer-name "$STUDENT_TOKENIZER_NAME" \
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
  --output artifacts/train_pairs_diff_absorb.jsonl \
  --noise-kind absorbing \
  --num-steps "$NUM_STEPS" \
  --timesteps "$TIMESTEPS" \
  --variants-per-timestep "$VARIANTS_PER_TIMESTEP" \
  --mask-token "$MASK_TOKEN" \
  --seed 7

python -m ar_gstd.materialize_diffusion_training_data \
  --cache artifacts/ar_transition_cache.jsonl \
  --output artifacts/train_pairs_diff_ar_absorb.jsonl \
  --noise-kind ar_absorb \
  --num-steps "$NUM_STEPS" \
  --timesteps "$TIMESTEPS" \
  --variants-per-timestep "$VARIANTS_PER_TIMESTEP" \
  --ar-strength "$AR_STRENGTH" \
  --mask-token "$MASK_TOKEN" \
  --seed 7

python -m ar_gstd.materialize_diffusion_training_data \
  --cache artifacts/fixed_transition_cache.jsonl \
  --output artifacts/train_pairs_diff_fixed_absorb.jsonl \
  --noise-kind ar_absorb \
  --num-steps "$NUM_STEPS" \
  --timesteps "$TIMESTEPS" \
  --variants-per-timestep "$VARIANTS_PER_TIMESTEP" \
  --ar-strength "$AR_STRENGTH" \
  --mask-token "$MASK_TOKEN" \
  --seed 7

python -m ar_gstd.train_seq2seq_denoiser \
  --train-file artifacts/train_pairs_diff_absorb.jsonl \
  --output-dir artifacts/denoiser_diff_absorb \
  "${TRAIN_MODEL_ARGS[@]}" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  "${TRAIN_EXTRA_ARGS[@]}" \
  $(resume_args "$RESUME_DIFF_ABSORB") \
  --eval-ratio "$EVAL_RATIO" \
  --train-split-output artifacts/train_pairs_diff_absorb_train.jsonl \
  --eval-split-output artifacts/train_pairs_diff_absorb_eval.jsonl

python -m ar_gstd.train_seq2seq_denoiser \
  --train-file artifacts/train_pairs_diff_ar_absorb.jsonl \
  --output-dir artifacts/denoiser_diff_ar_absorb \
  "${TRAIN_MODEL_ARGS[@]}" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  "${TRAIN_EXTRA_ARGS[@]}" \
  $(resume_args "$RESUME_DIFF_AR_ABSORB") \
  --eval-ratio "$EVAL_RATIO" \
  --train-split-output artifacts/train_pairs_diff_ar_absorb_train.jsonl \
  --eval-split-output artifacts/train_pairs_diff_ar_absorb_eval.jsonl

python -m ar_gstd.train_seq2seq_denoiser \
  --train-file artifacts/train_pairs_diff_fixed_absorb.jsonl \
  --output-dir artifacts/denoiser_diff_fixed_absorb \
  "${TRAIN_MODEL_ARGS[@]}" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  "${TRAIN_EXTRA_ARGS[@]}" \
  $(resume_args "$RESUME_DIFF_FIXED_ABSORB") \
  --eval-ratio "$EVAL_RATIO" \
  --train-split-output artifacts/train_pairs_diff_fixed_absorb_train.jsonl \
  --eval-split-output artifacts/train_pairs_diff_fixed_absorb_eval.jsonl

python -m ar_gstd.evaluate_denoiser \
  --model-dir artifacts/denoiser_diff_absorb/final \
  --eval-file artifacts/train_pairs_diff_absorb_eval.jsonl \
  --only-timestep "$NUM_STEPS" \
  --output-predictions artifacts/predictions_diff_absorb_tT.jsonl \
  --output-metrics artifacts/metrics_diff_absorb_tT.json \
  --batch-size "$BATCH_SIZE"

python -m ar_gstd.evaluate_denoiser \
  --model-dir artifacts/denoiser_diff_ar_absorb/final \
  --eval-file artifacts/train_pairs_diff_ar_absorb_eval.jsonl \
  --only-timestep "$NUM_STEPS" \
  --output-predictions artifacts/predictions_diff_ar_absorb_tT.jsonl \
  --output-metrics artifacts/metrics_diff_ar_absorb_tT.json \
  --batch-size "$BATCH_SIZE"

python -m ar_gstd.evaluate_denoiser \
  --model-dir artifacts/denoiser_diff_fixed_absorb/final \
  --eval-file artifacts/train_pairs_diff_fixed_absorb_eval.jsonl \
  --only-timestep "$NUM_STEPS" \
  --output-predictions artifacts/predictions_diff_fixed_absorb_tT.jsonl \
  --output-metrics artifacts/metrics_diff_fixed_absorb_tT.json \
  --batch-size "$BATCH_SIZE"

python -m ar_gstd.filter_eval_by_category \
  --input artifacts/train_pairs_diff_ar_absorb_eval.jsonl \
  --output artifacts/train_pairs_diff_ar_absorb_eval_schema.jsonl \
  --category schema_identifier

python -m ar_gstd.filter_eval_by_category \
  --input artifacts/train_pairs_diff_fixed_absorb_eval.jsonl \
  --output artifacts/train_pairs_diff_fixed_absorb_eval_schema.jsonl \
  --category schema_identifier

python -m ar_gstd.evaluate_denoiser \
  --model-dir artifacts/denoiser_diff_ar_absorb/final \
  --eval-file artifacts/train_pairs_diff_ar_absorb_eval_schema.jsonl \
  --only-timestep "$NUM_STEPS" \
  --output-predictions artifacts/predictions_diff_ar_absorb_schema_tT.jsonl \
  --output-metrics artifacts/metrics_diff_ar_absorb_schema_tT.json \
  --batch-size "$BATCH_SIZE"

python -m ar_gstd.evaluate_denoiser \
  --model-dir artifacts/denoiser_diff_fixed_absorb/final \
  --eval-file artifacts/train_pairs_diff_fixed_absorb_eval_schema.jsonl \
  --only-timestep "$NUM_STEPS" \
  --output-predictions artifacts/predictions_diff_fixed_absorb_schema_tT.jsonl \
  --output-metrics artifacts/metrics_diff_fixed_absorb_schema_tT.json \
  --batch-size "$BATCH_SIZE"

python -m ar_gstd.summarize_metrics \
  --output artifacts/metrics_diffusion_tT_summary.md \
  artifacts/metrics_diff_absorb_tT.json \
  artifacts/metrics_diff_ar_absorb_tT.json \
  artifacts/metrics_diff_fixed_absorb_tT.json

CONTROL_METRICS=(artifacts/metrics_clean_sft.json)
if [[ "$RUN_BASE_ZERO_SHOT" == "1" ]]; then
  CONTROL_METRICS=(artifacts/metrics_base_zero_shot.json "${CONTROL_METRICS[@]}")
fi
CONTROL_METRICS+=(
  artifacts/metrics_diff_absorb_tT.json
  artifacts/metrics_diff_ar_absorb_tT.json
  artifacts/metrics_diff_fixed_absorb_tT.json
)

python -m ar_gstd.summarize_metrics \
  --output artifacts/metrics_control_summary.md \
  "${CONTROL_METRICS[@]}"

python -m ar_gstd.summarize_metrics \
  --output artifacts/metrics_schema_focus_summary.md \
  artifacts/metrics_diff_ar_absorb_schema_tT.json \
  artifacts/metrics_diff_fixed_absorb_schema_tT.json

printf '\nFinished diffusion-style all-mask endpoint run.\n'
printf 'Main comparison: artifacts/metrics_diffusion_tT_summary.md\n'
printf 'Control comparison: artifacts/metrics_control_summary.md\n'
