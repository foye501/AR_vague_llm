# AR-Guided Token-Adaptive Masked Diffusion for Text-to-SQL

## Research Position

The strongest claim is not that AR next-token logits are exact diffusion reverse transitions. The stronger and cleaner claim is:

> An AR teacher can be used offline to estimate token difficulty and train a bidirectional masked diffusion student with token-adaptive masking for structured text generation.

For text-to-SQL, token difficulty is meaningful because output tokens have interpretable roles:

- easy: SQL keywords, punctuation, common operators
- hard: schema identifiers, literals, joins, aggregation choices

The student uses full bidirectional attention and no AR teacher at inference time.

## Core Hypothesis

Uniform masking wastes training budget on easy structural tokens. AR teacher surprisal identifies tokens that need more denoising pressure. If high-surprisal tokens are masked more often during training, the diffusion student should improve on schema and literal correctness at the same inference step budget.

## Method

For each example `(source, SQL)` and target token `y_i`:

```text
s_i = -log p_AR(y_i | source, y_<i)
```

Then convert `s_i` to a masking weight:

```text
w_i = normalize(s_i)
p_mask_i(t) = clamp(p_mask(t) * w_i, 0, 1)
```

At training time:

```text
input  = [source context || noisy SQL state]
target = clean SQL token ids
loss   = cross entropy on SQL target positions only
```

At inference time:

```text
start from all [MASK]
iteratively denoise with the bidirectional student
no AR teacher
```

## Experiment Matrix

Run these in the same tokenizer and architecture:

| Variant | Meaning |
| --- | --- |
| `bidir_absorb` | uniform absorbing masked diffusion |
| `bidir_ar_absorb` | AR top-k transition corruption, uniform masking |
| `bidir_fixed_absorb` | fixed top-k transition corruption, uniform masking |
| `bidir_ar_adaptive` | AR top-k transition corruption plus AR difficulty masking |

The first target result is:

```text
bidir_ar_adaptive > bidir_ar_absorb >= bidir_absorb
```

If this does not hold, inspect token categories. The adaptive method may still help schema/literal tokens while not improving global exact match.

## Metrics

Primary:

- SQL exact match
- SQL validity
- token F1
- schema-focused exact match

Diagnostics:

- mean AR surprisal by token category
- mask rate by token category
- performance by SQL token category

Efficiency:

- number of denoising steps
- tokens per second
- no AR teacher at inference
- offline AR-score cache cost reported separately

## Starter A100 Run

Use this for the first serious run:

```bash
git switch codex-bidirectional-qwen-denoiser

TEACHER_MODEL=Qwen/Qwen2.5-3B-Instruct \
STUDENT_MODEL=Qwen/Qwen2.5-0.5B \
TOKENIZER_NAME=Qwen/Qwen2.5-3B-Instruct \
TRAIN_BF16=1 \
DATASET_MAX_EXAMPLES=10000 \
BATCH_SIZE=16 \
GRADIENT_ACCUMULATION_STEPS=2 \
LEARNING_RATE=2e-5 \
EPOCHS=3 \
TOP_K=16 \
MAX_TARGET_TOKENS=160 \
TIMESTEPS=1,2,4,6,8,10 \
VARIANTS_PER_TIMESTEP=1 \
scripts/run_remote_bidirectional_qwen_experiment.sh
```

For scratch student training:

```bash
STUDENT_FROM_SCRATCH=1 \
LEARNING_RATE=3e-4 \
GRADIENT_ACCUMULATION_STEPS=2 \
TRAIN_BF16=1 \
scripts/run_remote_bidirectional_qwen_experiment.sh
```

Use pretrained adaptation first. Scratch training is cleaner scientifically, but pretrained adaptation is more likely to produce a readable signal quickly.

## Expected Artifacts

```text
artifacts/ar_token_scores.jsonl
artifacts/train_pairs_bidir_ar_adaptive.jsonl
artifacts/denoiser_bidir_ar_adaptive/final
artifacts/metrics_bidirectional_qwen_tT_summary.md
```

Inspect:

```bash
cat artifacts/metrics_bidirectional_qwen_tT_summary.md
cat artifacts/transition_cache_analysis.md
```

Run iterative evaluation only after checkpoints already exist:

```bash
RUN_PREP=0 \
RUN_TRAIN=0 \
ITERATIVE_VARIANT=bidir_ar_absorb \
ITERATIVE_STEPS=2,4,8 \
BATCH_SIZE=16 \
MAX_TARGET_TOKENS=160 \
scripts/run_remote_bidirectional_qwen_experiment.sh
```

This writes:

```text
artifacts/metrics_bidir_ar_absorb_iterative_summary.md
```

## Next Milestones

1. Add token-adaptive loss weights in addition to token-adaptive masking.
2. Add execution accuracy on Spider/BIRD.
3. Extend iterative denoising evaluation to learned remasking schedules.
4. Move from synthetic/subset SQL to Spider or BIRD.
5. Report offline AR-score cost separately from inference throughput.
