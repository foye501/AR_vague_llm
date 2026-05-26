# AR-Guided Semantic Transition Diffusion

This repository starts an experiment path for AR-guided semantic transition corruption in structured language generation.

Working hypothesis:

> Mask-only corruption teaches a model to fill blanks, while real LLM errors are often fluent, plausible, and semantically wrong. AR-guided top-k transition corruption should create better denoising training data for structured outputs.

The main method should be a conditional sparse transition cache from an autoregressive teacher:

```text
q_t(y_t[p] = j | y_0[p] = i, c_p)
  = (1 - beta_t) 1[j = i] + beta_t S_AR_topk(j | c_p)
```

where `c_p` is the source context plus the clean target prefix before position `p`. In implementation, `S_AR_topk` is not a dense vocabulary matrix. It is a cached sparse row with only top-k token ids and probabilities per example and target position.

Use the fixed top-k sparse matrix as an ablation, not as the main method. Fixed top-k removes context dependence, which is the main research claim.

For the diffusion-style experiment, the endpoint is still absorbing/all-mask:

```text
q_t = p_keep(t) I + p_ar(t) S_AR_topk + p_mask(t) delta_MASK
```

with `p_mask(T)=1` and `p_ar(T)=0`. This means sampling/evaluation at `t=T` starts from total `[MASK]` noise, just like absorbing masked diffusion; AR logits only affect intermediate training states.

## What Is Here

- `src/ar_gstd/prepare_sql_create_context.py`: prepares a text-to-SQL dataset from `b-mc2/sql-create-context`.
- `src/ar_gstd/build_transition_cache.py`: builds conditional AR top-k sparse transition rows from a teacher LM.
- `src/ar_gstd/make_fixed_transition_cache.py`: converts conditional rows into a fixed sparse top-k baseline.
- `src/ar_gstd/materialize_training_data.py`: samples noisy denoising pairs from a transition cache.
- `src/ar_gstd/materialize_diffusion_training_data.py`: samples multi-timestep diffusion-style pairs with an all-mask endpoint.
- `src/ar_gstd/train_seq2seq_denoiser.py`: trains a first seq2seq denoiser on those pairs.
- `src/ar_gstd/corruption.py`: four corruption strategies:
  - `mask`
  - `random`
  - `embedding`
  - `ar_guided`
- `src/ar_gstd/generate_corruptions.py`: CLI that creates corrupted variants from seed examples.
- `src/ar_gstd/evaluate_corruptions.py`: CLI that reports basic corruption sanity metrics.
- `data/meeting_summaries_seed.jsonl`: old toy seed file, kept only for quick local smoke tests.

## Quick Start

```bash
python -m pip install -e ".[dev]"
python -m pytest
PYTHONPATH=src python -m ar_gstd.generate_corruptions \
  --input data/meeting_summaries_seed.jsonl \
  --output artifacts/corruptions.jsonl \
  --beta 0.55 \
  --seed 7
PYTHONPATH=src python -m ar_gstd.generate_corruptions \
  --input data/meeting_summaries_seed.jsonl \
  --output artifacts/corruption_table.md \
  --format markdown \
  --limit 8 \
  --beta 0.55 \
  --seed 7
PYTHONPATH=src python -m ar_gstd.evaluate_corruptions \
  --input artifacts/corruptions.jsonl \
  --output artifacts/corruption_metrics.md
```

Or install the package once:

```bash
python -m pip install -e .
```

## Remote Server Training Run

On a remote server with Python 3.10+ and GPU access:

```bash
git clone <your-repo-url> Diffusion_llm
cd Diffusion_llm
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[train,dev]"
bash scripts/run_remote_ar_transition_experiment.sh
```

Useful overrides:

```bash
TEACHER_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
DENOISER_MODEL=google/flan-t5-base \
TOP_K=16 \
BETA=0.35 \
VARIANTS_PER_EXAMPLE=16 \
DATASET_MAX_EXAMPLES=1000 \
bash scripts/run_remote_ar_transition_experiment.sh
```

This produces:

- `artifacts/ar_transition_cache.jsonl`: conditional sparse transition cache.
- `artifacts/fixed_transition_cache.jsonl`: fixed sparse baseline cache.
- `artifacts/train_pairs_ar.jsonl`: sampled AR-top-k denoising pairs.
- `artifacts/train_pairs_fixed.jsonl`: sampled fixed-top-k denoising pairs.
- `artifacts/denoiser_ar/final`: first trained denoising model.
- `artifacts/denoiser_fixed/final`: fixed-transition baseline denoising model.
- `artifacts/metrics_ar_on_ar.json`: conditional AR model on held-out AR corruptions.
- `artifacts/metrics_fixed_on_ar.json`: fixed baseline model on the same held-out AR corruptions.
- `artifacts/metrics_summary.md`: table of AR/fixed train-test combinations.

## Diffusion-Style Run

To address the absorbing-diffusion gap directly:

```bash
git pull
python -m pip install -e ".[train,dev]"
bash scripts/run_remote_diffusion_experiment.sh
```

Useful larger run:

```bash
DATASET_MAX_EXAMPLES=1000 \
TEACHER_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
DENOISER_MODEL=google/flan-t5-base \
TOP_K=16 \
NUM_STEPS=10 \
TIMESTEPS=1,2,4,6,8,10 \
bash scripts/run_remote_diffusion_experiment.sh
```

This trains:

- `base_zero_shot`: the untouched pretrained student, evaluated with clean source-to-target generation prompts.
- `denoiser_clean_sft`: ordinary supervised fine-tuning on clean source-to-SQL pairs.
- `denoiser_diff_absorb`: pure absorbing masked diffusion-style noising.
- `denoiser_diff_ar_absorb`: absorbing endpoint plus conditional AR top-k intermediate states.
- `denoiser_diff_fixed_absorb`: absorbing endpoint plus fixed top-k intermediate states.

The main output is:

```bash
cat artifacts/metrics_diffusion_tT_summary.md
```

That table evaluates all models at `t=T`, where the input is total `[MASK]` noise. If `diff_ar_absorb_tT` beats `diff_absorb_tT` and `diff_fixed_absorb_tT`, then the result supports the claim that conditional AR logits improve diffusion training without changing the all-mask endpoint.

Also inspect:

```bash
cat artifacts/metrics_control_summary.md
```

This separates pretrained ability and clean SFT from the diffusion-style objectives. The diffusion claim should compare only models with the same pretrained initialization and no AR teacher at inference time.

## Manual Pipeline

Build the conditional AR transition cache:

Prepare text-to-SQL rows first:

```bash
python -m ar_gstd.prepare_sql_create_context \
  --dataset b-mc2/sql-create-context \
  --output artifacts/sql_create_context_subset.jsonl \
  --max-examples 200
```

```bash
python -m ar_gstd.build_transition_cache \
  --input artifacts/sql_create_context_subset.jsonl \
  --output artifacts/ar_transition_cache.jsonl \
  --teacher-model Qwen/Qwen2.5-0.5B-Instruct \
  --top-k 8 \
  --device auto
```

Create the fixed sparse baseline from the same rows:

```bash
python -m ar_gstd.make_fixed_transition_cache \
  --input artifacts/ar_transition_cache.jsonl \
  --output artifacts/fixed_transition_cache.jsonl \
  --top-k 8
```

Sample noisy denoising pairs:

```bash
python -m ar_gstd.materialize_training_data \
  --cache artifacts/ar_transition_cache.jsonl \
  --output artifacts/train_pairs_ar.jsonl \
  --beta 0.35 \
  --variants-per-example 8
```

Train a first denoiser:

```bash
python -m ar_gstd.train_seq2seq_denoiser \
  --train-file artifacts/train_pairs_ar.jsonl \
  --output-dir artifacts/denoiser_ar \
  --model-name google/flan-t5-small \
  --epochs 3 \
  --batch-size 4
```

Evaluate generated predictions:

```bash
python -m ar_gstd.evaluate_denoiser \
  --model-dir artifacts/denoiser_ar/final \
  --eval-file artifacts/train_pairs_ar_eval.jsonl \
  --output-predictions artifacts/predictions_ar_on_ar.jsonl \
  --output-metrics artifacts/metrics_ar_on_ar.json
```

The important metrics are:

- `sql_exact_match`: normalized SQL string match against the gold SQL.
- `sql_keyword_valid`: simple SQL-form sanity check.
- `sql_repair_delta`: whether prediction exact-match improves over the corrupted SQL.
- `token_f1_repair_delta`: whether prediction improves over the corrupted input.

Do not use training loss or eval loss as the main evidence. The paper needs held-out generated predictions and a comparison against fixed top-k, mask, random, and embedding baselines.

If the server cannot access your Git provider, create a bundle locally and copy it:

```bash
git bundle create Diffusion_llm.bundle main
scp Diffusion_llm.bundle <user>@<server>:~/
```

Then on the server:

```bash
git clone ~/Diffusion_llm.bundle Diffusion_llm
cd Diffusion_llm
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[train,dev]"
bash scripts/run_remote_ar_transition_experiment.sh
```

## Publishable Experiment Plan

1. Train equal-size denoisers on mask, random, fixed top-k, and conditional AR top-k corruptions for text-to-SQL.
2. Train diffusion-style variants with the same all-mask endpoint: absorbing, fixed+absorbing, and AR+absorbing.
3. Evaluate at `t=T` total-mask input and mixed intermediate timesteps.
4. Report SQL exact match, SQL validity, execution accuracy where database files are available, and repair deltas over corrupted inputs.
5. Add ablations over `top_k`, AR strength, teacher size, timestep schedule, and token-level versus clause/span-level SQL corruption.
6. After this proxy is proven, port the same transition cache into a native discrete diffusion objective.
