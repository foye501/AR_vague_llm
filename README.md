# AR-Guided Semantic Transition Diffusion

This repository starts an experiment path for AR-guided semantic transition corruption in structured language generation.

Working hypothesis:

> Mask-only corruption teaches a model to fill blanks, while real LLM errors are often fluent, plausible, and semantically wrong. AR-guided top-k transition corruption should create better denoising training data for structured outputs.

The main method should be a conditional sparse transition cache from an autoregressive teacher:

```text
q_t(y_t[p] = j | y_0[p] = i, c_p)
  = (1 - beta_t) 1[j = i] + beta_t S_AR_topk(j | c_p)
```

where `c_p` is the transcript plus the clean summary prefix before position `p`. In implementation, `S_AR_topk` is not a dense vocabulary matrix. It is a cached sparse row with only top-k token ids and probabilities per example and target position.

Use the fixed top-k sparse matrix as an ablation, not as the main method. Fixed top-k removes context dependence, which is the main research claim.

## What Is Here

- `data/meeting_summaries_seed.jsonl`: 20 small transcript/clean-summary examples.
- `src/ar_gstd/build_transition_cache.py`: builds conditional AR top-k sparse transition rows from a teacher LM.
- `src/ar_gstd/make_fixed_transition_cache.py`: converts conditional rows into a fixed sparse top-k baseline.
- `src/ar_gstd/materialize_training_data.py`: samples noisy denoising pairs from a transition cache.
- `src/ar_gstd/train_seq2seq_denoiser.py`: trains a first seq2seq denoiser on those pairs.
- `src/ar_gstd/corruption.py`: four corruption strategies:
  - `mask`
  - `random`
  - `embedding`
  - `ar_guided`
- `src/ar_gstd/generate_corruptions.py`: CLI that creates corrupted variants from seed examples.
- `src/ar_gstd/evaluate_corruptions.py`: CLI that reports basic corruption sanity metrics.
- `tests/test_corruption.py`: smoke tests for deterministic and structurally valid corruption.

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

## Manual Pipeline

Build the conditional AR transition cache:

```bash
python -m ar_gstd.build_transition_cache \
  --input data/meeting_summaries_seed.jsonl \
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

- `token_f1_repair_delta`: whether prediction improves over the corrupted input.
- `line_f1_repair_delta`: whether the model recovers exact summary bullets.
- `prediction_heading_valid`: whether the output keeps the required structured format.
- `owner_term_accuracy`, `deadline_term_accuracy`, `decision_term_accuracy`: task-specific recovery signals.

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

1. Train equal-size denoisers on mask, random, embedding, fixed top-k, and conditional AR top-k corruptions.
2. Evaluate on synthetic corruptions and real flawed AR summaries.
3. Report decision-status accuracy, owner/action/deadline recovery, section validity, and factual repair.
4. Add ablations over `top_k`, `beta`, teacher size, and token-level versus span-level corruption.
5. After the denoising effect is proven, port the same transition cache into a discrete diffusion objective.
