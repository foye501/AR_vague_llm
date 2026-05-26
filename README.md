# AR-Guided Semantic Transition Diffusion Prototype

This repository starts a small experimental scaffold for AR-guided semantic corruption in structured language generation.

Working hypothesis:

> Mask-only corruption teaches a model to fill blanks, while real LLM errors are often fluent, plausible, and semantically wrong. AR-guided top-k transition corruption should create better denoising training data for structured outputs.

## What Is Here

- `data/meeting_summaries_seed.jsonl`: 20 small transcript/clean-summary examples.
- `src/ar_gstd/corruption.py`: four corruption strategies:
  - `mask`
  - `random`
  - `embedding`
  - `ar_guided`
- `src/ar_gstd/generate_corruptions.py`: CLI that creates corrupted variants from seed examples.
- `src/ar_gstd/evaluate_corruptions.py`: CLI that reports basic corruption sanity metrics.
- `tests/test_corruption.py`: smoke tests for deterministic and structurally valid corruption.

The first goal is not to train a diffusion model yet. The first goal is to inspect whether AR-guided semantic noise is more realistic than mask/random/static-neighbor noise.

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

## Remote Server Run

On a remote server with Python 3.10+:

```bash
git clone <your-repo-url> Diffusion_llm
cd Diffusion_llm
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
bash scripts/run_seed_experiment.sh
```

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
python -m pip install -e ".[dev]"
bash scripts/run_seed_experiment.sh
```

The runner writes:

- `artifacts/corruptions.jsonl`
- `artifacts/corruption_table.md`
- `artifacts/corruption_metrics.md`

## Next Experiments

1. Replace the rule-based `ar_guided` strategy with a real teacher model that proposes top-k span replacements.
2. Add a real-error evaluation set where an AR model generates flawed summaries from transcripts.
3. Train equal-size denoisers on mask, random, embedding, and AR-guided corruptions.
4. Evaluate decision-status accuracy, owner/action/deadline recovery, section validity, and factual repair.
