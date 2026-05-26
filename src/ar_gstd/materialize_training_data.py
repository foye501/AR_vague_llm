from __future__ import annotations

import argparse
import json
from pathlib import Path

from .transitions import load_transition_caches, sample_corrupted_token_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample noisy denoising pairs from a sparse transition cache.")
    parser.add_argument("--cache", type=Path, default=Path("artifacts/ar_transition_cache.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/train_pairs_ar.jsonl"))
    parser.add_argument("--beta", type=float, default=0.35)
    parser.add_argument("--variants-per-example", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _require_transformers()
    from transformers import AutoTokenizer

    caches = load_transition_caches(args.cache)
    if not caches:
        raise SystemExit(f"No caches found in {args.cache}")

    tokenizer = AutoTokenizer.from_pretrained(caches[0].tokenizer_name, trust_remote_code=True)
    rows: list[dict[str, str]] = []
    for example_index, cache in enumerate(caches):
        for variant_index in range(args.variants_per_example):
            sample_seed = args.seed + example_index * 1009 + variant_index
            corrupted_ids = sample_corrupted_token_ids(cache, beta=args.beta, seed=sample_seed)
            rows.append(
                {
                    "id": f"{cache.example_id}#v{variant_index}",
                    "strategy": "ar_topk" if not cache.teacher_model.startswith("fixed-from:") else "fixed_topk",
                    "transcript": cache.transcript,
                    "corrupted_summary": tokenizer.decode(corrupted_ids, skip_special_tokens=True),
                    "clean_summary": cache.clean_summary,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({len(rows)} rows)")


def _require_transformers() -> None:
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit('Install training dependencies first: python -m pip install -e ".[train,dev]"') from exc


if __name__ == "__main__":
    main()
