from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .transitions import (
    SparseTransitionRow,
    TransitionCache,
    load_transition_caches,
    normalize_probs,
    write_transition_caches,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a conditional transition cache into a fixed sparse top-k baseline.")
    parser.add_argument("--input", type=Path, default=Path("artifacts/ar_transition_cache.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/fixed_transition_cache.jsonl"))
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    conditional = load_transition_caches(args.input)
    fixed_rows = build_fixed_rows(conditional, top_k=args.top_k)
    rewritten = [rewrite_cache_with_fixed_rows(cache, fixed_rows) for cache in conditional]
    write_transition_caches(args.output, rewritten)
    print(f"Wrote {args.output}")


def build_fixed_rows(caches: list[TransitionCache], *, top_k: int) -> dict[int, tuple[tuple[int, ...], tuple[float, ...], tuple[str, ...]]]:
    scores: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    texts: dict[int, str] = {}
    for cache in caches:
        for row in cache.rows:
            for token_id, prob, text in zip(row.top_token_ids, row.top_probs, row.top_texts, strict=True):
                scores[row.source_token_id][token_id] += prob
                texts[token_id] = text

    fixed: dict[int, tuple[tuple[int, ...], tuple[float, ...], tuple[str, ...]]] = {}
    for source_token_id, candidate_scores in scores.items():
        ranked = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        token_ids = tuple(token_id for token_id, _ in ranked)
        probs = normalize_probs(score for _, score in ranked)
        fixed[source_token_id] = (token_ids, probs, tuple(texts.get(token_id, "") for token_id in token_ids))
    return fixed


def rewrite_cache_with_fixed_rows(
    cache: TransitionCache,
    fixed_rows: dict[int, tuple[tuple[int, ...], tuple[float, ...], tuple[str, ...]]],
) -> TransitionCache:
    rows: list[SparseTransitionRow] = []
    for row in cache.rows:
        token_ids, probs, texts = fixed_rows.get(row.source_token_id, ((), (), ()))
        rows.append(
            SparseTransitionRow(
                position=row.position,
                source_token_id=row.source_token_id,
                source_text=row.source_text,
                top_token_ids=token_ids,
                top_probs=probs,
                top_texts=texts,
            )
        )
    return TransitionCache(
        example_id=cache.example_id,
        transcript=cache.transcript,
        clean_summary=cache.clean_summary,
        tokenizer_name=cache.tokenizer_name,
        teacher_model=f"fixed-from:{cache.teacher_model}",
        target_token_ids=cache.target_token_ids,
        rows=tuple(rows),
    )


if __name__ == "__main__":
    main()
