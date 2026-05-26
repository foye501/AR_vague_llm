from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class SparseTransitionRow:
    position: int
    source_token_id: int
    source_text: str
    top_token_ids: tuple[int, ...]
    top_probs: tuple[float, ...]
    top_texts: tuple[str, ...]


@dataclass(frozen=True)
class TransitionCache:
    example_id: str
    transcript: str
    clean_summary: str
    tokenizer_name: str
    teacher_model: str
    target_token_ids: tuple[int, ...]
    rows: tuple[SparseTransitionRow, ...]


def load_transition_caches(path: Path) -> list[TransitionCache]:
    return [transition_cache_from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def transition_cache_from_dict(row: dict[str, Any]) -> TransitionCache:
    return TransitionCache(
        example_id=str(row["id"]),
        transcript=str(row["transcript"]),
        clean_summary=str(row["clean_summary"]),
        tokenizer_name=str(row["tokenizer_name"]),
        teacher_model=str(row["teacher_model"]),
        target_token_ids=tuple(int(token_id) for token_id in row["target_token_ids"]),
        rows=tuple(_transition_row_from_dict(item) for item in row["rows"]),
    )


def transition_cache_to_dict(cache: TransitionCache) -> dict[str, Any]:
    return {
        "id": cache.example_id,
        "transcript": cache.transcript,
        "clean_summary": cache.clean_summary,
        "tokenizer_name": cache.tokenizer_name,
        "teacher_model": cache.teacher_model,
        "target_token_ids": list(cache.target_token_ids),
        "rows": [_transition_row_to_dict(row) for row in cache.rows],
    }


def sample_corrupted_token_ids(cache: TransitionCache, *, beta: float, seed: int, force_change: bool = True) -> list[int]:
    if not 0 <= beta <= 1:
        raise ValueError("beta must be between 0 and 1")

    rng = random.Random(seed)
    corrupted = list(cache.target_token_ids)
    changed_positions: list[int] = []
    rows_by_position = {row.position: row for row in cache.rows}

    for position, source_token_id in enumerate(cache.target_token_ids):
        row = rows_by_position.get(position)
        if row is None or not row.top_token_ids:
            continue
        if rng.random() >= beta:
            continue
        sampled = sample_from_sparse_row(row, rng)
        if sampled != source_token_id:
            corrupted[position] = sampled
            changed_positions.append(position)

    if force_change and not changed_positions:
        candidates = [row for row in cache.rows if row.top_token_ids]
        if candidates:
            row = candidates[seed % len(candidates)]
            corrupted[row.position] = sample_from_sparse_row(row, rng)

    return corrupted


def sample_from_sparse_row(row: SparseTransitionRow, rng: random.Random) -> int:
    if not row.top_token_ids:
        return row.source_token_id
    return rng.choices(list(row.top_token_ids), weights=list(row.top_probs), k=1)[0]


def normalize_probs(probs: Iterable[float]) -> tuple[float, ...]:
    values = tuple(max(float(prob), 0.0) for prob in probs)
    total = sum(values)
    if total <= 0:
        if not values:
            return ()
        return tuple(1.0 / len(values) for _ in values)
    return tuple(prob / total for prob in values)


def write_transition_caches(path: Path, caches: Iterable[TransitionCache]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(transition_cache_to_dict(cache), ensure_ascii=False) for cache in caches)
    path.write_text(payload + "\n", encoding="utf-8")


def _transition_row_from_dict(row: dict[str, Any]) -> SparseTransitionRow:
    return SparseTransitionRow(
        position=int(row["position"]),
        source_token_id=int(row["source_token_id"]),
        source_text=str(row.get("source_text", "")),
        top_token_ids=tuple(int(token_id) for token_id in row["top_token_ids"]),
        top_probs=tuple(float(prob) for prob in row["top_probs"]),
        top_texts=tuple(str(text) for text in row.get("top_texts", [])),
    )


def _transition_row_to_dict(row: SparseTransitionRow) -> dict[str, Any]:
    return {
        "position": row.position,
        "source_token_id": row.source_token_id,
        "source_text": row.source_text,
        "top_token_ids": list(row.top_token_ids),
        "top_probs": list(row.top_probs),
        "top_texts": list(row.top_texts),
    }
