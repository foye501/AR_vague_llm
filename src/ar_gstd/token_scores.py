from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TokenScoreRow:
    position: int
    token_id: int
    token_text: str
    category: str
    gold_prob: float
    surprisal: float
    rank: int
    top1_prob: float
    top2_prob: float
    margin: float
    topk_entropy: float


@dataclass(frozen=True)
class TokenScoreCache:
    example_id: str
    transcript: str
    clean_summary: str
    tokenizer_name: str
    teacher_model: str
    target_token_ids: tuple[int, ...]
    rows: tuple[TokenScoreRow, ...]


def load_token_score_caches(path: Path) -> dict[str, TokenScoreCache]:
    return {
        cache.example_id: cache
        for cache in (
            token_score_cache_from_dict(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def write_token_score_caches(path: Path, caches: Iterable[TokenScoreCache]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(token_score_cache_to_dict(cache), ensure_ascii=False) for cache in caches)
    path.write_text(payload + "\n", encoding="utf-8")


def token_score_cache_from_dict(row: dict) -> TokenScoreCache:
    return TokenScoreCache(
        example_id=str(row["id"]),
        transcript=str(row["transcript"]),
        clean_summary=str(row["clean_summary"]),
        tokenizer_name=str(row["tokenizer_name"]),
        teacher_model=str(row["teacher_model"]),
        target_token_ids=tuple(int(token_id) for token_id in row["target_token_ids"]),
        rows=tuple(token_score_row_from_dict(item) for item in row["rows"]),
    )


def token_score_cache_to_dict(cache: TokenScoreCache) -> dict:
    return {
        "id": cache.example_id,
        "transcript": cache.transcript,
        "clean_summary": cache.clean_summary,
        "tokenizer_name": cache.tokenizer_name,
        "teacher_model": cache.teacher_model,
        "target_token_ids": list(cache.target_token_ids),
        "rows": [token_score_row_to_dict(row) for row in cache.rows],
    }


def token_score_row_from_dict(row: dict) -> TokenScoreRow:
    return TokenScoreRow(
        position=int(row["position"]),
        token_id=int(row["token_id"]),
        token_text=str(row["token_text"]),
        category=str(row.get("category", "other")),
        gold_prob=float(row["gold_prob"]),
        surprisal=float(row["surprisal"]),
        rank=int(row["rank"]),
        top1_prob=float(row["top1_prob"]),
        top2_prob=float(row["top2_prob"]),
        margin=float(row["margin"]),
        topk_entropy=float(row["topk_entropy"]),
    )


def token_score_row_to_dict(row: TokenScoreRow) -> dict:
    return {
        "position": row.position,
        "token_id": row.token_id,
        "token_text": row.token_text,
        "category": row.category,
        "gold_prob": row.gold_prob,
        "surprisal": row.surprisal,
        "rank": row.rank,
        "top1_prob": row.top1_prob,
        "top2_prob": row.top2_prob,
        "margin": row.margin,
        "topk_entropy": row.topk_entropy,
    }


def difficulty_weights(
    rows: tuple[TokenScoreRow, ...],
    *,
    strength: float,
    floor: float = 0.5,
    ceiling: float = 2.0,
) -> dict[int, float]:
    if strength <= 0 or not rows:
        return {row.position: 1.0 for row in rows}
    values = [row.surprisal for row in rows]
    min_value = min(values)
    max_value = max(values)
    if math.isclose(min_value, max_value):
        return {row.position: 1.0 for row in rows}
    weights: dict[int, float] = {}
    for row in rows:
        normalized = (row.surprisal - min_value) / (max_value - min_value)
        weight = 1.0 + strength * (normalized - 0.5) * 2.0
        weights[row.position] = min(max(weight, floor), ceiling)
    return weights


def adaptive_mask_probability(base_mask_prob: float, difficulty_weight: float) -> float:
    return min(max(base_mask_prob * difficulty_weight, 0.0), 1.0)
