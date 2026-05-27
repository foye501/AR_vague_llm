from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

from .transitions import TransitionCache, load_transition_caches

SQL_KEYWORDS = {
    "select",
    "from",
    "where",
    "and",
    "or",
    "group",
    "by",
    "order",
    "having",
    "limit",
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "join",
    "on",
    "as",
    "distinct",
}


@dataclass
class Bucket:
    rows: int = 0
    gold_in_topk: int = 0
    gold_rank_sum: float = 0.0
    gold_prob_sum: float = 0.0
    entropy_sum: float = 0.0

    def add(self, *, gold_rank: int | None, gold_prob: float, entropy: float) -> None:
        self.rows += 1
        if gold_rank is not None:
            self.gold_in_topk += 1
            self.gold_rank_sum += gold_rank
        self.gold_prob_sum += gold_prob
        self.entropy_sum += entropy

    def to_metrics(self) -> dict[str, float | int]:
        if self.rows == 0:
            return {
                "rows": 0,
                "gold_topk_rate": 0.0,
                "mean_gold_rank": 0.0,
                "mean_gold_prob": 0.0,
                "mean_entropy": 0.0,
            }
        return {
            "rows": self.rows,
            "gold_topk_rate": self.gold_in_topk / self.rows,
            "mean_gold_rank": self.gold_rank_sum / self.gold_in_topk if self.gold_in_topk else 0.0,
            "mean_gold_prob": self.gold_prob_sum / self.rows,
            "mean_entropy": self.entropy_sum / self.rows,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze conditional vs fixed sparse transition caches.")
    parser.add_argument("--conditional-cache", type=Path, default=Path("artifacts/ar_transition_cache.jsonl"))
    parser.add_argument("--fixed-cache", type=Path, default=Path("artifacts/fixed_transition_cache.jsonl"))
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/transition_cache_analysis.json"))
    parser.add_argument("--output-markdown", type=Path, default=Path("artifacts/transition_cache_analysis.md"))
    args = parser.parse_args()

    conditional = load_transition_caches(args.conditional_cache)
    fixed = load_transition_caches(args.fixed_cache)
    analysis = {
        "conditional": analyze_caches(conditional),
        "fixed": analyze_caches(fixed),
    }
    analysis["delta"] = diff_analysis(analysis["conditional"], analysis["fixed"])

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_markdown.write_text(to_markdown(analysis), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_markdown}")


def analyze_caches(caches: list[TransitionCache]) -> dict[str, dict[str, float | int]]:
    buckets: dict[str, Bucket] = defaultdict(Bucket)
    for cache in caches:
        schema_terms = schema_identifiers(cache.transcript)
        for row in cache.rows:
            category = token_category(row.source_text, schema_terms)
            gold_rank, gold_prob = gold_stats(row.source_token_id, row.top_token_ids, row.top_probs)
            entropy = row_entropy(row.top_probs)
            buckets["all"].add(gold_rank=gold_rank, gold_prob=gold_prob, entropy=entropy)
            buckets[category].add(gold_rank=gold_rank, gold_prob=gold_prob, entropy=entropy)
    return {name: bucket.to_metrics() for name, bucket in sorted(buckets.items())}


def diff_analysis(
    conditional: dict[str, dict[str, float | int]],
    fixed: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for category in sorted(set(conditional) | set(fixed)):
        cond = conditional.get(category, {})
        base = fixed.get(category, {})
        result[category] = {
            "gold_topk_rate_delta": float(cond.get("gold_topk_rate", 0.0)) - float(base.get("gold_topk_rate", 0.0)),
            "mean_gold_prob_delta": float(cond.get("mean_gold_prob", 0.0)) - float(base.get("mean_gold_prob", 0.0)),
            "mean_entropy_delta": float(cond.get("mean_entropy", 0.0)) - float(base.get("mean_entropy", 0.0)),
        }
    return result


def gold_stats(source_token_id: int, top_token_ids: tuple[int, ...], top_probs: tuple[float, ...]) -> tuple[int | None, float]:
    for index, token_id in enumerate(top_token_ids):
        if token_id == source_token_id:
            return index + 1, float(top_probs[index]) if index < len(top_probs) else 0.0
    return None, 0.0


def row_entropy(probs: tuple[float, ...]) -> float:
    return -sum(prob * math.log(prob) for prob in probs if prob > 0)


def schema_identifiers(source_context: str) -> set[str]:
    identifiers: set[str] = set()
    for match in re.finditer(r"CREATE\s+TABLE\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)", source_context, flags=re.I | re.S):
        identifiers.add(match.group(1).lower())
        for column in match.group(2).split(","):
            column = column.strip()
            column_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", column)
            if column_match:
                identifiers.add(column_match.group(1).lower())
    return identifiers


def token_category(token_text: str, schema_terms: set[str]) -> str:
    normalized = normalize_token(token_text)
    if not normalized:
        return "punct_or_space"
    if normalized in SQL_KEYWORDS:
        return "sql_keyword"
    if normalized in schema_terms:
        return "schema_identifier"
    if normalized.isdigit():
        return "literal"
    if re.fullmatch(r"[<>=!]+", normalized):
        return "operator"
    return "other"


def normalize_token(token_text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_<>!=]+", "", token_text).lower()


def to_markdown(analysis: dict[str, dict]) -> str:
    categories = sorted(set(analysis["conditional"]) | set(analysis["fixed"]))
    lines = [
        "# Transition Cache Analysis",
        "",
        "| Category | Rows | Cond Gold@K | Fixed Gold@K | Delta | Cond Gold Prob | Fixed Gold Prob | Delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category in categories:
        cond = analysis["conditional"].get(category, {})
        fixed = analysis["fixed"].get(category, {})
        delta = analysis["delta"].get(category, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    category,
                    str(cond.get("rows", fixed.get("rows", 0))),
                    _fmt(cond.get("gold_topk_rate", 0.0)),
                    _fmt(fixed.get("gold_topk_rate", 0.0)),
                    _fmt(delta.get("gold_topk_rate_delta", 0.0)),
                    _fmt(cond.get("mean_gold_prob", 0.0)),
                    _fmt(fixed.get("mean_gold_prob", 0.0)),
                    _fmt(delta.get("mean_gold_prob_delta", 0.0)),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _fmt(value: float | int) -> str:
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
