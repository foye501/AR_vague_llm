from __future__ import annotations

import argparse
import json
from pathlib import Path

from .corruption import Strategy, corrupt_example, load_examples

STRATEGIES: tuple[Strategy, ...] = ("mask", "random", "embedding", "ar_guided")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate corrupted meeting-summary examples.")
    parser.add_argument("--input", type=Path, default=Path("data/meeting_summaries_seed.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/corruptions.jsonl"))
    parser.add_argument("--beta", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Maximum clean examples to process; 0 means all.")
    parser.add_argument("--format", choices=("jsonl", "markdown"), default="jsonl")
    args = parser.parse_args()

    examples = load_examples(args.input.read_text(encoding="utf-8").splitlines())
    if args.limit:
        examples = examples[: args.limit]

    rows = []
    for example_index, example in enumerate(examples):
        for strategy_index, strategy in enumerate(STRATEGIES):
            rows.append(
                corrupt_example(
                    example,
                    strategy,
                    beta=args.beta,
                    seed=args.seed + example_index * 101 + strategy_index,
                )
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "markdown":
        args.output.write_text(_to_markdown(rows), encoding="utf-8")
    else:
        payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
        args.output.write_text(payload, encoding="utf-8")


def _to_markdown(rows: list[dict[str, str]]) -> str:
    grouped: dict[str, dict[str, str]] = {}
    for row in rows:
        item = grouped.setdefault(
            row["id"],
            {
                "clean": _compact(row["clean_summary"]),
                "mask": "",
                "random": "",
                "embedding": "",
                "ar_guided": "",
            },
        )
        item[row["strategy"]] = _compact(row["corrupted_summary"])

    lines = [
        "# Corruption Inspection Table",
        "",
        "| ID | Clean | Mask | Random | Embedding | AR-guided |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for example_id, item in grouped.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape(example_id),
                    _escape(item["clean"]),
                    _escape(item["mask"]),
                    _escape(item["random"]),
                    _escape(item["embedding"]),
                    _escape(item["ar_guided"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _compact(value: str) -> str:
    return " ".join(value.split())


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


if __name__ == "__main__":
    main()
