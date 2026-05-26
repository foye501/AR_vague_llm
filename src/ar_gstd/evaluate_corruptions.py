from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

REQUIRED_HEADINGS = ("## Key Decisions", "## Risks and Open Issues", "## To-do")
OWNER_TERMS = ("Kevin", "Maya", "Alex")
DEADLINE_TERMS = ("by Friday", "by Monday")
ACTION_TERMS = ("test", "review", "update")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate corruption artifact sanity metrics.")
    parser.add_argument("--input", type=Path, default=Path("artifacts/corruptions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/corruption_metrics.md"))
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    metrics = _compute_metrics(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_to_markdown(metrics), encoding="utf-8")


def _compute_metrics(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    metrics: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        strategy = row["strategy"]
        clean = row["clean_summary"]
        corrupted = row["corrupted_summary"]
        bucket = metrics[strategy]

        bucket["rows"] += 1
        bucket["changed"] += int(clean != corrupted)
        bucket["heading_valid"] += int(all(heading in corrupted for heading in REQUIRED_HEADINGS))
        bucket["contains_mask"] += int("[MASK]" in corrupted)
        bucket["owner_changed"] += int(_term_presence_changed(clean, corrupted, OWNER_TERMS))
        bucket["deadline_changed"] += int(_term_presence_changed(clean, corrupted, DEADLINE_TERMS))
        bucket["action_changed"] += int(_term_presence_changed(clean, corrupted, ACTION_TERMS))
    return {strategy: dict(values) for strategy, values in metrics.items()}


def _term_presence_changed(clean: str, corrupted: str, terms: tuple[str, ...]) -> bool:
    return any((term in clean) != (term in corrupted) for term in terms)


def _to_markdown(metrics: dict[str, dict[str, int]]) -> str:
    lines = [
        "# Corruption Metrics",
        "",
        "| Strategy | Rows | Changed | Heading valid | Contains mask | Owner changed | Deadline changed | Action changed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy in sorted(metrics):
        values = metrics[strategy]
        lines.append(
            "| "
            + " | ".join(
                [
                    strategy,
                    str(values.get("rows", 0)),
                    str(values.get("changed", 0)),
                    str(values.get("heading_valid", 0)),
                    str(values.get("contains_mask", 0)),
                    str(values.get("owner_changed", 0)),
                    str(values.get("deadline_changed", 0)),
                    str(values.get("action_changed", 0)),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
