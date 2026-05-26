from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare text-to-SQL rows from b-mc2/sql-create-context.")
    parser.add_argument("--dataset", default="b-mc2/sql-create-context")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=Path("artifacts/sql_create_context_subset.jsonl"))
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-source-chars", type=int, default=5000)
    parser.add_argument("--max-target-chars", type=int, default=1000)
    args = parser.parse_args()

    _require_datasets()
    from datasets import load_dataset

    dataset = load_dataset(args.dataset, split=args.split)
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)

    rows: list[dict[str, str]] = []
    for index in indices:
        raw = dataset[int(index)]
        prepared = prepare_row(raw, row_id=f"sql-{index}", max_source_chars=args.max_source_chars, max_target_chars=args.max_target_chars)
        if prepared is None:
            continue
        rows.append(prepared)
        if args.max_examples and len(rows) >= args.max_examples:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({len(rows)} rows)")


def prepare_row(
    row: dict[str, Any],
    *,
    row_id: str,
    max_source_chars: int,
    max_target_chars: int,
) -> dict[str, str] | None:
    question = _first_present(row, "question", "prompt", "input")
    schema = _first_present(row, "context", "sql_context", "schema", "create_table", "create_context")
    sql = _first_present(row, "answer", "query", "sql", "target", "output")
    if not question or not schema or not sql:
        return None

    source = f"Question:\n{question}\n\nDatabase schema:\n{schema}"
    target = str(sql).strip()
    if len(source) > max_source_chars or len(target) > max_target_chars:
        return None

    # Keep legacy field names so the existing transition/training pipeline remains compatible.
    return {
        "id": row_id,
        "task": "text_to_sql",
        "source_text": source,
        "target_text": target,
        "transcript": source,
        "clean_summary": target,
        "question": str(question).strip(),
        "schema": str(schema).strip(),
        "sql": target,
    }


def _first_present(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _require_datasets() -> None:
    try:
        import datasets  # noqa: F401
    except ImportError as exc:
        raise SystemExit('Install training dependencies first: python -m pip install -e ".[train,dev]"') from exc


if __name__ == "__main__":
    main()
