from __future__ import annotations

import argparse
import json
from pathlib import Path

from .corruption import load_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize clean source-to-target rows for SFT/base evaluation.")
    parser.add_argument("--input", type=Path, default=Path("artifacts/sql_create_context_subset.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/train_pairs_clean_sft.jsonl"))
    args = parser.parse_args()

    examples = load_examples(args.input.read_text(encoding="utf-8").splitlines())
    rows = [
        {
            "id": example.example_id,
            "strategy": "clean_sft",
            "prompt_mode": "generate",
            "transcript": example.transcript,
            "corrupted_summary": "",
            "clean_summary": example.clean_summary,
        }
        for example in examples
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
