from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze_transition_cache import schema_identifiers


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter evaluation rows by target token category.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category", choices=("schema_identifier",), default="schema_identifier")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    filtered = [row for row in rows if has_schema_identifier_target(row)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in filtered) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({len(filtered)} / {len(rows)} rows)")


def has_schema_identifier_target(row: dict[str, str]) -> bool:
    identifiers = schema_identifiers(row["transcript"])
    target = row["clean_summary"].lower()
    return any(identifier in target for identifier in identifiers)


if __name__ == "__main__":
    main()
