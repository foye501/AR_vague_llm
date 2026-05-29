from __future__ import annotations

import argparse
import json
from pathlib import Path

KEY_METRICS = (
    "sql_exact_match",
    "sql_keyword_valid",
    "sql_repair_delta",
    "prediction_token_f1",
    "corrupted_token_f1",
    "token_f1_repair_delta",
    "prediction_line_f1",
    "line_f1_repair_delta",
    "prediction_heading_valid",
    "owner_term_accuracy",
    "deadline_term_accuracy",
    "decision_term_accuracy",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize denoiser metric JSON files as a Markdown table.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/metrics_summary.md"))
    parser.add_argument("metrics", nargs="+", type=Path)
    args = parser.parse_args()

    rows = [(path.stem.replace("metrics_", ""), json.loads(path.read_text(encoding="utf-8"))) for path in args.metrics]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(to_markdown(rows), encoding="utf-8")
    print(f"Wrote {args.output}")


def to_markdown(rows: list[tuple[str, dict[str, float | int]]]) -> str:
    lines = [
        "# Denoiser Metrics Summary",
        "",
        "| Run | Rows | SQL EM | SQL Valid | SQL Repair Delta | Token F1 | Schema F1 | Literal F1 | Operator F1 | Corrupted Token F1 | Token Repair Delta | Line F1 | Line Repair Delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, metrics in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(metrics.get("rows", 0)),
                    _fmt(metrics.get("sql_exact_match", 0.0)),
                    _fmt(metrics.get("sql_keyword_valid", 0.0)),
                    _fmt(metrics.get("sql_repair_delta", 0.0)),
                    _fmt(metrics.get("prediction_token_f1", 0.0)),
                    _fmt(metrics.get("schema_identifier_token_f1", 0.0)),
                    _fmt(metrics.get("literal_token_f1", 0.0)),
                    _fmt(metrics.get("operator_token_f1", 0.0)),
                    _fmt(metrics.get("corrupted_token_f1", 0.0)),
                    _fmt(metrics.get("token_f1_repair_delta", 0.0)),
                    _fmt(metrics.get("prediction_line_f1", 0.0)),
                    _fmt(metrics.get("line_f1_repair_delta", 0.0)),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _fmt(value: float | int) -> str:
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
