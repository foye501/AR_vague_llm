from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from statistics import mean

from .train_seq2seq_denoiser import build_denoising_prompt

REQUIRED_HEADINGS = ("## Key Decisions", "## Risks and Open Issues", "## To-do")
OWNER_TERMS = ("Kevin", "Maya", "Alex")
DEADLINE_TERMS = ("by Friday", "by Monday", "next week", "later", "after review")
DECISION_TERMS = ("decided", "discussed", "considered", "proposed", "agreed", "resolved", "committed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and score denoiser predictions.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--output-predictions", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-source-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    args = parser.parse_args()

    _require_train_deps()
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    rows = [json.loads(line) for line in args.eval_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"No rows found in {args.eval_file}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    device = _resolve_device(args.device, torch)
    model.to(device)
    model.eval()

    predictions: list[dict[str, str]] = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        prompts = [build_denoising_prompt(row["transcript"], row["corrupted_summary"]) for row in batch]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_source_length,
        ).to(device)
        with torch.no_grad():
            generated = model.generate(**encoded, max_new_tokens=args.max_new_tokens)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for row, prediction in zip(batch, decoded, strict=True):
            predictions.append(
                {
                    "id": row["id"],
                    "strategy": row.get("strategy", ""),
                    "transcript": row["transcript"],
                    "corrupted_summary": row["corrupted_summary"],
                    "prediction": prediction,
                    "clean_summary": row["clean_summary"],
                }
            )

    metrics = score_predictions(predictions)
    args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.output_predictions.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in predictions) + "\n",
        encoding="utf-8",
    )
    args.output_metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def score_predictions(rows: list[dict[str, str]]) -> dict[str, float | int]:
    corrupted_token_f1 = [_token_f1(row["corrupted_summary"], row["clean_summary"]) for row in rows]
    prediction_token_f1 = [_token_f1(row["prediction"], row["clean_summary"]) for row in rows]
    corrupted_line_f1 = [_line_f1(row["corrupted_summary"], row["clean_summary"]) for row in rows]
    prediction_line_f1 = [_line_f1(row["prediction"], row["clean_summary"]) for row in rows]

    return {
        "rows": len(rows),
        "prediction_exact_match": _rate(_normalize(row["prediction"]) == _normalize(row["clean_summary"]) for row in rows),
        "sql_exact_match": _rate(_normalize_sql(row["prediction"]) == _normalize_sql(row["clean_summary"]) for row in rows),
        "sql_keyword_valid": _rate(_looks_like_sql(row["prediction"]) for row in rows),
        "sql_repair_delta": _rate(_normalize_sql(row["prediction"]) == _normalize_sql(row["clean_summary"]) for row in rows)
        - _rate(_normalize_sql(row["corrupted_summary"]) == _normalize_sql(row["clean_summary"]) for row in rows),
        "prediction_heading_valid": _rate(all(heading in row["prediction"] for heading in REQUIRED_HEADINGS) for row in rows),
        "prediction_token_f1": mean(prediction_token_f1),
        "corrupted_token_f1": mean(corrupted_token_f1),
        "token_f1_repair_delta": mean(prediction_token_f1) - mean(corrupted_token_f1),
        "prediction_line_f1": mean(prediction_line_f1),
        "corrupted_line_f1": mean(corrupted_line_f1),
        "line_f1_repair_delta": mean(prediction_line_f1) - mean(corrupted_line_f1),
        "owner_term_accuracy": _field_accuracy(rows, OWNER_TERMS),
        "deadline_term_accuracy": _field_accuracy(rows, DEADLINE_TERMS),
        "decision_term_accuracy": _field_accuracy(rows, DECISION_TERMS),
    }


def _field_accuracy(rows: list[dict[str, str]], terms: tuple[str, ...]) -> float:
    return _rate(_present_terms(row["prediction"], terms) == _present_terms(row["clean_summary"], terms) for row in rows)


def _present_terms(text: str, terms: tuple[str, ...]) -> tuple[str, ...]:
    lowered = text.lower()
    return tuple(term.lower() for term in terms if term.lower() in lowered)


def _token_f1(candidate: str, target: str) -> float:
    candidate_tokens = _tokens(candidate)
    target_tokens = _tokens(target)
    if not candidate_tokens and not target_tokens:
        return 1.0
    if not candidate_tokens or not target_tokens:
        return 0.0
    overlap = sum((Counter(candidate_tokens) & Counter(target_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def _line_f1(candidate: str, target: str) -> float:
    candidate_lines = set(_summary_lines(candidate))
    target_lines = set(_summary_lines(target))
    if not candidate_lines and not target_lines:
        return 1.0
    if not candidate_lines or not target_lines:
        return 0.0
    overlap = len(candidate_lines & target_lines)
    if overlap == 0:
        return 0.0
    precision = overlap / len(candidate_lines)
    recall = overlap / len(target_lines)
    return 2 * precision * recall / (precision + recall)


def _summary_lines(text: str) -> list[str]:
    return [_normalize(line) for line in text.splitlines() if line.strip().startswith("-")]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _normalize_sql(text: str) -> str:
    text = text.strip().rstrip(";").lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([(),=<>+\-*/])\s*", r"\1", text)
    return text


def _looks_like_sql(text: str) -> bool:
    normalized = _normalize_sql(text)
    if not normalized.startswith("select "):
        return False
    if " from " not in f" {normalized} ":
        return False
    return normalized.count("(") == normalized.count(")")


def _rate(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def _resolve_device(device: str, torch):
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _require_train_deps() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit('Install training dependencies first: python -m pip install -e ".[train,dev]"') from exc


if __name__ == "__main__":
    main()
