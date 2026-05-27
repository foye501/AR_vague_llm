from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SAMPLES = (
    "SELECT age FROM singer WHERE name = 'Alice'",
    "CREATE TABLE singer (id INTEGER, name TEXT, release_year INTEGER)",
    "The owner approved the migration deadline after the final review.",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect compatibility between teacher and student tokenizers.")
    parser.add_argument("--teacher-tokenizer", required=True)
    parser.add_argument("--student-tokenizer", required=True)
    parser.add_argument(
        "--max-vocab-scan",
        type=int,
        default=0,
        help="Maximum token ids to scan for projection stats. Use 0 to scan the full tokenizer length.",
    )
    parser.add_argument("--sample", action="append", default=[], help="Extra text sample to tokenize.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    args = parser.parse_args()

    _require_transformers()
    from transformers import AutoTokenizer

    teacher = AutoTokenizer.from_pretrained(args.teacher_tokenizer, trust_remote_code=True)
    student = AutoTokenizer.from_pretrained(args.student_tokenizer, trust_remote_code=True)

    samples = tuple(DEFAULT_SAMPLES) + tuple(args.sample)
    report = build_report(
        teacher_name=args.teacher_tokenizer,
        student_name=args.student_tokenizer,
        teacher=teacher,
        student=student,
        samples=samples,
        max_vocab_scan=args.max_vocab_scan,
    )

    markdown = report_to_markdown(report)
    print(markdown)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown + "\n", encoding="utf-8")


def build_report(
    *,
    teacher_name: str,
    student_name: str,
    teacher,
    student,
    samples: tuple[str, ...],
    max_vocab_scan: int,
) -> dict[str, Any]:
    teacher_raw_vocab = set(teacher.get_vocab())
    student_raw_vocab = set(student.get_vocab())
    raw_overlap = _overlap_stats(teacher_raw_vocab, student_raw_vocab)

    teacher_decoded, teacher_decoded_stats = _decoded_vocab_set(teacher, max_vocab_scan=max_vocab_scan)
    student_decoded, student_decoded_stats = _decoded_vocab_set(student, max_vocab_scan=max_vocab_scan)
    decoded_overlap = _overlap_stats(teacher_decoded, student_decoded)

    teacher_to_student = _projection_stats(
        source=teacher,
        target=student,
        max_vocab_scan=max_vocab_scan,
    )
    student_to_teacher = _projection_stats(
        source=student,
        target=teacher,
        max_vocab_scan=max_vocab_scan,
    )

    return {
        "teacher": _describe_tokenizer(teacher, teacher_name),
        "student": _describe_tokenizer(student, student_name),
        "same_tokenizer_name": teacher_name == student_name,
        "raw_vocab_overlap": raw_overlap,
        "decoded_vocab_overlap": decoded_overlap,
        "teacher_decoded_vocab_stats": teacher_decoded_stats,
        "student_decoded_vocab_stats": student_decoded_stats,
        "teacher_to_student_projection": teacher_to_student,
        "student_to_teacher_projection": student_to_teacher,
        "samples": [
            {
                "text": text,
                "teacher": _tokenize_sample(teacher, text),
                "student": _tokenize_sample(student, text),
            }
            for text in samples
        ],
        "interpretation": _interpret(
            same_name=teacher_name == student_name,
            decoded_jaccard=decoded_overlap["jaccard"],
            teacher_to_student_ratio=teacher_to_student["single_token_ratio"],
        ),
    }


def report_to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Tokenizer Compatibility Report",
        "",
        f"Teacher tokenizer: `{report['teacher']['name']}`",
        f"Student tokenizer: `{report['student']['name']}`",
        "",
        "## Summary",
        "",
        f"- Same tokenizer name: `{report['same_tokenizer_name']}`",
        f"- Teacher class: `{report['teacher']['class_name']}`",
        f"- Student class: `{report['student']['class_name']}`",
        f"- Teacher vocab size / len: `{report['teacher']['vocab_size']}` / `{report['teacher']['length']}`",
        f"- Student vocab size / len: `{report['student']['vocab_size']}` / `{report['student']['length']}`",
        f"- Teacher mask token: `{report['teacher']['mask_token']}`",
        f"- Student mask token: `{report['student']['mask_token']}`",
        f"- Raw vocab Jaccard: `{report['raw_vocab_overlap']['jaccard']:.6f}`",
        f"- Decoded single-token text Jaccard: `{report['decoded_vocab_overlap']['jaccard']:.6f}`",
        "- Teacher token fragments that become exactly one student token: "
        f"`{report['teacher_to_student_projection']['single_token_ratio']:.4f}`",
        "- Student token fragments that become exactly one teacher token: "
        f"`{report['student_to_teacher_projection']['single_token_ratio']:.4f}`",
        "",
        f"Interpretation: **{report['interpretation']}**",
        "",
        "## Tokenization Samples",
        "",
        "| Text | Teacher tokens | Student tokens | Ratio student/teacher |",
        "| --- | ---: | ---: | ---: |",
    ]
    for item in report["samples"]:
        teacher_count = item["teacher"]["token_count"]
        student_count = item["student"]["token_count"]
        ratio = student_count / teacher_count if teacher_count else 0.0
        text = item["text"].replace("|", "\\|")
        lines.append(f"| `{text}` | {teacher_count} | {student_count} | {ratio:.3f} |")

    lines.extend(["", "## Projection Stats", ""])
    for title, key in (
        ("Teacher -> Student", "teacher_to_student_projection"),
        ("Student -> Teacher", "student_to_teacher_projection"),
    ):
        stats = report[key]
        lines.extend(
            [
                f"### {title}",
                "",
                f"- Scanned non-special, non-empty source tokens: `{stats['source_tokens_scanned']}`",
                f"- Exactly one target token: `{stats['single_token']}`",
                f"- Multiple target tokens: `{stats['multi_token']}`",
                f"- Zero target tokens: `{stats['zero_token']}`",
                f"- Single-token ratio: `{stats['single_token_ratio']:.4f}`",
                "",
            ]
        )

    lines.extend(["## Sample Pieces", ""])
    for item in report["samples"]:
        lines.extend(
            [
                f"### `{item['text']}`",
                "",
                f"Teacher pieces: `{_preview(item['teacher']['pieces'])}`",
                "",
                f"Student pieces: `{_preview(item['student']['pieces'])}`",
                "",
            ]
        )

    return "\n".join(lines)


def _describe_tokenizer(tokenizer, name: str) -> dict[str, Any]:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_model = ""
    backend_pre_tokenizer = ""
    if backend is not None:
        backend_model = type(getattr(backend, "model", None)).__name__
        backend_pre_tokenizer = str(getattr(backend, "pre_tokenizer", ""))
    return {
        "name": name,
        "class_name": type(tokenizer).__name__,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "backend_model": backend_model,
        "backend_pre_tokenizer": backend_pre_tokenizer,
        "vocab_size": int(getattr(tokenizer, "vocab_size", len(tokenizer))),
        "length": int(len(tokenizer)),
        "model_max_length": int(getattr(tokenizer, "model_max_length", 0)),
        "mask_token": _nullable_str(getattr(tokenizer, "mask_token", None)),
        "mask_token_id": getattr(tokenizer, "mask_token_id", None),
        "pad_token": _nullable_str(getattr(tokenizer, "pad_token", None)),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token": _nullable_str(getattr(tokenizer, "eos_token", None)),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "bos_token": _nullable_str(getattr(tokenizer, "bos_token", None)),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "special_tokens_map": _jsonable(getattr(tokenizer, "special_tokens_map", {})),
    }


def _decoded_vocab_set(tokenizer, *, max_vocab_scan: int) -> tuple[set[str], dict[str, int]]:
    limit = _scan_limit(tokenizer, max_vocab_scan)
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    decoded: set[str] = set()
    skipped_special = 0
    empty = 0
    failed = 0
    for token_id in range(limit):
        if token_id in special_ids:
            skipped_special += 1
            continue
        try:
            text = tokenizer.decode([token_id], skip_special_tokens=False)
        except Exception:
            failed += 1
            continue
        if text == "":
            empty += 1
            continue
        decoded.add(text)
    return decoded, {
        "scan_limit": limit,
        "decoded_unique": len(decoded),
        "skipped_special": skipped_special,
        "empty": empty,
        "failed": failed,
    }


def _projection_stats(*, source, target, max_vocab_scan: int) -> dict[str, Any]:
    limit = _scan_limit(source, max_vocab_scan)
    special_ids = set(getattr(source, "all_special_ids", []))
    source_tokens_scanned = 0
    single_token = 0
    multi_token = 0
    zero_token = 0
    examples: dict[str, list[dict[str, Any]]] = {"single": [], "multi": [], "zero": []}

    for source_id in range(limit):
        if source_id in special_ids:
            continue
        try:
            text = source.decode([source_id], skip_special_tokens=False)
        except Exception:
            continue
        if text == "":
            continue
        source_tokens_scanned += 1
        target_ids = target.encode(text, add_special_tokens=False)
        if len(target_ids) == 1:
            single_token += 1
            _append_example(examples["single"], source_id, text, target_ids)
        elif len(target_ids) == 0:
            zero_token += 1
            _append_example(examples["zero"], source_id, text, target_ids)
        else:
            multi_token += 1
            _append_example(examples["multi"], source_id, text, target_ids)

    return {
        "scan_limit": limit,
        "source_tokens_scanned": source_tokens_scanned,
        "single_token": single_token,
        "multi_token": multi_token,
        "zero_token": zero_token,
        "single_token_ratio": single_token / source_tokens_scanned if source_tokens_scanned else 0.0,
        "examples": examples,
    }


def _tokenize_sample(tokenizer, text: str) -> dict[str, Any]:
    ids = [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    if isinstance(tokens, str):
        tokens = [tokens]
    pieces = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in ids]
    return {
        "token_count": len(ids),
        "ids": ids,
        "tokens": [str(token) for token in tokens],
        "pieces": pieces,
    }


def _overlap_stats(left: set[str], right: set[str]) -> dict[str, Any]:
    intersection = left & right
    union = left | right
    return {
        "left": len(left),
        "right": len(right),
        "intersection": len(intersection),
        "union": len(union),
        "jaccard": len(intersection) / len(union) if union else 1.0,
        "intersection_examples": sorted(intersection)[:25],
    }


def _interpret(*, same_name: bool, decoded_jaccard: float, teacher_to_student_ratio: float) -> str:
    if same_name:
        return "same tokenizer; AR transition rows are in the same discrete state space."
    if decoded_jaccard >= 0.9:
        return "different names but near-identical decoded token vocabulary."
    if teacher_to_student_ratio < 0.5:
        return (
            "highly lossy tokenizer bridge; previous projected token-level AR transitions are useful only as "
            "a weak text-fragment proxy, not as clean same-vocabulary logit distillation."
        )
    return (
        "different tokenizers with partial single-token projection; projected runs are interpretable, "
        "but same-tokenizer experiments are cleaner for a publication claim."
    )


def _scan_limit(tokenizer, max_vocab_scan: int) -> int:
    length = len(tokenizer)
    if max_vocab_scan <= 0:
        return length
    return min(max_vocab_scan, length)


def _append_example(bucket: list[dict[str, Any]], source_id: int, text: str, target_ids: list[int]) -> None:
    if len(bucket) >= 8:
        return
    bucket.append({"source_id": int(source_id), "text": text, "target_ids": [int(token_id) for token_id in target_ids]})


def _preview(items: list[str], max_items: int = 40) -> str:
    shown = items[:max_items]
    suffix = " ..." if len(items) > max_items else ""
    return " | ".join(item.replace("`", "\\`") for item in shown) + suffix


def _nullable_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _require_transformers() -> None:
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit('Install training dependencies first: python -m pip install -e ".[train,dev]"') from exc


if __name__ == "__main__":
    main()
