from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BidirectionalDenoisingFeatures:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    target_start: int
    target_length: int


def build_bidirectional_denoising_features(
    row: dict[str, Any],
    *,
    tokenizer,
    max_sequence_length: int,
    max_target_length: int,
) -> BidirectionalDenoisingFeatures:
    clean_ids, corrupted_ids = aligned_target_token_ids(row, tokenizer=tokenizer, max_target_length=max_target_length)
    if not clean_ids:
        raise ValueError("target token sequence is empty")
    prefix_ids = build_prefix_ids(
        row,
        tokenizer=tokenizer,
        max_prefix_length=max_sequence_length - len(corrupted_ids),
    )
    if len(prefix_ids) + len(corrupted_ids) > max_sequence_length:
        target_budget = max(1, max_sequence_length - len(prefix_ids))
        clean_ids = clean_ids[:target_budget]
        corrupted_ids = corrupted_ids[:target_budget]

    input_ids = prefix_ids + corrupted_ids
    labels = [-100] * len(prefix_ids) + clean_ids
    return BidirectionalDenoisingFeatures(
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        labels=labels,
        target_start=len(prefix_ids),
        target_length=len(clean_ids),
    )


def aligned_target_token_ids(
    row: dict[str, Any],
    *,
    tokenizer,
    max_target_length: int,
) -> tuple[list[int], list[int]]:
    clean_ids = _token_ids_from_row(row, key="clean_token_ids", text_key="clean_summary", tokenizer=tokenizer)
    corrupted_ids = _token_ids_from_row(row, key="corrupted_token_ids", text_key="corrupted_summary", tokenizer=tokenizer)
    length = min(len(clean_ids), len(corrupted_ids), max_target_length)
    return clean_ids[:length], corrupted_ids[:length]


def build_prefix_ids(row: dict[str, Any], *, tokenizer, max_prefix_length: int) -> list[int]:
    if max_prefix_length <= 0:
        return []
    metadata = ""
    if row.get("timestep") is not None and row.get("num_steps") is not None:
        metadata += f"Diffusion timestep: {row['timestep']}/{row['num_steps']}\n"
    if row.get("noise_kind") or row.get("strategy"):
        metadata += f"Noise process: {row.get('noise_kind') or row.get('strategy')}\n"
    prefix_text = (
        "Bidirectionally denoise the target output using the source context.\n\n"
        f"{metadata}"
        f"Source context:\n{row['transcript']}"
    )
    suffix_text = "\n\nNoisy target output:\n"
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
    source_budget = max(0, max_prefix_length - len(suffix_ids))
    source_ids = tokenizer.encode(
        prefix_text,
        add_special_tokens=False,
        truncation=True,
        max_length=source_budget,
    )
    return source_ids + suffix_ids[-max_prefix_length:]


def _token_ids_from_row(row: dict[str, Any], *, key: str, text_key: str, tokenizer) -> list[int]:
    if key in row and row[key] is not None:
        return [int(token_id) for token_id in row[key]]
    return [int(token_id) for token_id in tokenizer.encode(str(row[text_key]), add_special_tokens=False)]
