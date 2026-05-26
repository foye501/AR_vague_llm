from __future__ import annotations

from dataclasses import dataclass
import random
import re
from typing import Iterable, Literal

Strategy = Literal["mask", "random", "embedding", "ar_guided"]


@dataclass(frozen=True)
class Example:
    example_id: str
    transcript: str
    clean_summary: str


@dataclass(frozen=True)
class SpanRule:
    label: str
    pattern: str
    mask: str
    random_tokens: tuple[str, ...]
    embedding_neighbors: tuple[str, ...]
    ar_guided_candidates: tuple[str, ...]


SPAN_RULES: tuple[SpanRule, ...] = (
    SpanRule(
        label="decision_priority",
        pattern=r"\bdecided to prioritize\b",
        mask="[MASK]",
        random_tokens=("umbrella cabinet", "oxygen ladder", "marble violet"),
        embedding_neighbors=("agreed to focus on", "resolved to emphasize", "committed to advance"),
        ar_guided_candidates=("discussed prioritizing", "considered prioritizing", "proposed prioritizing"),
    ),
    SpanRule(
        label="decision_release_delay",
        pattern=r"\bdecided to delay the release\b",
        mask="[MASK]",
        random_tokens=("oxygen fold the ocean", "marble paint the ladder", "umbrella slice the calendar"),
        embedding_neighbors=("agreed to postpone the release", "resolved to defer the launch", "committed to move the release"),
        ar_guided_candidates=("discussed release timing", "considered delaying the rollout", "proposed reviewing the launch plan"),
    ),
    SpanRule(
        label="decision_status",
        pattern=r"\bdecided to\b",
        mask="[MASK]",
        random_tokens=("umbrella", "oxygen", "marble"),
        embedding_neighbors=("agreed to", "resolved to", "committed to"),
        ar_guided_candidates=("discussed", "considered", "proposed"),
    ),
    SpanRule(
        label="priority",
        pattern=r"\bprioritize\b",
        mask="[MASK]",
        random_tokens=("ladder", "violet", "cabinet"),
        embedding_neighbors=("focus on", "emphasize", "advance"),
        ar_guided_candidates=("consider", "review", "defer"),
    ),
    SpanRule(
        label="owner_kevin",
        pattern=r"\bKevin\b",
        mask="[MASK]",
        random_tokens=("Mercury", "chair", "river"),
        embedding_neighbors=("the engineer", "the owner", "the assignee"),
        ar_guided_candidates=("someone", "the team", "the PM"),
    ),
    SpanRule(
        label="owner_maya",
        pattern=r"\bMaya\b",
        mask="[MASK]",
        random_tokens=("Saturn", "window", "pencil"),
        embedding_neighbors=("the analyst", "the reviewer", "the lead"),
        ar_guided_candidates=("someone", "the team", "the PM"),
    ),
    SpanRule(
        label="owner_alex",
        pattern=r"\bAlex\b",
        mask="[MASK]",
        random_tokens=("Neptune", "lamp", "forest"),
        embedding_neighbors=("the developer", "the maintainer", "the lead"),
        ar_guided_candidates=("someone", "the team", "the PM"),
    ),
    SpanRule(
        label="deadline_friday",
        pattern=r"\bby Friday\b",
        mask="[MASK]",
        random_tokens=("under glass", "near oxygen", "beside Tuesday"),
        embedding_neighbors=("by next week", "by Thursday", "before Friday"),
        ar_guided_candidates=("later", "next week", "after review"),
    ),
    SpanRule(
        label="deadline_monday",
        pattern=r"\bby Monday\b",
        mask="[MASK]",
        random_tokens=("inside marble", "near winter", "before chair"),
        embedding_neighbors=("by Tuesday", "early next week", "before Monday"),
        ar_guided_candidates=("later", "next week", "after review"),
    ),
    SpanRule(
        label="risk_unresolved",
        pattern=r"\bunresolved\b",
        mask="[MASK]",
        random_tokens=("blue", "square", "silent"),
        embedding_neighbors=("uncertain", "open", "pending"),
        ar_guided_candidates=("improving", "less severe", "acceptable"),
    ),
    SpanRule(
        label="action_test",
        pattern=r"\btest\b",
        mask="[MASK]",
        random_tokens=("orbit", "paint", "slice"),
        embedding_neighbors=("validate", "check", "verify"),
        ar_guided_candidates=("review", "follow up on", "monitor"),
    ),
    SpanRule(
        label="quality_asr",
        pattern=r"\bASR quality\b",
        mask="[MASK]",
        random_tokens=("weather quality", "chair quality", "planet quality"),
        embedding_neighbors=("transcription quality", "audio quality", "recognition quality"),
        ar_guided_candidates=("model quality", "summary quality", "audio quality"),
    ),
    SpanRule(
        label="release_delay",
        pattern=r"\bdelay the release\b",
        mask="[MASK]",
        random_tokens=("fold the ocean", "paint the ladder", "slice the calendar"),
        embedding_neighbors=("postpone the release", "defer the launch", "move the release"),
        ar_guided_candidates=("discuss the release timing", "delay the rollout", "review the launch plan"),
    ),
    SpanRule(
        label="fine_tuning",
        pattern=r"\bbefore model fine-tuning\b",
        mask="[MASK]",
        random_tokens=("under model thunder", "inside model carpet", "near model orange"),
        embedding_neighbors=("before tuning", "ahead of fine-tuning", "prior to fine-tuning"),
        ar_guided_candidates=("during model fine-tuning", "after model fine-tuning", "before deployment"),
    ),
)


def load_examples(lines: Iterable[str]) -> list[Example]:
    import json

    examples: list[Example] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        examples.append(
            Example(
                example_id=str(row["id"]),
                transcript=str(row["transcript"]),
                clean_summary=str(row["clean_summary"]),
            )
        )
    return examples


def corrupt_example(example: Example, strategy: Strategy, *, beta: float = 0.35, seed: int = 0) -> dict[str, str]:
    return {
        "id": example.example_id,
        "strategy": strategy,
        "transcript": example.transcript,
        "clean_summary": example.clean_summary,
        "corrupted_summary": corrupt_summary(example.clean_summary, strategy, beta=beta, seed=seed),
    }


def corrupt_summary(summary: str, strategy: Strategy, *, beta: float = 0.35, seed: int = 0) -> str:
    if not 0 <= beta <= 1:
        raise ValueError("beta must be between 0 and 1")
    if strategy not in ("mask", "random", "embedding", "ar_guided"):
        raise ValueError(f"unknown strategy: {strategy}")

    rng = random.Random(seed)
    corrupted = summary
    matches = _candidate_matches(summary)

    for rule, text in matches:
        if rng.random() > beta:
            continue
        replacement = _replacement(rule, strategy, rng)
        corrupted = _replace_once(corrupted, text, replacement)

    if corrupted == summary and matches:
        rule, text = matches[seed % len(matches)]
        corrupted = _replace_once(corrupted, text, _replacement(rule, strategy, rng))

    return corrupted


def _candidate_matches(summary: str) -> list[tuple[SpanRule, str]]:
    matches: list[tuple[SpanRule, str]] = []
    occupied: list[range] = []
    for rule in SPAN_RULES:
        match = re.search(rule.pattern, summary)
        if match:
            span = range(match.start(), match.end())
            if any(_overlaps(span, existing) for existing in occupied):
                continue
            occupied.append(span)
            matches.append((rule, match.group(0)))
    return matches


def _overlaps(left: range, right: range) -> bool:
    return left.start < right.stop and right.start < left.stop


def _replacement(rule: SpanRule, strategy: Strategy, rng: random.Random) -> str:
    if strategy == "mask":
        return rule.mask
    if strategy == "random":
        return rng.choice(rule.random_tokens)
    if strategy == "embedding":
        return rng.choice(rule.embedding_neighbors)
    return rng.choice(rule.ar_guided_candidates)


def _replace_once(text: str, old: str, new: str) -> str:
    return text.replace(old, new, 1)
