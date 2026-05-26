from ar_gstd.corruption import Example, corrupt_example, corrupt_summary


SUMMARY = """## Key Decisions
- The team decided to prioritize ASR correction before model fine-tuning.

## Risks and Open Issues
- ASR quality remains unresolved and may limit final summary quality.

## To-do
- Kevin will test ASR-corrected transcripts by Friday."""


def test_mask_corruption_preserves_structure() -> None:
    corrupted = corrupt_summary(SUMMARY, "mask", beta=1.0, seed=3)

    assert "## Key Decisions" in corrupted
    assert "## Risks and Open Issues" in corrupted
    assert "## To-do" in corrupted
    assert "[MASK]" in corrupted


def test_ar_guided_corruption_is_semantic_not_masked() -> None:
    corrupted = corrupt_summary(SUMMARY, "ar_guided", beta=1.0, seed=3)

    assert "[MASK]" not in corrupted
    assert corrupted != SUMMARY
    assert any(term in corrupted for term in ("discussed", "considered", "proposed"))


def test_corruption_is_deterministic_for_seed() -> None:
    first = corrupt_summary(SUMMARY, "embedding", beta=0.55, seed=11)
    second = corrupt_summary(SUMMARY, "embedding", beta=0.55, seed=11)

    assert first == second


def test_corrupt_example_payload_contains_training_fields() -> None:
    example = Example("demo", "Transcript text", SUMMARY)
    row = corrupt_example(example, "random", beta=0.7, seed=5)

    assert row["id"] == "demo"
    assert row["strategy"] == "random"
    assert row["transcript"] == "Transcript text"
    assert row["clean_summary"] == SUMMARY
    assert row["corrupted_summary"] != SUMMARY
