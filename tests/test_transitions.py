from ar_gstd.make_fixed_transition_cache import build_fixed_rows, rewrite_cache_with_fixed_rows
from ar_gstd.transitions import (
    SparseTransitionRow,
    TransitionCache,
    normalize_probs,
    sample_corrupted_token_ids,
)


def _cache() -> TransitionCache:
    return TransitionCache(
        example_id="demo",
        transcript="Transcript",
        clean_summary="Clean",
        tokenizer_name="toy",
        teacher_model="teacher",
        target_token_ids=(10, 20, 10),
        rows=(
            SparseTransitionRow(0, 10, "A", (11, 12), (0.7, 0.3), ("B", "C")),
            SparseTransitionRow(1, 20, "D", (21,), (1.0,), ("E",)),
            SparseTransitionRow(2, 10, "A", (13,), (1.0,), ("F",)),
        ),
    )


def test_normalize_probs_handles_zero_mass() -> None:
    assert normalize_probs([0, 0]) == (0.5, 0.5)


def test_sample_corrupted_token_ids_changes_when_beta_one() -> None:
    corrupted = sample_corrupted_token_ids(_cache(), beta=1.0, seed=4)

    assert corrupted != [10, 20, 10]
    assert corrupted[1] == 21


def test_fixed_rows_remove_context_dependence_for_same_source_token() -> None:
    fixed_rows = build_fixed_rows([_cache()], top_k=2)
    rewritten = rewrite_cache_with_fixed_rows(_cache(), fixed_rows)
    first = rewritten.rows[0]
    third = rewritten.rows[2]

    assert first.source_token_id == third.source_token_id
    assert first.top_token_ids == third.top_token_ids
    assert first.top_probs == third.top_probs
