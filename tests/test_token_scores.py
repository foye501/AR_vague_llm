from pathlib import Path

from ar_gstd.token_scores import (
    TokenScoreCache,
    TokenScoreRow,
    adaptive_mask_probability,
    difficulty_weights,
    load_token_score_caches,
    write_token_score_caches,
)


def test_difficulty_weights_prioritize_high_surprisal_tokens() -> None:
    rows = (
        TokenScoreRow(0, 10, " SELECT", "sql_keyword", 0.8, 0.1, 1, 0.8, 0.1, 0.7, 0.2),
        TokenScoreRow(1, 11, " singer", "schema_identifier", 0.01, 4.0, 20, 0.3, 0.2, 0.1, 1.0),
    )

    weights = difficulty_weights(rows, strength=0.5)

    assert weights[1] > weights[0]
    assert adaptive_mask_probability(0.5, weights[1]) > adaptive_mask_probability(0.5, weights[0])


def test_token_score_cache_round_trip(tmp_path: Path) -> None:
    cache = TokenScoreCache(
        example_id="x",
        transcript="Question: q",
        clean_summary="SELECT age",
        tokenizer_name="tok",
        teacher_model="teacher",
        target_token_ids=(1,),
        rows=(TokenScoreRow(0, 1, "SELECT", "sql_keyword", 0.5, 0.693, 1, 0.5, 0.2, 0.3, 0.7),),
    )
    path = tmp_path / "scores.jsonl"

    write_token_score_caches(path, [cache])
    loaded = load_token_score_caches(path)

    assert loaded["x"].rows[0].token_text == "SELECT"
    assert loaded["x"].rows[0].category == "sql_keyword"
