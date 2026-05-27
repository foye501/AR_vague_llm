from ar_gstd.analyze_transition_cache import (
    analyze_caches,
    diff_analysis,
    schema_identifiers,
    token_category,
)
from ar_gstd.transitions import SparseTransitionRow, TransitionCache


def test_schema_identifiers_extracts_create_table_names() -> None:
    context = "Question:\nQ\n\nDatabase schema:\nCREATE TABLE singer (name TEXT, age INTEGER)"

    assert schema_identifiers(context) == {"singer", "name", "age"}


def test_token_category_identifies_schema_and_keywords() -> None:
    schema = {"singer", "age"}

    assert token_category(" SELECT", schema) == "sql_keyword"
    assert token_category(" age", schema) == "schema_identifier"
    assert token_category(">", schema) == "operator"


def test_analyze_caches_reports_gold_topk_delta() -> None:
    conditional = [
        TransitionCache(
            example_id="x",
            transcript="CREATE TABLE singer (name TEXT)",
            clean_summary="SELECT name FROM singer",
            tokenizer_name="toy",
            teacher_model="teacher",
            target_token_ids=(1,),
            rows=(SparseTransitionRow(0, 1, " name", (1, 2), (0.8, 0.2), (" name", " age")),),
        )
    ]
    fixed = [
        TransitionCache(
            example_id="x",
            transcript="CREATE TABLE singer (name TEXT)",
            clean_summary="SELECT name FROM singer",
            tokenizer_name="toy",
            teacher_model="fixed",
            target_token_ids=(1,),
            rows=(SparseTransitionRow(0, 1, " name", (2,), (1.0,), (" age",)),),
        )
    ]

    cond_metrics = analyze_caches(conditional)
    fixed_metrics = analyze_caches(fixed)
    delta = diff_analysis(cond_metrics, fixed_metrics)

    assert cond_metrics["schema_identifier"]["gold_topk_rate"] == 1.0
    assert fixed_metrics["schema_identifier"]["gold_topk_rate"] == 0.0
    assert delta["schema_identifier"]["gold_topk_rate_delta"] == 1.0
