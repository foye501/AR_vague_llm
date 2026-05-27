from ar_gstd.filter_eval_by_category import has_schema_identifier_target


def test_has_schema_identifier_target_detects_target_schema_use() -> None:
    row = {
        "transcript": "Question:\nQ\n\nDatabase schema:\nCREATE TABLE singer (name TEXT, age INTEGER)",
        "clean_summary": "SELECT name FROM singer",
    }

    assert has_schema_identifier_target(row)


def test_has_schema_identifier_target_rejects_no_schema_terms() -> None:
    row = {
        "transcript": "Question:\nQ\n\nDatabase schema:\nCREATE TABLE singer (name TEXT, age INTEGER)",
        "clean_summary": "SELECT COUNT(*)",
    }

    assert not has_schema_identifier_target(row)
