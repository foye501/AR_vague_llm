from ar_gstd.prepare_sql_create_context import prepare_row


def test_prepare_row_builds_compatible_text_to_sql_fields() -> None:
    row = prepare_row(
        {
            "question": "Which singers are older than 30?",
            "context": "CREATE TABLE singer (name TEXT, age INTEGER)",
            "answer": "SELECT name FROM singer WHERE age > 30",
        },
        row_id="sql-1",
        max_source_chars=1000,
        max_target_chars=1000,
    )

    assert row is not None
    assert row["task"] == "text_to_sql"
    assert "Question:" in row["transcript"]
    assert "CREATE TABLE singer" in row["transcript"]
    assert row["clean_summary"].startswith("SELECT")
