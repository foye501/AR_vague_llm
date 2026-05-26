from ar_gstd.summarize_metrics import to_markdown


def test_to_markdown_includes_run_and_metric_values() -> None:
    markdown = to_markdown(
        [
            (
                "ar_on_ar",
                {
                    "rows": 4,
                    "sql_exact_match": 0.8,
                    "sql_repair_delta": 0.3,
                    "prediction_token_f1": 0.9,
                    "corrupted_token_f1": 0.5,
                    "token_f1_repair_delta": 0.4,
                },
            )
        ]
    )

    assert "ar_on_ar" in markdown
    assert "0.8000" in markdown
    assert "0.9000" in markdown
    assert "0.3000" in markdown
