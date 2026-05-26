from ar_gstd.evaluate_denoiser import score_predictions


def test_score_predictions_reports_repair_delta() -> None:
    rows = [
        {
            "corrupted_summary": "## Key Decisions\n- The team discussed timing.\n\n## Risks and Open Issues\n- Risk remains unresolved.\n\n## To-do\n- Someone will follow up later.",
            "prediction": "## Key Decisions\n- The team decided to prioritize ASR correction.\n\n## Risks and Open Issues\n- ASR quality remains unresolved.\n\n## To-do\n- Kevin will test ASR-corrected transcripts by Friday.",
            "clean_summary": "## Key Decisions\n- The team decided to prioritize ASR correction.\n\n## Risks and Open Issues\n- ASR quality remains unresolved.\n\n## To-do\n- Kevin will test ASR-corrected transcripts by Friday.",
        }
    ]

    metrics = score_predictions(rows)

    assert metrics["rows"] == 1
    assert metrics["prediction_exact_match"] == 1.0
    assert metrics["token_f1_repair_delta"] > 0
    assert metrics["prediction_heading_valid"] == 1.0


def test_score_predictions_reports_sql_metrics() -> None:
    rows = [
        {
            "corrupted_summary": "SELECT name FROM singer WHERE age > 30",
            "prediction": "SELECT name FROM singer WHERE age = 30;",
            "clean_summary": "select name from singer where age=30",
        }
    ]

    metrics = score_predictions(rows)

    assert metrics["sql_exact_match"] == 1.0
    assert metrics["sql_keyword_valid"] == 1.0
    assert metrics["sql_repair_delta"] == 1.0
