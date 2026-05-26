import json

from ar_gstd.materialize_clean_training_data import main


def test_materialize_clean_training_data_cli(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "clean.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "id": "sql-1",
                "transcript": "Question:\nQ\n\nDatabase schema:\nCREATE TABLE t (x INT)",
                "clean_summary": "SELECT x FROM t",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["prog", "--input", str(input_path), "--output", str(output_path)])

    main()

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["prompt_mode"] == "generate"
    assert row["corrupted_summary"] == ""
    assert row["clean_summary"] == "SELECT x FROM t"
