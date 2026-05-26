from ar_gstd.evaluate_corruptions import _compute_metrics


def test_compute_metrics_tracks_structure_and_mask() -> None:
    rows = [
        {
            "strategy": "mask",
            "clean_summary": "## Key Decisions\nKevin will test by Friday.\n## Risks and Open Issues\nOpen.\n## To-do\nKevin will test by Friday.",
            "corrupted_summary": "## Key Decisions\n[MASK] will test by Friday.\n## Risks and Open Issues\nOpen.\n## To-do\n[MASK] will test [MASK].",
        }
    ]

    metrics = _compute_metrics(rows)

    assert metrics["mask"]["rows"] == 1
    assert metrics["mask"]["changed"] == 1
    assert metrics["mask"]["heading_valid"] == 1
    assert metrics["mask"]["contains_mask"] == 1
    assert metrics["mask"]["owner_changed"] == 1
