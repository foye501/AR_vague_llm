from ar_gstd.evaluate_bidirectional_qwen_denoiser import top_confidence_positions


def test_top_confidence_positions_selects_highest_values() -> None:
    assert top_confidence_positions([0.1, 0.9, 0.3, 0.7], 2) == [1, 3]


def test_top_confidence_positions_clamps_count_to_length() -> None:
    assert top_confidence_positions([0.2, 0.1], 5) == [0, 1]
