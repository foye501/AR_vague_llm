from ar_gstd.bidirectional_denoising import aligned_target_token_ids, build_bidirectional_denoising_features


class ToyTokenizer:
    def __init__(self):
        self.vocab = {
            "Bidirectionally": 1,
            "denoise": 2,
            "the": 3,
            "target": 4,
            "output": 5,
            "using": 6,
            "source": 7,
            "context.": 8,
            "Source": 9,
            "context:": 10,
            "Question:": 11,
            "q": 12,
            "Noisy": 13,
            "output:": 14,
            "[MASK]": 99,
            "SELECT": 20,
            "age": 21,
        }

    def encode(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [self.vocab.get(piece, 42) for piece in text.replace("\n", " ").split()]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return ids


def test_aligned_target_token_ids_prefers_materialized_ids() -> None:
    row = {
        "clean_summary": "wrong fallback",
        "corrupted_summary": "wrong fallback",
        "clean_token_ids": [20, 21],
        "corrupted_token_ids": [99, 21],
    }

    clean_ids, corrupted_ids = aligned_target_token_ids(row, tokenizer=ToyTokenizer(), max_target_length=8)

    assert clean_ids == [20, 21]
    assert corrupted_ids == [99, 21]


def test_bidirectional_features_mask_loss_to_target_positions() -> None:
    row = {
        "id": "x",
        "transcript": "Question: q",
        "clean_summary": "SELECT age",
        "corrupted_summary": "[MASK] age",
        "clean_token_ids": [20, 21],
        "corrupted_token_ids": [99, 21],
        "timestep": 10,
        "num_steps": 10,
        "noise_kind": "absorbing",
    }

    features = build_bidirectional_denoising_features(
        row,
        tokenizer=ToyTokenizer(),
        max_sequence_length=64,
        max_target_length=8,
    )

    assert features.input_ids[-2:] == [99, 21]
    assert features.labels[: features.target_start] == [-100] * features.target_start
    assert features.labels[features.target_start :] == [20, 21]
    assert features.target_length == 2
