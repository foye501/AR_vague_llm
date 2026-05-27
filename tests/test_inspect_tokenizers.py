from ar_gstd.inspect_tokenizers import _projection_stats, _tokenize_sample


class ToyTokenizer:
    all_special_ids = {0}

    def __init__(self, decode_map, encode_map):
        self.decode_map = decode_map
        self.encode_map = encode_map

    def __len__(self):
        return len(self.decode_map)

    def decode(self, token_ids, skip_special_tokens=False):
        return self.decode_map[token_ids[0]]

    def encode(self, text, add_special_tokens=False):
        return self.encode_map.get(text, [])

    def convert_ids_to_tokens(self, token_ids):
        return [f"tok_{token_id}" for token_id in token_ids]


def test_projection_stats_counts_single_multi_and_zero_token_mappings() -> None:
    source = ToyTokenizer(
        {
            0: "<special>",
            1: " age",
            2: " release year",
            3: "∅",
            4: "",
        },
        {},
    )
    target = ToyTokenizer(
        {},
        {
            " age": [10],
            " release year": [11, 12],
        },
    )

    stats = _projection_stats(source=source, target=target, max_vocab_scan=0)

    assert stats["source_tokens_scanned"] == 3
    assert stats["single_token"] == 1
    assert stats["multi_token"] == 1
    assert stats["zero_token"] == 1
    assert stats["single_token_ratio"] == 1 / 3


def test_tokenize_sample_returns_decoded_pieces() -> None:
    tokenizer = ToyTokenizer({1: "SELECT", 2: " age"}, {"SELECT age": [1, 2]})

    result = _tokenize_sample(tokenizer, "SELECT age")

    assert result["token_count"] == 2
    assert result["tokens"] == ["tok_1", "tok_2"]
    assert result["pieces"] == ["SELECT", " age"]
