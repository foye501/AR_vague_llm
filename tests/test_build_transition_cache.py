from ar_gstd.build_transition_cache import project_teacher_topk_to_student


class ToyTokenizer:
    def __init__(self, decode_map, encode_map):
        self.decode_map = decode_map
        self.encode_map = encode_map

    def decode(self, token_ids, skip_special_tokens=False):
        return self.decode_map[token_ids[0]]

    def encode(self, text, add_special_tokens=False):
        return self.encode_map[text]


def test_project_teacher_topk_to_student_keeps_single_student_tokens() -> None:
    teacher = ToyTokenizer({1: " age", 2: " release year", 3: " name"}, {})
    student = ToyTokenizer(
        {10: " age", 11: " name"},
        {
            " age": [10],
            " release year": [12, 13],
            " name": [11],
        },
    )

    token_ids, probs, texts = project_teacher_topk_to_student(
        teacher_tokenizer=teacher,
        student_tokenizer=student,
        teacher_top_ids=[1, 2, 3],
        teacher_top_probs=[0.5, 0.3, 0.2],
        source_token_id=99,
        top_k=4,
    )

    assert token_ids == [10, 11]
    assert probs == [0.7142857142857143, 0.28571428571428575]
    assert texts == [" age", " name"]
