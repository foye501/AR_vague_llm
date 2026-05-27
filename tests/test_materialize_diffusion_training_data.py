from ar_gstd.materialize_diffusion_training_data import (
    add_diffusion_special_tokens,
    corrupt_diffusion_token_ids,
    diffusion_noise_probs,
    parse_timesteps,
)
from ar_gstd.transitions import SparseTransitionRow, TransitionCache


class ToyTokenizer:
    pad_token_id = None
    mask_token_id = None

    def __init__(self):
        self.next_id = 100

    def add_special_tokens(self, values):
        if "pad_token" in values:
            self.pad_token_id = self.next_id
            self.next_id += 1
        if "mask_token" in values:
            self.mask_token_id = self.next_id
            self.next_id += 1


def test_absorbing_endpoint_is_all_mask_probability() -> None:
    mask_prob, ar_prob = diffusion_noise_probs(
        timestep=10,
        num_steps=10,
        noise_kind="ar_absorb",
        ar_strength=0.65,
        mask_power=1.0,
    )

    assert mask_prob == 1.0
    assert ar_prob == 0.0


def test_ar_probability_only_exists_at_intermediate_timesteps() -> None:
    mask_prob, ar_prob = diffusion_noise_probs(
        timestep=5,
        num_steps=10,
        noise_kind="ar_absorb",
        ar_strength=0.65,
        mask_power=1.0,
    )

    assert 0 < mask_prob < 1
    assert ar_prob > 0


def test_parse_timesteps_defaults_to_full_schedule() -> None:
    assert parse_timesteps("", 3) == [1, 2, 3]


def test_corrupt_diffusion_token_ids_preserves_target_length_at_endpoint() -> None:
    tokenizer = ToyTokenizer()
    add_diffusion_special_tokens(tokenizer, mask_token="[MASK]", pad_token="[PAD]")
    cache = TransitionCache(
        example_id="x",
        transcript="Question: q",
        clean_summary="SELECT age",
        tokenizer_name="toy",
        teacher_model="teacher",
        target_token_ids=(10, 11, 12),
        rows=(
            SparseTransitionRow(
                position=0,
                source_token_id=10,
                source_text="SELECT",
                top_token_ids=(20,),
                top_probs=(1.0,),
                top_texts=("FROM",),
            ),
        ),
    )

    corrupted = corrupt_diffusion_token_ids(
        cache,
        tokenizer=tokenizer,
        timestep=10,
        num_steps=10,
        noise_kind="ar_absorb",
        seed=7,
        mask_token="[MASK]",
        ar_strength=0.65,
        mask_power=1.0,
    )

    assert corrupted == [tokenizer.mask_token_id, tokenizer.mask_token_id, tokenizer.mask_token_id]
    assert len(corrupted) == len(cache.target_token_ids)
