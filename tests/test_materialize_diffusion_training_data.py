from ar_gstd.materialize_diffusion_training_data import diffusion_noise_probs, parse_timesteps


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
