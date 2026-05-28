from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from .token_scores import adaptive_mask_probability, difficulty_weights, load_token_score_caches
from .transitions import TransitionCache, load_transition_caches, sample_from_sparse_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize diffusion-style noisy training pairs.")
    parser.add_argument("--cache", type=Path, default=Path("artifacts/ar_transition_cache.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/train_pairs_diffusion_ar_absorb.jsonl"))
    parser.add_argument("--noise-kind", choices=("absorbing", "ar_absorb"), default="ar_absorb")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--timesteps", default="", help="Comma-separated timesteps. Defaults to 1..num-steps.")
    parser.add_argument("--variants-per-timestep", type=int, default=1)
    parser.add_argument("--ar-strength", type=float, default=0.65)
    parser.add_argument("--mask-power", type=float, default=1.0)
    parser.add_argument("--mask-token", default="[MASK]")
    parser.add_argument("--pad-token", default="[PAD]")
    parser.add_argument("--token-score-cache", type=Path, default=None)
    parser.add_argument("--mask-policy", choices=("uniform", "ar_difficulty"), default="uniform")
    parser.add_argument("--difficulty-strength", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _require_transformers()
    from transformers import AutoTokenizer

    caches = load_transition_caches(args.cache)
    if not caches:
        raise SystemExit(f"No caches found in {args.cache}")
    tokenizer = AutoTokenizer.from_pretrained(caches[0].tokenizer_name, trust_remote_code=True)
    add_diffusion_special_tokens(tokenizer, mask_token=args.mask_token, pad_token=args.pad_token)
    timesteps = parse_timesteps(args.timesteps, args.num_steps)
    token_score_caches = load_token_score_caches(args.token_score_cache) if args.token_score_cache is not None else {}

    rows: list[dict[str, object]] = []
    for example_index, cache in enumerate(caches):
        for timestep in timesteps:
            for variant_index in range(args.variants_per_timestep):
                sample_seed = args.seed + example_index * 100_003 + timestep * 101 + variant_index
                token_score_cache = token_score_caches.get(cache.example_id)
                corrupted_token_ids = corrupt_diffusion_token_ids(
                    cache,
                    tokenizer=tokenizer,
                    timestep=timestep,
                    num_steps=args.num_steps,
                    noise_kind=args.noise_kind,
                    seed=sample_seed,
                    mask_token=args.mask_token,
                    ar_strength=args.ar_strength,
                    mask_power=args.mask_power,
                    difficulty_by_position=(
                        difficulty_weights(token_score_cache.rows, strength=args.difficulty_strength)
                        if args.mask_policy == "ar_difficulty" and token_score_cache is not None
                        else None
                    ),
                )
                corrupted = decode_diffusion_token_ids(tokenizer, corrupted_token_ids, mask_token=args.mask_token)
                rows.append(
                    {
                        "id": f"{cache.example_id}#t{timestep}#v{variant_index}",
                        "strategy": args.noise_kind,
                        "noise_kind": args.noise_kind,
                        "timestep": timestep,
                        "num_steps": args.num_steps,
                        "transcript": cache.transcript,
                        "corrupted_summary": corrupted,
                        "clean_summary": cache.clean_summary,
                        "corrupted_token_ids": corrupted_token_ids,
                        "clean_token_ids": list(cache.target_token_ids),
                        "mask_policy": args.mask_policy,
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({len(rows)} rows)")


def corrupt_diffusion_state(
    cache: TransitionCache,
    *,
    tokenizer,
    timestep: int,
    num_steps: int,
    noise_kind: str,
    seed: int,
    mask_token: str,
    ar_strength: float,
    mask_power: float,
    difficulty_by_position: dict[int, float] | None = None,
) -> str:
    corrupted_token_ids = corrupt_diffusion_token_ids(
        cache,
        tokenizer=tokenizer,
        timestep=timestep,
        num_steps=num_steps,
        noise_kind=noise_kind,
        seed=seed,
        mask_token=mask_token,
        ar_strength=ar_strength,
        mask_power=mask_power,
        difficulty_by_position=difficulty_by_position,
    )
    return decode_diffusion_token_ids(tokenizer, corrupted_token_ids, mask_token=mask_token)


def corrupt_diffusion_token_ids(
    cache: TransitionCache,
    *,
    tokenizer,
    timestep: int,
    num_steps: int,
    noise_kind: str,
    seed: int,
    mask_token: str,
    ar_strength: float,
    mask_power: float,
    difficulty_by_position: dict[int, float] | None = None,
) -> list[int]:
    if not 0 <= timestep <= num_steps:
        raise ValueError("timestep must be between 0 and num_steps")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if tokenizer.mask_token_id is None:
        raise ValueError("tokenizer must define a mask token before corruption")

    rng = random.Random(seed)
    mask_prob, ar_prob = diffusion_noise_probs(
        timestep=timestep,
        num_steps=num_steps,
        noise_kind=noise_kind,
        ar_strength=ar_strength,
        mask_power=mask_power,
    )
    rows_by_position = {row.position: row for row in cache.rows}
    corrupted_token_ids: list[int] = []

    for position, source_token_id in enumerate(cache.target_token_ids):
        position_mask_prob = mask_prob
        if difficulty_by_position is not None and mask_prob < 1.0:
            position_mask_prob = adaptive_mask_probability(mask_prob, difficulty_by_position.get(position, 1.0))
        position_ar_prob = min(ar_prob, 1.0 - position_mask_prob)
        draw = rng.random()
        if draw < position_mask_prob:
            corrupted_token_ids.append(int(tokenizer.mask_token_id))
            continue
        if draw < position_mask_prob + position_ar_prob:
            row = rows_by_position.get(position)
            if row is not None and row.top_token_ids:
                corrupted_token_ids.append(int(sample_from_sparse_row(row, rng)))
                continue
        corrupted_token_ids.append(int(source_token_id))

    return corrupted_token_ids


def decode_diffusion_token_ids(tokenizer, token_ids: list[int], *, mask_token: str) -> str:
    return compact_mask_text(tokenizer.decode(token_ids, skip_special_tokens=False), mask_token=mask_token)


def add_diffusion_special_tokens(tokenizer, *, mask_token: str, pad_token: str) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": pad_token})
    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": mask_token})


def diffusion_noise_probs(
    *,
    timestep: int,
    num_steps: int,
    noise_kind: str,
    ar_strength: float,
    mask_power: float,
) -> tuple[float, float]:
    progress = timestep / num_steps
    mask_prob = min(max(progress**mask_power, 0.0), 1.0)
    if noise_kind == "absorbing":
        return mask_prob, 0.0
    if noise_kind == "ar_absorb":
        ar_prob = min(max(ar_strength * progress * (1.0 - mask_prob), 0.0), 1.0 - mask_prob)
        return mask_prob, ar_prob
    raise ValueError(f"unknown noise kind: {noise_kind}")


def parse_timesteps(value: str, num_steps: int) -> list[int]:
    if value.strip():
        timesteps = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        timesteps = list(range(1, num_steps + 1))
    for timestep in timesteps:
        if timestep < 0 or timestep > num_steps:
            raise ValueError(f"timestep {timestep} is outside [0, {num_steps}]")
    return timesteps


def compact_mask_text(text: str, *, mask_token: str) -> str:
    text = " ".join(text.split())
    text = text.replace(f"{mask_token} {mask_token}", f"{mask_token} {mask_token}")
    return text


def _require_transformers() -> None:
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit('Install training dependencies first: python -m pip install -e ".[train,dev]"') from exc


if __name__ == "__main__":
    main()
