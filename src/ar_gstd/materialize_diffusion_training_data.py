from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

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
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _require_transformers()
    from transformers import AutoTokenizer

    caches = load_transition_caches(args.cache)
    if not caches:
        raise SystemExit(f"No caches found in {args.cache}")
    tokenizer = AutoTokenizer.from_pretrained(caches[0].tokenizer_name, trust_remote_code=True)
    timesteps = parse_timesteps(args.timesteps, args.num_steps)

    rows: list[dict[str, object]] = []
    for example_index, cache in enumerate(caches):
        for timestep in timesteps:
            for variant_index in range(args.variants_per_timestep):
                sample_seed = args.seed + example_index * 100_003 + timestep * 101 + variant_index
                corrupted = corrupt_diffusion_state(
                    cache,
                    tokenizer=tokenizer,
                    timestep=timestep,
                    num_steps=args.num_steps,
                    noise_kind=args.noise_kind,
                    seed=sample_seed,
                    mask_token=args.mask_token,
                    ar_strength=args.ar_strength,
                    mask_power=args.mask_power,
                )
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
) -> str:
    if not 0 <= timestep <= num_steps:
        raise ValueError("timestep must be between 0 and num_steps")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    rng = random.Random(seed)
    mask_prob, ar_prob = diffusion_noise_probs(
        timestep=timestep,
        num_steps=num_steps,
        noise_kind=noise_kind,
        ar_strength=ar_strength,
        mask_power=mask_power,
    )
    rows_by_position = {row.position: row for row in cache.rows}
    pieces: list[str] = []

    for position, source_token_id in enumerate(cache.target_token_ids):
        draw = rng.random()
        if draw < mask_prob:
            pieces.append(mask_token)
            continue
        if draw < mask_prob + ar_prob:
            row = rows_by_position.get(position)
            if row is not None and row.top_token_ids:
                pieces.append(tokenizer.decode([sample_from_sparse_row(row, rng)], skip_special_tokens=True))
                continue
        pieces.append(tokenizer.decode([source_token_id], skip_special_tokens=True))

    return compact_mask_text("".join(pieces), mask_token=mask_token)


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
