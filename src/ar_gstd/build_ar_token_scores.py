from __future__ import annotations

import argparse
import math
from pathlib import Path

from .analyze_transition_cache import schema_identifiers, token_category
from .build_transition_cache import build_teacher_prompt, _resolve_device, _torch_dtype
from .corruption import load_examples
from .token_scores import TokenScoreCache, TokenScoreRow, write_token_score_caches


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AR teacher token-difficulty scores for adaptive diffusion.")
    parser.add_argument("--input", type=Path, default=Path("artifacts/sql_create_context_subset.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/ar_token_scores.jsonl"))
    parser.add_argument("--teacher-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--tokenizer-name", default="", help="Defaults to --teacher-model.")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-target-tokens", type=int, default=256)
    parser.add_argument("--max-context-tokens", type=int, default=3072)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--device-map-auto", action="store_true", help="Use Transformers device_map='auto' for larger teachers.")
    args = parser.parse_args()

    _require_transformers()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_name = args.tokenizer_name or args.teacher_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = _torch_dtype(args.dtype, torch)
    if args.device_map_auto:
        model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        device = next(model.parameters()).device
    else:
        device = _resolve_device(args.device, torch)
        model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)
    model.eval()

    examples = load_examples(args.input.read_text(encoding="utf-8").splitlines())
    if args.max_examples:
        examples = examples[: args.max_examples]

    caches: list[TokenScoreCache] = []
    for index, example in enumerate(examples, start=1):
        target_token_ids = tokenizer.encode(example.clean_summary, add_special_tokens=False)[: args.max_target_tokens]
        schema_terms = schema_identifiers(example.transcript)
        rows: list[TokenScoreRow] = []
        for position, gold_token_id in enumerate(target_token_ids):
            prefix = tokenizer.decode(target_token_ids[:position], skip_special_tokens=False)
            prompt = build_teacher_prompt(example.transcript, prefix)
            stats = score_gold_token(
                prompt=prompt,
                tokenizer=tokenizer,
                model=model,
                device=device,
                gold_token_id=int(gold_token_id),
                top_k=args.top_k,
                temperature=args.temperature,
                max_context_tokens=args.max_context_tokens,
            )
            token_text = tokenizer.decode([gold_token_id], skip_special_tokens=False)
            rows.append(
                TokenScoreRow(
                    position=position,
                    token_id=int(gold_token_id),
                    token_text=token_text,
                    category=token_category(token_text, schema_terms),
                    **stats,
                )
            )
        caches.append(
            TokenScoreCache(
                example_id=example.example_id,
                transcript=example.transcript,
                clean_summary=example.clean_summary,
                tokenizer_name=tokenizer_name,
                teacher_model=args.teacher_model,
                target_token_ids=tuple(target_token_ids),
                rows=tuple(rows),
            )
        )
        print(f"[{index}/{len(examples)}] scored {example.example_id}: {len(rows)} tokens")

    write_token_score_caches(args.output, caches)
    print(f"Wrote {args.output}")


def score_gold_token(
    *,
    prompt: str,
    tokenizer,
    model,
    device,
    gold_token_id: int,
    top_k: int,
    temperature: float,
    max_context_tokens: int,
) -> dict[str, float | int]:
    import torch

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_context_tokens,
    )
    if not hasattr(model, "hf_device_map"):
        encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits[0, -1].float()

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    gold_prob = float(probs[gold_token_id].item())
    rank = int((scaled_logits > scaled_logits[gold_token_id]).sum().item()) + 1
    values, _ = torch.topk(probs, k=min(top_k, probs.shape[-1]))
    top_values = [float(value) for value in values.tolist()]
    top1_prob = top_values[0] if top_values else 0.0
    top2_prob = top_values[1] if len(top_values) > 1 else 0.0
    return {
        "gold_prob": gold_prob,
        "surprisal": -math.log(max(gold_prob, 1e-12)),
        "rank": rank,
        "top1_prob": top1_prob,
        "top2_prob": top2_prob,
        "margin": top1_prob - top2_prob,
        "topk_entropy": -sum(prob * math.log(max(prob, 1e-12)) for prob in top_values),
    }


def _require_transformers() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit('Install training dependencies first: python -m pip install -e ".[train,dev]"') from exc


if __name__ == "__main__":
    main()
