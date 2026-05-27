from __future__ import annotations

import argparse
from pathlib import Path

from .corruption import load_examples
from .transitions import SparseTransitionRow, TransitionCache, normalize_probs, write_transition_caches


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a conditional sparse AR top-k transition cache.")
    parser.add_argument("--input", type=Path, default=Path("data/meeting_summaries_seed.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/ar_transition_cache.jsonl"))
    parser.add_argument("--teacher-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--tokenizer-name", default="", help="Defaults to --teacher-model.")
    parser.add_argument("--student-tokenizer-name", default="", help="If set, cache transitions in the student tokenizer space.")
    parser.add_argument("--top-k", type=int, default=8)
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

    teacher_tokenizer_name = args.tokenizer_name or args.teacher_model
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_tokenizer_name, trust_remote_code=True)
    if teacher_tokenizer.pad_token_id is None and teacher_tokenizer.eos_token_id is not None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    student_tokenizer_name = args.student_tokenizer_name or teacher_tokenizer_name
    student_tokenizer = (
        teacher_tokenizer
        if student_tokenizer_name == teacher_tokenizer_name
        else AutoTokenizer.from_pretrained(student_tokenizer_name, trust_remote_code=True)
    )

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

    caches: list[TransitionCache] = []
    for index, example in enumerate(examples, start=1):
        target_token_ids = student_tokenizer.encode(example.clean_summary, add_special_tokens=False)[: args.max_target_tokens]
        rows: list[SparseTransitionRow] = []
        for position, source_token_id in enumerate(target_token_ids):
            prefix = student_tokenizer.decode(target_token_ids[:position], skip_special_tokens=False)
            prompt = build_teacher_prompt(example.transcript, prefix)
            teacher_top_ids, teacher_top_probs = _next_token_topk(
                prompt=prompt,
                tokenizer=teacher_tokenizer,
                model=model,
                device=device,
                source_token_id=None,
                top_k=args.top_k * 8,
                temperature=args.temperature,
                max_context_tokens=args.max_context_tokens,
            )
            top_ids, top_probs, top_texts = project_teacher_topk_to_student(
                teacher_tokenizer=teacher_tokenizer,
                student_tokenizer=student_tokenizer,
                teacher_top_ids=teacher_top_ids,
                teacher_top_probs=teacher_top_probs,
                source_token_id=source_token_id,
                top_k=args.top_k,
            )
            rows.append(
                SparseTransitionRow(
                    position=position,
                    source_token_id=int(source_token_id),
                    source_text=student_tokenizer.decode([source_token_id], skip_special_tokens=False),
                    top_token_ids=tuple(top_ids),
                    top_probs=tuple(top_probs),
                    top_texts=tuple(top_texts),
                )
            )
        caches.append(
            TransitionCache(
                example_id=example.example_id,
                transcript=example.transcript,
                clean_summary=example.clean_summary,
                tokenizer_name=student_tokenizer_name,
                teacher_model=args.teacher_model,
                target_token_ids=tuple(target_token_ids),
                rows=tuple(rows),
            )
        )
        print(f"[{index}/{len(examples)}] cached {example.example_id}: {len(rows)} transition rows")

    write_transition_caches(args.output, caches)
    print(f"Wrote {args.output}")


def build_teacher_prompt(transcript: str, clean_prefix: str) -> str:
    return (
        "You are generating a target output from the given source context.\n\n"
        f"Source context:\n{transcript}\n\n"
        "Target output prefix:\n"
        f"{clean_prefix}"
    )


def _next_token_topk(
    *,
    prompt: str,
    tokenizer,
    model,
    device,
    source_token_id: int | None,
    top_k: int,
    temperature: float,
    max_context_tokens: int,
) -> tuple[list[int], list[float]]:
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
    probs = torch.softmax(logits / temperature, dim=-1)
    values, ids = torch.topk(probs, k=min(top_k + 8, probs.shape[-1]))

    candidate_ids: list[int] = []
    candidate_probs: list[float] = []
    for token_id, prob in zip(ids.tolist(), values.tolist(), strict=True):
        if source_token_id is not None and int(token_id) == int(source_token_id):
            continue
        candidate_ids.append(int(token_id))
        candidate_probs.append(float(prob))
        if len(candidate_ids) >= top_k:
            break

    return candidate_ids, list(normalize_probs(candidate_probs))


def project_teacher_topk_to_student(
    *,
    teacher_tokenizer,
    student_tokenizer,
    teacher_top_ids: list[int],
    teacher_top_probs: list[float],
    source_token_id: int,
    top_k: int,
) -> tuple[list[int], list[float], list[str]]:
    scores: dict[int, float] = {}
    texts: dict[int, str] = {}
    for teacher_token_id, prob in zip(teacher_top_ids, teacher_top_probs, strict=True):
        text = teacher_tokenizer.decode([teacher_token_id], skip_special_tokens=False)
        student_ids = student_tokenizer.encode(text, add_special_tokens=False)
        if len(student_ids) != 1:
            continue
        student_id = int(student_ids[0])
        if student_id == int(source_token_id):
            continue
        scores[student_id] = scores.get(student_id, 0.0) + float(prob)
        texts[student_id] = student_tokenizer.decode([student_id], skip_special_tokens=False)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    token_ids = [token_id for token_id, _ in ranked]
    probs = list(normalize_probs(score for _, score in ranked))
    return token_ids, probs, [texts[token_id] for token_id in token_ids]


def _resolve_device(device: str, torch):
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _torch_dtype(dtype: str, torch):
    if dtype == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]


def _require_transformers() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit('Install training dependencies first: python -m pip install -e ".[train,dev]"') from exc


if __name__ == "__main__":
    main()
