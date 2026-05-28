from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bidirectional_denoising import build_bidirectional_denoising_features
from .evaluate_denoiser import score_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a bidirectional Qwen-family masked denoiser.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--output-predictions", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--only-timestep", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    args = parser.parse_args()

    _require_train_deps()
    import torch
    from transformers import AutoConfig, AutoTokenizer

    from .bidirectional_qwen import build_bidirectional_qwen_for_masked_lm_class

    rows = [json.loads(line) for line in args.eval_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.only_timestep is not None:
        rows = [row for row in rows if int(row.get("timestep", -1)) == args.only_timestep]
    if not rows:
        raise SystemExit(f"No rows found in {args.eval_file}")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)
    config = AutoConfig.from_pretrained(str(args.model_dir), trust_remote_code=True)
    model_cls = build_bidirectional_qwen_for_masked_lm_class()
    model = model_cls.from_pretrained(str(args.model_dir), config=config, trust_remote_code=True)
    model._disable_causal_attention_flags()
    device = _resolve_device(args.device, torch)
    model.to(device)
    model.eval()

    predictions: list[dict[str, object]] = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        features = [
            build_bidirectional_denoising_features(
                row,
                tokenizer=tokenizer,
                max_sequence_length=args.max_sequence_length,
                max_target_length=args.max_target_length,
            )
            for row in batch_rows
        ]
        batch = pad_eval_features(features, pad_token_id=tokenizer.pad_token_id)
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        for row, feature, row_logits in zip(batch_rows, features, logits, strict=True):
            target_logits = row_logits[feature.target_start : feature.target_start + feature.target_length]
            prediction_ids = target_logits.argmax(dim=-1).tolist()
            prediction = tokenizer.decode(prediction_ids, skip_special_tokens=True)
            predictions.append(
                {
                    "id": row["id"],
                    "strategy": row.get("strategy", ""),
                    "prompt_mode": row.get("prompt_mode", "repair"),
                    "noise_kind": row.get("noise_kind", ""),
                    "timestep": row.get("timestep", ""),
                    "num_steps": row.get("num_steps", ""),
                    "transcript": row["transcript"],
                    "corrupted_summary": row["corrupted_summary"],
                    "prediction": prediction,
                    "clean_summary": row["clean_summary"],
                }
            )

    metrics = score_predictions(predictions)
    args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.output_predictions.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in predictions) + "\n",
        encoding="utf-8",
    )
    args.output_metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def pad_eval_features(features, *, pad_token_id: int):
    import torch

    max_length = max(len(feature.input_ids) for feature in features)
    input_ids = []
    attention_mask = []
    for feature in features:
        padding = max_length - len(feature.input_ids)
        input_ids.append(feature.input_ids + [pad_token_id] * padding)
        attention_mask.append(feature.attention_mask + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def _resolve_device(device: str, torch):
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _require_train_deps() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit('Install training dependencies first: python -m pip install -e ".[train,dev]"') from exc


if __name__ == "__main__":
    main()
