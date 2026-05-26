from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import random


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a seq2seq denoiser on sampled corruption pairs.")
    parser.add_argument("--train-file", type=Path, default=Path("artifacts/train_pairs_ar.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/denoiser_ar"))
    parser.add_argument("--model-name", default="google/flan-t5-small")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--eval-file", type=Path, default=None)
    parser.add_argument("--train-split-output", type=Path, default=None)
    parser.add_argument("--eval-split-output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-source-length", type=int, default=768)
    parser.add_argument("--max-target-length", type=int, default=256)
    args = parser.parse_args()

    _require_train_deps()
    from datasets import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    rows = [json.loads(line) for line in args.train_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 2:
        raise SystemExit("Need at least two training rows.")
    if args.eval_file:
        train_rows = rows
        eval_rows = [json.loads(line) for line in args.eval_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        train_rows, eval_rows = split_rows_by_example_id(rows, eval_ratio=args.eval_ratio, seed=args.seed)
    if not eval_rows:
        raise SystemExit("Evaluation split is empty.")
    if args.train_split_output:
        write_jsonl(args.train_split_output, train_rows)
    if args.eval_split_output:
        write_jsonl(args.eval_split_output, eval_rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    def preprocess(batch):
        inputs = [build_prompt_from_batch(batch, index) for index in range(len(batch["transcript"]))]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, truncation=True)
        labels = tokenizer(text_target=batch["clean_summary"], max_length=args.max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = Dataset.from_list(train_rows).map(preprocess, batched=True, remove_columns=list(train_rows[0].keys()))
    eval_dataset = Dataset.from_list(eval_rows).map(preprocess, batched=True, remove_columns=list(eval_rows[0].keys()))
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        **build_training_args_kwargs(
            Seq2SeqTrainingArguments,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
    )
    trainer = Seq2SeqTrainer(
        **build_trainer_kwargs(
            Seq2SeqTrainer,
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            tokenizer=tokenizer,
        )
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(str(args.output_dir / "final"))


def build_prompt_from_batch(batch: dict[str, list], index: int) -> str:
    return build_prompt(
        transcript=batch["transcript"][index],
        corrupted_summary=_batch_value(batch, "corrupted_summary", index) or "",
        prompt_mode=_batch_value(batch, "prompt_mode", index) or "repair",
        timestep=_batch_value(batch, "timestep", index),
        num_steps=_batch_value(batch, "num_steps", index),
        noise_kind=_batch_value(batch, "noise_kind", index) or _batch_value(batch, "strategy", index),
    )


def build_prompt_from_row(row: dict[str, object]) -> str:
    return build_prompt(
        transcript=str(row["transcript"]),
        corrupted_summary=str(row.get("corrupted_summary", "")),
        prompt_mode=str(row.get("prompt_mode", "repair")),
        timestep=row.get("timestep"),
        num_steps=row.get("num_steps"),
        noise_kind=row.get("noise_kind") or row.get("strategy"),
    )


def build_prompt(
    *,
    transcript: str,
    corrupted_summary: str,
    prompt_mode: str,
    timestep: object | None = None,
    num_steps: object | None = None,
    noise_kind: object | None = None,
) -> str:
    if prompt_mode == "generate":
        return build_generation_prompt(transcript)
    if prompt_mode == "repair":
        return build_denoising_prompt(
            transcript,
            corrupted_summary,
            timestep=timestep,
            num_steps=num_steps,
            noise_kind=noise_kind,
        )
    raise ValueError(f"unknown prompt_mode: {prompt_mode}")


def build_generation_prompt(transcript: str) -> str:
    return (
        "Generate the target output using the source context.\n\n"
        f"Source context:\n{transcript}\n\n"
        "Target output:"
    )


def build_denoising_prompt(
    transcript: str,
    corrupted_summary: str,
    *,
    timestep: object | None = None,
    num_steps: object | None = None,
    noise_kind: object | None = None,
) -> str:
    metadata = ""
    if timestep is not None and num_steps is not None:
        metadata += f"Diffusion timestep: {timestep}/{num_steps}\n"
    if noise_kind:
        metadata += f"Noise process: {noise_kind}\n"
    if metadata:
        metadata = metadata + "\n"

    return (
        "Repair the noisy target output using the source context.\n\n"
        f"{metadata}"
        f"Source context:\n{transcript}\n\n"
        f"Noisy target output:\n{corrupted_summary}\n\n"
        "Clean target output:"
    )


def _batch_value(batch: dict[str, list], key: str, index: int) -> object | None:
    if key not in batch:
        return None
    return batch[key][index]


def split_rows_by_example_id(
    rows: list[dict[str, str]],
    *,
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0 < eval_ratio < 1:
        raise ValueError("eval_ratio must be between 0 and 1")

    group_ids = sorted({base_example_id(row["id"]) for row in rows})
    random.Random(seed).shuffle(group_ids)
    eval_count = max(1, round(len(group_ids) * eval_ratio))
    if eval_count >= len(group_ids):
        eval_count = len(group_ids) - 1
    eval_ids = set(group_ids[:eval_count])

    train_rows = [row for row in rows if base_example_id(row["id"]) not in eval_ids]
    eval_rows = [row for row in rows if base_example_id(row["id"]) in eval_ids]
    return train_rows, eval_rows


def base_example_id(row_id: str) -> str:
    return row_id.split("#", 1)[0]


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def build_training_args_kwargs(
    training_args_cls,
    *,
    output_dir: Path,
    epochs: float,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, object]:
    parameters = inspect.signature(training_args_cls.__init__).parameters
    kwargs: dict[str, object] = {
        "output_dir": str(output_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate,
        "logging_steps": 10,
    }

    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = "epoch"

    optional_kwargs = {
        "save_strategy": "epoch",
        "predict_with_generate": True,
        "seed": seed,
        "report_to": [],
    }
    for key, value in optional_kwargs.items():
        if key in parameters:
            kwargs[key] = value

    return kwargs


def build_trainer_kwargs(
    trainer_cls,
    *,
    model,
    training_args,
    train_dataset,
    eval_dataset,
    data_collator,
    tokenizer,
) -> dict[str, object]:
    parameters = inspect.signature(trainer_cls.__init__).parameters
    kwargs: dict[str, object] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }

    if "processing_class" in parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in parameters:
        kwargs["tokenizer"] = tokenizer

    return kwargs


def _require_train_deps() -> None:
    try:
        import datasets  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit('Install training dependencies first: python -m pip install -e ".[train,dev]"') from exc


if __name__ == "__main__":
    main()
