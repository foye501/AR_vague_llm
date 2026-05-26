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
    random.Random(args.seed).shuffle(rows)
    split_at = max(1, int(len(rows) * (1 - args.eval_ratio)))
    train_rows = rows[:split_at]
    eval_rows = rows[split_at:] or rows[:1]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    def preprocess(batch):
        inputs = [
            build_denoising_prompt(transcript, corrupted)
            for transcript, corrupted in zip(batch["transcript"], batch["corrupted_summary"], strict=True)
        ]
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


def build_denoising_prompt(transcript: str, corrupted_summary: str) -> str:
    return (
        "Repair the noisy structured summary using the transcript.\n\n"
        f"Transcript:\n{transcript}\n\n"
        f"Noisy structured summary:\n{corrupted_summary}\n\n"
        "Clean structured summary:"
    )


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
