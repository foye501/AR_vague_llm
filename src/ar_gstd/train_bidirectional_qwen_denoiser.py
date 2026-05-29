from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bidirectional_denoising import build_bidirectional_denoising_features
from .train_seq2seq_denoiser import build_trainer_kwargs, build_training_args_kwargs, split_rows_by_example_id, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a bidirectional Qwen-family masked denoiser.")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--tokenizer-name", default="", help="Defaults to --model-name.")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--mask-token", default="[MASK]")
    parser.add_argument("--pad-token", default="[PAD]")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--save-strategy", default="no", choices=("no", "epoch", "steps"))
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--eval-file", type=Path, default=None)
    parser.add_argument("--train-split-output", type=Path, default=None)
    parser.add_argument("--eval-split-output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--max-target-length", type=int, default=256)
    args = parser.parse_args()

    _require_train_deps()
    from datasets import Dataset
    from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments

    from .bidirectional_qwen import load_bidirectional_qwen_for_masked_lm, load_tokenizer_with_diffusion_tokens

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

    tokenizer = load_tokenizer_with_diffusion_tokens(
        AutoTokenizer,
        tokenizer_name=args.tokenizer_name or args.model_name,
        pad_token=args.pad_token,
        mask_token=args.mask_token,
    )
    model = load_bidirectional_qwen_for_masked_lm(
        AutoConfig,
        model_name=args.model_name,
        tokenizer=tokenizer,
        from_scratch=args.from_scratch,
    )
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    def preprocess(batch):
        features = []
        keys = list(batch)
        for index in range(len(batch["transcript"])):
            row = {key: batch[key][index] for key in keys}
            item = build_bidirectional_denoising_features(
                row,
                tokenizer=tokenizer,
                max_sequence_length=args.max_sequence_length,
                max_target_length=args.max_target_length,
            )
            features.append(
                {
                    "input_ids": item.input_ids,
                    "attention_mask": item.attention_mask,
                    "labels": item.labels,
                }
            )
        return {
            "input_ids": [item["input_ids"] for item in features],
            "attention_mask": [item["attention_mask"] for item in features],
            "labels": [item["labels"] for item in features],
        }

    train_dataset = Dataset.from_list(train_rows).map(preprocess, batched=True, remove_columns=list(train_rows[0].keys()))
    eval_dataset = Dataset.from_list(eval_rows).map(preprocess, batched=True, remove_columns=list(eval_rows[0].keys()))

    training_args = TrainingArguments(
        **build_training_args_kwargs(
            TrainingArguments,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed,
            bf16=args.bf16,
            fp16=args.fp16,
            gradient_checkpointing=args.gradient_checkpointing,
            save_strategy=args.save_strategy,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
        )
    )
    trainer = Trainer(
        **build_trainer_kwargs(
            Trainer,
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=MaskedDenoisingCollator(pad_token_id=tokenizer.pad_token_id),
            tokenizer=tokenizer,
        )
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(str(args.output_dir / "final"))


class MaskedDenoisingCollator:
    def __init__(self, *, pad_token_id: int, label_pad_token_id: int = -100) -> None:
        self.pad_token_id = int(pad_token_id)
        self.label_pad_token_id = int(label_pad_token_id)

    def __call__(self, features):
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [self.label_pad_token_id] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _require_train_deps() -> None:
    try:
        import datasets  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit('Install training dependencies first: python -m pip install -e ".[train,dev]"') from exc


if __name__ == "__main__":
    main()
