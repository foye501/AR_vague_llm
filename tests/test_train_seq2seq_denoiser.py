from pathlib import Path

from ar_gstd.train_seq2seq_denoiser import build_trainer_kwargs, build_training_args_kwargs


class NewTrainingArgs:
    def __init__(self, output_dir, eval_strategy=None, save_strategy=None, report_to=None):
        pass


class OldTrainingArgs:
    def __init__(self, output_dir, evaluation_strategy=None, save_strategy=None, report_to=None):
        pass


class ProcessingClassTrainer:
    def __init__(self, model, args, train_dataset, eval_dataset, data_collator, processing_class=None):
        pass


class TokenizerTrainer:
    def __init__(self, model, args, train_dataset, eval_dataset, data_collator, tokenizer=None):
        pass


class MinimalTrainer:
    def __init__(self, model, args, train_dataset, eval_dataset, data_collator):
        pass


def test_training_args_uses_eval_strategy_when_supported() -> None:
    kwargs = build_training_args_kwargs(
        NewTrainingArgs,
        output_dir=Path("out"),
        epochs=1,
        batch_size=2,
        learning_rate=1e-4,
        seed=7,
    )

    assert kwargs["eval_strategy"] == "epoch"
    assert "evaluation_strategy" not in kwargs


def test_training_args_uses_evaluation_strategy_when_supported() -> None:
    kwargs = build_training_args_kwargs(
        OldTrainingArgs,
        output_dir=Path("out"),
        epochs=1,
        batch_size=2,
        learning_rate=1e-4,
        seed=7,
    )

    assert kwargs["evaluation_strategy"] == "epoch"
    assert "eval_strategy" not in kwargs


def test_trainer_kwargs_use_processing_class_when_supported() -> None:
    kwargs = build_trainer_kwargs(
        ProcessingClassTrainer,
        model="model",
        training_args="args",
        train_dataset="train",
        eval_dataset="eval",
        data_collator="collator",
        tokenizer="tokenizer",
    )

    assert kwargs["processing_class"] == "tokenizer"
    assert "tokenizer" not in kwargs


def test_trainer_kwargs_use_tokenizer_when_supported() -> None:
    kwargs = build_trainer_kwargs(
        TokenizerTrainer,
        model="model",
        training_args="args",
        train_dataset="train",
        eval_dataset="eval",
        data_collator="collator",
        tokenizer="tokenizer",
    )

    assert kwargs["tokenizer"] == "tokenizer"
    assert "processing_class" not in kwargs


def test_trainer_kwargs_omit_tokenizer_when_unsupported() -> None:
    kwargs = build_trainer_kwargs(
        MinimalTrainer,
        model="model",
        training_args="args",
        train_dataset="train",
        eval_dataset="eval",
        data_collator="collator",
        tokenizer="tokenizer",
    )

    assert "tokenizer" not in kwargs
    assert "processing_class" not in kwargs
