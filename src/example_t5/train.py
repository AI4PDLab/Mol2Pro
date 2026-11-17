#!/usr/bin/env python3
"""Minimal seq2seq training entry point for tokenized DatasetDict on disk.

Expects the directory produced by ``tokenize_dataset.py`` (``DatasetDict.save_to_disk``).
Extend hyperparameters and metrics as needed for your experiments.
"""

from __future__ import annotations

import argparse
import json
import os

from datasets import DatasetDict, load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

try:
    import torch
except ImportError:
    torch = None  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train seq2seq model on tokenized Mol2Pro data.")
    p.add_argument(
        "--tokenized_dir",
        required=True,
        type=str,
        help="Directory passed to tokenize_dataset.py --output_dir (DatasetDict on disk).",
    )
    p.add_argument(
        "--model_name_or_path",
        required=True,
        type=str,
        help="Base seq2seq checkpoint (e.g. google/t5-small or a local path).",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Directory for checkpoints and logs.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Split name in the tokenized DatasetDict for training.",
    )
    p.add_argument(
        "--eval_split",
        type=str,
        default="validation",
        help="Split name for evaluation (use 'test' if you have no validation).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    ds: DatasetDict = load_from_disk(args.tokenized_dir)
    if args.train_split not in ds:
        raise KeyError(
            f"Train split {args.train_split!r} not in DatasetDict. Available: {list(ds.keys())}"
        )

    train_ds = ds[args.train_split]
    eval_ds = None
    if args.eval_split in ds:
        eval_ds = ds[args.eval_split]
    else:
        print(
            f"[warn] Eval split {args.eval_split!r} not found; training without evaluation. "
            f"Available splits: {list(ds.keys())}"
        )

    meta_path = os.path.join(args.tokenized_dir, "tokenize_metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        tok_path = meta.get("tokenizer_aa", args.model_name_or_path)
    else:
        meta = {}
        tok_path = args.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
    )

    use_cuda = bool(torch and torch.cuda.is_available())
    eval_on = eval_ds is not None
    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy="steps" if eval_on else "no",
        eval_steps=args.eval_steps if eval_on else None,
        save_total_limit=3,
        load_best_model_at_end=False,
        predict_with_generate=False,
        fp16=use_cuda,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    if meta:
        with open(os.path.join(args.output_dir, "train_run_metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"tokenize_metadata": meta, "cli": vars(args)}, f, indent=2)

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Training finished. Model saved under {args.output_dir}")


if __name__ == "__main__":
    main()
