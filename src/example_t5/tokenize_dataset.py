#!/usr/bin/env python3
"""Tokenize Mol2Pro JSON records for seq2seq training (T5 example).

Loads data **Hub-first** via ``datasets.load_dataset`` when ``--dataset`` is not
an existing local path; otherwise loads JSON/JSONL with the same ``load_dataset``
API (``"json"`` builder). This matches split outputs from ``data/random_splits.py``
and ``data/max_protein_dist_splits.py`` (``train.jsonl``, ``val.jsonl``, etc.).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

DEFAULT_HUB_DATASET = "contributor-anonymous/Mol2Pro-Binder-Dataset"


def get_unknown_token_id(tokenizer) -> Optional[int]:
    """Best-effort unknown-token id for Hugging Face tokenizers."""
    if tokenizer is None:
        return None
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None:
        return unk_id
    unk_token = getattr(tokenizer, "unk_token", None)
    if unk_token is None:
        return None
    vocabulary = getattr(tokenizer, "vocabulary", None)
    if isinstance(vocabulary, dict) and unk_token in vocabulary:
        return vocabulary[unk_token]
    special_tokens = getattr(tokenizer, "special_tokens", None)
    if isinstance(special_tokens, dict) and unk_token in special_tokens:
        return special_tokens[unk_token]
    return None


def parse_kv_pairs(pairs: Optional[List[str]]) -> Optional[Dict[str, str]]:
    if not pairs:
        return None
    out: Dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Expected key=value in --data_files, got: {item!r}")
        key, val = item.split("=", 1)
        out[key.strip()] = val.strip()
    return out


def gather_splits_from_directory(root: str) -> Dict[str, str]:
    """Infer split -> file path for a directory of JSONL/JSON split files."""
    mapping: Dict[str, str] = {}
    # Prefer canonical names written by data/random_splits.py
    candidates = [
        ("train", ["train.jsonl", "train.json"]),
        ("validation", ["val.jsonl", "val.json", "validation.jsonl", "validation.json"]),
        ("test", ["test.jsonl", "test.json"]),
    ]
    for split, filenames in candidates:
        for fn in filenames:
            path = os.path.join(root, fn)
            if os.path.isfile(path):
                mapping[split] = path
                break
    return mapping


def resolve_local_data_files(
    dataset_path: str,
    data_files_kv: Optional[Dict[str, str]],
) -> Union[str, Dict[str, str]]:
    if os.path.isfile(dataset_path):
        return dataset_path
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Local dataset path not found: {dataset_path}")

    if data_files_kv:
        resolved = {}
        for split, rel in data_files_kv.items():
            p = rel if os.path.isabs(rel) else os.path.join(dataset_path, rel)
            if not os.path.isfile(p):
                raise FileNotFoundError(f"--data_files split {split}: file not found: {p}")
            resolved[split] = p
        return resolved

    inferred = gather_splits_from_directory(dataset_path)
    if inferred:
        return inferred

    raise ValueError(
        f"No split JSON/JSONL files found under {dataset_path}. "
        "Provide explicit --data_files train=train.jsonl val=val.jsonl test=test.jsonl "
        "or place train.jsonl / val.jsonl / test.jsonl in that directory."
    )


def load_raw_dataset(
    dataset_arg: str,
    *,
    split: Optional[str],
    cache_dir: Optional[str],
    revision: Optional[str],
    trust_remote_code: bool,
    data_files_kv: Optional[Dict[str, str]],
) -> Union[Dataset, DatasetDict]:
    """Hub repo id or local path -> Dataset or DatasetDict via ``load_dataset``."""
    if os.path.exists(dataset_arg):
        data_files = resolve_local_data_files(dataset_arg, data_files_kv)
        return load_dataset(
            "json",
            data_files=data_files,
            cache_dir=cache_dir,
            split=split,
        )
    kwargs = {
        "cache_dir": cache_dir,
        "split": split,
        "trust_remote_code": trust_remote_code,
    }
    if revision:
        kwargs["revision"] = revision
    return load_dataset(dataset_arg, **kwargs)


def as_dataset_dict(ds: Union[Dataset, DatasetDict], *, single_split_name: str) -> DatasetDict:
    if isinstance(ds, DatasetDict):
        return ds
    return DatasetDict({single_split_name: ds})


def load_tokenizer(path: str) -> Any:
    """Load a Hugging Face ``AutoTokenizer`` from a local path or Hub id."""
    if not path:
        raise ValueError("Tokenizer path is empty")
    return AutoTokenizer.from_pretrained(path)


def detect_source_target(first_translation: Dict[str, Any]) -> Tuple[str, str]:
    """Infer encoder input field and decoder target field inside ``translation``."""
    if first_translation.get("Reaction") is not None:
        src = "Reaction"
        if "Sequence" in first_translation and first_translation.get("Sequence") is not None:
            tgt = "Sequence"
        else:
            tgt = "Protein sequence"
        return src, tgt

    if first_translation.get("SELFIES") is not None:
        src = "SELFIES"
    else:
        src = "Canonical Smiles"

    if "Protein sequence" in first_translation and first_translation.get("Protein sequence") is not None:
        tgt = "Protein sequence"
    elif "Sequence" in first_translation and first_translation.get("Sequence") is not None:
        tgt = "Sequence"
    else:
        tgt = "Protein sequence"

    return src, tgt


def preprocess_builder(
    tokenizer_enc,
    tokenizer_dec,
    *,
    source_lang: str,
    target_lang: str,
    enc_unk_id: Optional[int],
    dec_unk_id: Optional[int],
    max_length: int,
    max_length_aa: int,
):
    """Return a batched preprocess callable for ``Dataset.map``."""

    def preprocess_batch(examples: Dict[str, Any]) -> Dict[str, Any]:
        translations = examples["translation"]
        inputs = [ex[source_lang] for ex in translations]
        targets = [ex[target_lang] for ex in translations]
        batch_size = len(inputs)

        model_inputs = tokenizer_enc(
            inputs,
            max_length=max_length,
            padding=False,
            truncation=True,
        )
        enc_ids = model_inputs.get("input_ids", [])
        enc_total = [len(seq) for seq in enc_ids]
        if enc_unk_id is None:
            enc_unk = [0] * batch_size
        else:
            enc_unk = [sum(1 for t in seq if t == enc_unk_id) for seq in enc_ids]

        labels = tokenizer_dec(
            text_target=targets,
            max_length=max_length_aa,
            padding=False,
            truncation=True,
        )

        dec_ids = labels["input_ids"]
        dec_total = [len(seq) for seq in dec_ids]
        if dec_unk_id is None:
            dec_unk = [0] * batch_size
        else:
            dec_unk = [sum(1 for t in seq if t == dec_unk_id) for seq in dec_ids]

        model_inputs["labels"] = dec_ids
        model_inputs["_encoder_total_tokens"] = enc_total
        model_inputs["_encoder_unknown_tokens"] = enc_unk
        model_inputs["_decoder_total_tokens"] = dec_total
        model_inputs["_decoder_unknown_tokens"] = dec_unk

        has_contacts = any(
            isinstance(ex, dict) and ex.get("AA_binding") is not None for ex in translations
        )
        if has_contacts:
            model_inputs["AA_binding"] = [ex.get("AA_binding", []) for ex in translations]

        return model_inputs

    return preprocess_batch


def strip_stat_columns(ds, columns: List[str]):
    to_drop = [c for c in columns if c in ds.column_names]
    return ds.remove_columns(to_drop) if to_drop else ds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tokenize Mol2Pro-style dataset for seq2seq training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_HUB_DATASET,
        help="Hub dataset id, or an existing path to a JSON/JSONL file or directory of splits.",
    )
    p.add_argument(
        "--data_files",
        action="append",
        default=None,
        metavar="split=file",
        help="Local-only: explicit splits, repeat as needed, e.g. --data_files train=train.jsonl "
        "--data_files validation=val.jsonl --data_files test=test.jsonl",
    )
    p.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional Hugging Face split name; if omitted, loads full DatasetDict when supported.",
    )
    p.add_argument(
        "--single_split_name",
        type=str,
        default="train",
        help="When --split returns a single Dataset, store it under this key in DatasetDict.",
    )
    p.add_argument("--tokenizer_aa", required=True, type=str, help="Protein / AA tokenizer path")
    p.add_argument("--tokenizer_mol", required=True, type=str, help="Molecule tokenizer path")
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for DatasetDict.save_to_disk (tokenized splits).",
    )
    p.add_argument("--cache_dir", type=str, default=None, help="Hugging Face datasets cache dir")
    p.add_argument("--revision", type=str, default=None, help="Hub dataset revision (branch/tag/commit)")
    p.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Forward to load_dataset for Hub datasets that need it.",
    )
    p.add_argument("--max_source_length", type=int, default=512)
    p.add_argument("--max_target_length", type=int, default=1024)
    p.add_argument("--batch_size_map", type=int, default=512)
    p.add_argument(
        "--num_proc",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 4))),
        help="Parallel workers for Dataset.map (lower if Windows workers misbehave).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data_files_kv = parse_kv_pairs(args.data_files)

    raw = load_raw_dataset(
        args.dataset,
        split=args.split,
        cache_dir=args.cache_dir,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        data_files_kv=data_files_kv,
    )
    ds_dict = as_dataset_dict(raw, single_split_name=args.single_split_name)

    # Peek first example for source/target detection
    first_split = next(iter(ds_dict.values()))
    if "translation" not in first_split.column_names:
        raise ValueError(
            "Expected a 'translation' column with nested reaction/SMILES and protein fields. "
            f"Got columns: {first_split.column_names}"
        )
    source_lang, target_lang = detect_source_target(first_split[0]["translation"])
    print(f"Inferred encoder field: {source_lang!r}, decoder field: {target_lang!r}")

    tokenizer_enc = load_tokenizer(args.tokenizer_mol)
    tokenizer_dec = load_tokenizer(args.tokenizer_aa)
    enc_unk_id = get_unknown_token_id(tokenizer_enc)
    dec_unk_id = get_unknown_token_id(tokenizer_dec)

    preprocess = preprocess_builder(
        tokenizer_enc,
        tokenizer_dec,
        source_lang=source_lang,
        target_lang=target_lang,
        enc_unk_id=enc_unk_id,
        dec_unk_id=dec_unk_id,
        max_length=args.max_source_length,
        max_length_aa=args.max_target_length,
    )

    stat_cols = [
        "_encoder_total_tokens",
        "_encoder_unknown_tokens",
        "_decoder_total_tokens",
        "_decoder_unknown_tokens",
    ]

    tokenized_splits: Dict[str, Dataset] = {}
    for split_name, split_ds in ds_dict.items():
        print(f"Tokenizing split {split_name!r} ({len(split_ds)} rows)...")
        mapped = split_ds.map(
            preprocess,
            batched=True,
            batch_size=args.batch_size_map,
            num_proc=args.num_proc,
            remove_columns=split_ds.column_names,
            desc=f"Tokenizing {split_name}",
            load_from_cache_file=False,
            writer_batch_size=1000,
        )
        mapped = strip_stat_columns(mapped, stat_cols)
        tokenized_splits[split_name] = mapped

    tokenized = DatasetDict(tokenized_splits)
    tokenized.save_to_disk(args.output_dir)

    meta = {
        "dataset": args.dataset,
        "split": args.split,
        "revision": args.revision,
        "source_field": source_lang,
        "target_field": target_lang,
        "tokenizer_aa": args.tokenizer_aa,
        "tokenizer_mol": args.tokenizer_mol,
        "max_source_length": args.max_source_length,
        "max_target_length": args.max_target_length,
        "splits": list(tokenized.keys()),
    }
    meta_path = os.path.join(args.output_dir, "tokenize_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved tokenized DatasetDict to {args.output_dir}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
