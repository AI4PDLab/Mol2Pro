#!/usr/bin/env python3
"""Create random train/validation/test splits without molecule leakage.

The split is assigned at the molecule level: every record with the same
Canonical SMILES or SELFIES goes to the same split.
"""

import argparse
import json
import os
import random
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create random molecule-level splits without SMILES leakage",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_json", required=True, type=str, help="Path to JSON/JSONL dataset")
    parser.add_argument(
        "--emb_dir",
        default=None,
        type=str,
        help="Deprecated compatibility argument; embeddings are not used by random_splits.py",
    )
    parser.add_argument("--output_path", default=".", type=str, help="Directory where split files are saved")
    parser.add_argument(
        "--cache_dir",
        default="./tmp_cache",
        type=str,
        help="Deprecated compatibility argument; kept so old commands do not fail",
    )
    parser.add_argument("--test_size", default=2000, type=int, help="Number of unique molecules in the test set")
    parser.add_argument("--val_size", default=0.04, type=float, help="Fraction of non-test molecules for validation")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    return parser.parse_args()


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read().strip()

    if not text:
        return []

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        return [data]
    except json.JSONDecodeError:
        records = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records


def molecule_key(example: Dict[str, Any]) -> str | None:
    translation = example.get("translation", {})
    return (
        translation.get("Canonical Smiles")
        or translation.get("SELFIES")
        or translation.get("Reaction")
    )


def collect_molecules(dataset: Iterable[Dict[str, Any]]) -> Set[str]:
    molecules = set()
    for example in dataset:
        mol = molecule_key(example)
        if mol:
            molecules.add(mol)
    return molecules


def sample_without_replacement(pool: Sequence[str], n: int, rng: random.Random) -> Set[str]:
    if n <= 0 or not pool:
        return set()
    n = min(n, len(pool))
    return set(rng.sample(list(pool), n))


def create_molecule_splits(
    unique_molecules: Set[str],
    test_size: int,
    val_size: float,
    seed: int,
) -> Tuple[Set[str], Set[str], Set[str]]:
    rng = random.Random(seed)
    all_molecules = sorted(unique_molecules)
    target_test = min(test_size, len(all_molecules))

    test_mols = sample_without_replacement(all_molecules, target_test, rng)

    remaining = sorted(unique_molecules - test_mols)
    target_val = int(len(remaining) * val_size)

    val_mols = sample_without_replacement(remaining, target_val, rng)

    train_mols = unique_molecules - test_mols - val_mols
    return train_mols, val_mols, test_mols


def check_no_molecule_leakage(train_mols: Set[str], val_mols: Set[str], test_mols: Set[str]) -> None:
    overlaps = {
        "train/val": train_mols & val_mols,
        "train/test": train_mols & test_mols,
        "val/test": val_mols & test_mols,
    }
    leaked = {name: values for name, values in overlaps.items() if values}
    if leaked:
        details = {name: len(values) for name, values in leaked.items()}
        raise ValueError(f"Molecule leakage detected across splits: {details}")


def split_records(
    dataset: Iterable[Dict[str, Any]],
    train_mols: Set[str],
    val_mols: Set[str],
    test_mols: Set[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_data, val_data, test_data = [], [], []
    for example in dataset:
        mol = molecule_key(example)
        if not mol:
            continue
        if mol in test_mols:
            test_data.append(example)
        elif mol in val_mols:
            val_data.append(example)
        elif mol in train_mols:
            train_data.append(example)
    return train_data, val_data, test_data


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle)
            handle.write("\n")


def save_splits(
    dataset: List[Dict[str, Any]],
    train_mols: Set[str],
    val_mols: Set[str],
    test_mols: Set[str],
    args,
) -> Tuple[int, int, int]:
    train_data, val_data, test_data = split_records(dataset, train_mols, val_mols, test_mols)

    full_dataset_filename = os.path.basename(args.dataset_json).replace(".json", "").replace(".jsonl", "")
    base_name = full_dataset_filename

    os.makedirs(args.output_path, exist_ok=True)
    train_json = os.path.join(args.output_path, f"{base_name}_train_seed{args.seed}_molecule_random.json")
    val_json = os.path.join(args.output_path, f"{base_name}_val_seed{args.seed}_molecule_random.json")
    test_json = os.path.join(args.output_path, f"{base_name}_test_seed{args.seed}_molecule_random.json")

    write_jsonl(train_json, train_data)
    write_jsonl(val_json, val_data)
    write_jsonl(test_json, test_data)
    return len(train_data), len(val_data), len(test_data)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset = load_json_or_jsonl(args.dataset_json)
    molecule_set = collect_molecules(dataset)
    print(f"Found {len(molecule_set)} unique molecules in main dataset")

    train_mols, val_mols, test_mols = create_molecule_splits(
        molecule_set,
        args.test_size,
        args.val_size,
        args.seed,
    )
    check_no_molecule_leakage(train_mols, val_mols, test_mols)

    print(f"Total molecules - Train: {len(train_mols)} | Validation: {len(val_mols)} | Test: {len(test_mols)}")

    main_counts = save_splits(dataset, train_mols, val_mols, test_mols, args)
    print(f"Main dataset examples - Train: {main_counts[0]} | Val: {main_counts[1]} | Test: {main_counts[2]}")


if __name__ == "__main__":
    main()
