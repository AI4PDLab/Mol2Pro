#!/usr/bin/env python3
"""Prepare Boltz-2 YAML inputs from inference.py outputs.

Selects, per target index, the *best* generation using one of two strategies
(tried in order):

1. **pLDDT** – highest mean pLDDT from average_plddt_<idx>.csv files found
   under <run_dir>/<foldings_subdir>/ (or --foldings_dir).
2. **GT identity** – highest effective_identity from gt_mmseqs/detailed_results.json
   (written by gt_mmseq_interpretation.py).  Used as fallback when pLDDT CSVs
   are unavailable.

The selected sequence is read from the matching FASTA, paired with the SMILES
from molecule_input_metadata.json, and written as single-sequence YAML (msa:
empty, no templates) for boltz predict.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Optional

import pandas as pd


def _as_float(x: object) -> Optional[float]:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if v != v:
        return None
    return v


def _resolve_foldings_dir(run_dir: str, foldings_dir: Optional[str], foldings_subdir: str) -> Optional[str]:
    if foldings_dir:
        return os.path.abspath(foldings_dir)
    cand = os.path.join(run_dir, foldings_subdir)
    if os.path.isdir(cand) and glob_has_average_plddt(cand):
        return os.path.abspath(cand)
    if glob_has_average_plddt(run_dir):
        return os.path.abspath(run_dir)
    return None


def glob_has_average_plddt(d: str) -> bool:
    import glob

    return bool(glob.glob(os.path.join(d, "average_plddt_*.csv")))


def load_top_k(run_dir: str, override: Optional[int], default: int = 15) -> int:
    if override is not None:
        return int(override)
    path = os.path.join(run_dir, "generation_parameters.json")
    if not os.path.isfile(path):
        print(f"[info] {path} not found, using default top_k={default}")
        return default
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    tk = data.get("top_k")
    if tk is None:
        print(f"[info] generation_parameters.json has no top_k, using default={default}")
        return default
    return int(tk)


def parse_generation_index_from_pdb_filename(fname: str) -> Optional[int]:
    """e.g. top_output_topk15_file-34.fasta_3.pdb -> 3."""
    m = re.search(r"\.fasta_(\d+)\.pdb", fname, re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def read_fasta_sequence(fasta_path: str, record_index: int) -> str:
    """FASTA from inference.py uses >0, >1, ... as headers; record_index matches generation."""
    seqs: list[str] = []
    buf: list[str] = []
    with open(fasta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if buf:
                    seqs.append("".join(buf))
                    buf = []
            else:
                buf.append(line)
        if buf:
            seqs.append("".join(buf))
    if record_index < 0 or record_index >= len(seqs):
        raise IndexError(f"Record {record_index} not in {fasta_path} ({len(seqs)} sequences)")
    out = seqs[record_index].replace(" ", "")
    if not out:
        raise ValueError(f"Empty sequence at index {record_index} in {fasta_path}")
    return out


def resolve_smiles(metadata: dict[str, Any], run_dir: str, target_index: int, top_k: int) -> str:
    """Match molecule_input_metadata.json key for this target."""
    candidates = [
        f"output_topk{top_k}_file-{target_index}.fasta",
        os.path.join("fastas", f"output_topk{top_k}_file-{target_index}.fasta"),
        os.path.join(run_dir, f"output_topk{top_k}_file-{target_index}.fasta"),
    ]
    for key in candidates:
        if key in metadata:
            return str(metadata[key])
    # Any key ending with the basename
    suffix = f"output_topk{top_k}_file-{target_index}.fasta"
    for k, v in metadata.items():
        if k.endswith(suffix):
            return str(v)
    raise KeyError(
        f"No SMILES in molecule_input_metadata.json for target {target_index} "
        f"(tried keys like {candidates[0]})"
    )


def _resolve_gt_mmseqs_dir(run_dir: str) -> Optional[str]:
    """Check if gt_mmseqs/ subfolder with detailed_results.json exists."""
    cand = os.path.join(run_dir, "gt_mmseqs")
    if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "detailed_results.json")):
        return os.path.abspath(cand)
    return None


def _load_gt_mmseqs_detailed(gt_mmseqs_dir: str) -> dict[str, Any]:
    path = os.path.join(gt_mmseqs_dir, "detailed_results.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def best_generation_from_gt_mmseqs(
    detailed: dict[str, Any], fasta_name: str
) -> tuple[str, float, int]:
    """Pick the generation with highest effective_identity from detailed_results.json.

    Returns (pdb_key, best_effective_identity, generation_index).
    """
    fasta_entries = detailed.get(fasta_name)
    if not fasta_entries:
        raise KeyError(f"FASTA {fasta_name!r} not in detailed_results.json")

    best_pdb = ""
    best_score = float("-inf")
    best_gen: Optional[int] = None
    for pdb_key, entry in fasta_entries.items():
        eff = _as_float(entry.get("effective_identity"))
        if eff is None:
            continue
        if eff > best_score:
            best_score = eff
            best_pdb = pdb_key
            best_gen = parse_generation_index_from_pdb_filename(pdb_key)

    if best_gen is None or best_score == float("-inf"):
        raise ValueError(f"No valid effective_identity entries for {fasta_name}")
    return best_pdb, best_score, best_gen


def yaml_escape_smiles(smiles: str) -> str:
    """Single-quote YAML string; double internal single quotes."""
    return smiles.replace("'", "''")


def write_boltz_yaml(
    path: str,
    protein_sequence: str,
    smiles: str,
    *,
    protein_id: str = "A",
    ligand_id: str = "B",
) -> None:
    body = "\n".join(
        [
            "version: 1",
            "sequences:",
            "  - protein:",
            f"      id: {protein_id}",
            f'      sequence: "{protein_sequence}"',
            "      msa: empty",
            "  - ligand:",
            f"      id: {ligand_id}",
            f"      smiles: '{yaml_escape_smiles(smiles)}'",
            "properties:",
            "  - affinity:",
            f"      binder: {ligand_id}",
            "",
        ]
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)


def best_row_from_plddt_csv(csv_path: str) -> tuple[str, float, int]:
    df = pd.read_csv(csv_path)
    if "file" not in df.columns:
        raise ValueError(f"{csv_path} missing 'file' column")
    score_col = None
    for c in ("average_b_factor", "average_plddt"):
        if c in df.columns:
            score_col = c
            break
    if score_col is None:
        raise ValueError(f"{csv_path} has neither average_b_factor nor average_plddt")
    best_score = float("-inf")
    best_file = ""
    for _, row in df.iterrows():
        fname = str(row.get("file", "")).strip()
        if not fname:
            continue
        s = _as_float(row.get(score_col))
        if s is None:
            continue
        if s > best_score:
            best_score = s
            best_file = fname
    if not best_file or best_score == float("-inf"):
        raise ValueError(f"No valid rows in {csv_path}")
    gen = parse_generation_index_from_pdb_filename(best_file)
    if gen is None:
        raise ValueError(f"Could not parse generation index from file={best_file!r} in {csv_path}")
    return best_file, best_score, gen


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True, help="Inference output folder (fastas + metadata + CSVs)")
    p.add_argument(
        "--foldings_dir",
        default=None,
        help="Folder with average_plddt_*.csv (default: RUN_DIR/FOLDINGS_SUBDIR or RUN_DIR)",
    )
    p.add_argument(
        "--foldings_subdir",
        default="foldings_840",
        help="If --foldings_dir omitted, try RUN_DIR/<this>/",
    )
    p.add_argument("--top_k", type=int, default=None, help="Override top_k (else generation_parameters.json)")
    p.add_argument(
        "--out_dir",
        default=None,
        help="Where to write inputs/ and manifest.json (default: RUN_DIR/cofolding_results)",
    )
    p.add_argument(
        "--indices",
        default=None,
        help="Comma-separated target indices to include (default: all average_plddt_*.csv found)",
    )
    args = p.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    out_root = os.path.abspath(args.out_dir or os.path.join(run_dir, "cofolding_results"))
    inputs_dir = os.path.join(out_root, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    top_k = load_top_k(run_dir, args.top_k)

    meta_path = os.path.join(run_dir, "molecule_input_metadata.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}")
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)

    # Decide selection strategy: pLDDT (foldings) or GT identity (gt_mmseqs)
    foldings = _resolve_foldings_dir(run_dir, args.foldings_dir, args.foldings_subdir)
    gt_mmseqs_dir = None
    if foldings is None:
        print('Folding folder not found, trying GT identity')
        gt_mmseqs_dir = _resolve_gt_mmseqs_dir(run_dir)
        if gt_mmseqs_dir is None:
            raise FileNotFoundError(
                f"Neither foldings (average_plddt_*.csv under {args.foldings_subdir}/ or run_dir) "
                f"nor gt_mmseqs/detailed_results.json found in {run_dir}."
            )
    else:
        print(f"Folding folder found: {foldings}")

    use_plddt = foldings is not None
    selection_method = "plddt" if use_plddt else "gt_identity"
    print(f"[info] Selection method: {selection_method}")

    wanted: Optional[set[int]] = None
    if args.indices:
        wanted = {int(x.strip()) for x in args.indices.split(",") if x.strip()}

    targets: list[dict[str, Any]] = []

    if use_plddt:
        targets = _select_by_plddt(
            run_dir, foldings, top_k, metadata, wanted, inputs_dir  # type: ignore[arg-type]
        )
    else:
        targets = _select_by_gt_identity(
            run_dir, gt_mmseqs_dir, top_k, metadata, wanted, inputs_dir  # type: ignore[arg-type]
        )

    manifest = {
        "run_dir": run_dir,
        "foldings_dir": foldings,
        "gt_mmseqs_dir": gt_mmseqs_dir,
        "selection_method": selection_method,
        "top_k": top_k,
        "targets": targets,
    }
    manifest_path = os.path.join(out_root, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(targets)} YAMLs under {inputs_dir}")
    print(f"Manifest: {manifest_path}")


def _select_by_plddt(
    run_dir: str,
    foldings: str,
    top_k: int,
    metadata: dict[str, Any],
    wanted: Optional[set[int]],
    inputs_dir: str,
) -> list[dict[str, Any]]:
    import glob

    csv_paths = sorted(glob.glob(os.path.join(foldings, "average_plddt_*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No average_plddt_*.csv in {foldings}")

    index_re = re.compile(r"average_plddt_(\d+)\.csv$")
    targets: list[dict[str, Any]] = []

    for csv_path in csv_paths:
        m = index_re.search(os.path.basename(csv_path))
        if not m:
            continue
        target_index = int(m.group(1))
        if wanted is not None and target_index not in wanted:
            continue
        fasta_name = f"output_topk{top_k}_file-{target_index}.fasta"
        fasta_path = os.path.join(run_dir, fasta_name)
        if not os.path.isfile(fasta_path):
            print(f"[skip] missing FASTA {fasta_path}")
            continue
        try:
            pdb_file, sel_plddt, gen_idx = best_row_from_plddt_csv(csv_path)
        except Exception as e:
            print(f"[skip] {csv_path}: {e}")
            continue
        try:
            smiles = resolve_smiles(metadata, run_dir, target_index, top_k)
        except KeyError as e:
            print(f"[skip] target {target_index}: {e}")
            continue
        try:
            protein_seq = read_fasta_sequence(fasta_path, gen_idx)
        except Exception as e:
            print(f"[skip] target {target_index}: {e}")
            continue

        stem = f"target_{target_index:05d}"
        yaml_path = os.path.join(inputs_dir, f"{stem}.yaml")
        write_boltz_yaml(yaml_path, protein_seq, smiles)

        targets.append(
            {
                "target_index": target_index,
                "stem": stem,
                "yaml_path": yaml_path,
                "fasta_path": fasta_path,
                "generation_index": gen_idx,
                "selected_fold_pdb": pdb_file,
                "selection_score": sel_plddt,
                "selection_metric": "mean_plddt_esmfold",
                "smiles": smiles,
                "protein_sequence": protein_seq,
            }
        )
    return targets


def _select_by_gt_identity(
    run_dir: str,
    gt_mmseqs_dir: str,
    top_k: int,
    metadata: dict[str, Any],
    wanted: Optional[set[int]],
    inputs_dir: str,
) -> list[dict[str, Any]]:
    detailed = _load_gt_mmseqs_detailed(gt_mmseqs_dir)
    fasta_idx_re = re.compile(r"output_topk\d+_file-(\d+)\.fasta$")

    targets: list[dict[str, Any]] = []

    for fasta_name, entries in detailed.items():
        m = fasta_idx_re.search(fasta_name)
        if not m:
            print(f"[skip] cannot parse target index from {fasta_name}")
            continue
        target_index = int(m.group(1))
        if wanted is not None and target_index not in wanted:
            continue

        fasta_path = os.path.join(run_dir, fasta_name)
        if not os.path.isfile(fasta_path):
            print(f"[skip] missing FASTA {fasta_path}")
            continue

        try:
            pdb_key, eff_id, gen_idx = best_generation_from_gt_mmseqs(detailed, fasta_name)
        except Exception as e:
            print(f"[skip] {fasta_name}: {e}")
            continue
        try:
            smiles = resolve_smiles(metadata, run_dir, target_index, top_k)
        except KeyError as e:
            print(f"[skip] target {target_index}: {e}")
            continue
        try:
            protein_seq = read_fasta_sequence(fasta_path, gen_idx)
        except Exception as e:
            print(f"[skip] target {target_index}: {e}")
            continue

        stem = f"target_{target_index:05d}"
        yaml_path = os.path.join(inputs_dir, f"{stem}.yaml")
        write_boltz_yaml(yaml_path, protein_seq, smiles)

        targets.append(
            {
                "target_index": target_index,
                "stem": stem,
                "yaml_path": yaml_path,
                "fasta_path": fasta_path,
                "generation_index": gen_idx,
                "selected_fold_pdb": pdb_key,
                "selection_score": eff_id,
                "selection_metric": "effective_identity_gt_mmseqs",
                "smiles": smiles,
                "protein_sequence": protein_seq,
            }
        )
    return targets


if __name__ == "__main__":
    main()
