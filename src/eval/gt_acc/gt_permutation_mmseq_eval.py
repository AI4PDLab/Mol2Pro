#!/usr/bin/env python3
"""
Permutation GT evaluation (single MMseqs search, pairwise interpretation).

For each query molecule M (25 generations), evaluate against each other molecule
M' GT set **individually**:
  - Per generation: best effective identity among M' GT proteins
  - GT_Acc(M, M')  = max across generations
  - GT_Mean(M, M') = mean across generations (0 if no hit)

Then for each M:
  exclude_self:         average GT_Acc / GT_Mean over all M' != M
  exclude_self_overlap: same, but drop any M' whose GT set contains at least one
                        protein with effective identity > threshold to any protein
                        in M's own GT (the whole M' set is dropped).

Global: mean across all M of the per-M averages (balanced).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import subprocess
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


def _sf(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def read_fasta_sequences(fasta_path: str) -> Dict[str, str]:
    sequences: Dict[str, str] = {}
    current_header: Optional[str] = None
    current_seq: List[str] = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header is not None:
                    sequences[current_header] = "".join(current_seq)
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

        if current_header is not None:
            sequences[current_header] = "".join(current_seq)

    return sequences


def collect_unique_gt_and_per_molecule(
    gt_fasta_folder: str,
) -> Tuple[List[str], Dict[str, Set[str]], Dict[str, str]]:
    with open(os.path.join(gt_fasta_folder, "folder_names.json"), "r") as f:
        folder_names: Dict[str, str] = json.load(f)

    per_mol: Dict[str, Set[str]] = defaultdict(set)
    all_seq: Set[str] = set()

    for smiles, safe in folder_names.items():
        gt_path = os.path.join(gt_fasta_folder, safe, f"{safe}.fasta")
        if not os.path.exists(gt_path):
            print(f"Warning: missing GT fasta {gt_path}")
            continue
        seqs = read_fasta_sequences(gt_path)
        for _h, s in sorted(seqs.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
            per_mol[smiles].add(s)
            all_seq.add(s)

    unique_sorted = sorted(all_seq)
    return unique_sorted, dict(per_mol), folder_names


def load_overlap_pkl(path: str) -> Dict[str, Set[str]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    raw = data.get("overlap_by_sequence", {})
    out: Dict[str, Set[str]] = {}
    for k, v in raw.items():
        out[k] = set(v) if not isinstance(v, set) else v
    return out


def run_mmseqs(
    mmseqs_bin: str,
    query_fa: str,
    target_fa: str,
    out_tab: str,
    tmp_dir: str,
) -> None:
    cmd = [
        mmseqs_bin,
        "easy-search",
        query_fa,
        target_fa,
        out_tab,
        tmp_dir,
        "--min-seq-id",
        "0.0",
        "--format-output",
        "query,target,fident,alnlen,qlen,tlen",
        "-c",
        "0.3",
        "--cov-mode",
        "0",
    ]
    print("[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def write_fasta_indexed(path: str, sequences: List[str]) -> None:
    with open(path, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">{i}\n{seq}\n")


def concat_inference_fastas(
    inference_folder: str,
    molecule_metadata: Dict[str, str],
) -> Tuple[str, List[Tuple[str, str, str, int]]]:
    """
    Returns path to concatenated FASTA and list of
    (fasta_basename, smiles, local_query_key, global_query_index) in order.
    """
    by_basename: Dict[str, str] = {}
    for path, smi in molecule_metadata.items():
        base = os.path.basename(path)
        if base not in by_basename:
            by_basename[base] = smi

    fasta_pattern = re.compile(r"_file-(\d+)\.fasta$")
    files = [fn for fn in os.listdir(inference_folder) if fn.endswith(".fasta")]
    indexed: List[Tuple[int, str]] = []
    for fn in files:
        m = fasta_pattern.search(fn)
        if m:
            indexed.append((int(m.group(1)), fn))
    indexed.sort(key=lambda x: x[0])

    records: List[Tuple[str, str, str, int]] = []
    global_i = 0
    out_lines: List[str] = []

    for _idx, fn in indexed:
        path = os.path.join(inference_folder, fn)
        smiles = by_basename.get(fn)
        if smiles is None:
            smiles = molecule_metadata.get(path)
        if smiles is None:
            print(f"Warning: no metadata for {fn}, skip")
            continue
        seqs = read_fasta_sequences(path)
        keys = sorted(seqs.keys(), key=lambda k: int(k) if str(k).isdigit() else k)
        for k in keys:
            out_lines.append(f">{global_i}\n{seqs[k]}\n")
            records.append((fn, smiles, str(k), global_i))
            global_i += 1

    concat_path = os.path.join(inference_folder, "_permutation_all_queries.fasta")
    with open(concat_path, "w") as f:
        f.write("".join(out_lines))
    print(f"Wrote {global_i} query sequences to {concat_path}")
    return concat_path, records


# ---------------------------------------------------------------------------
# Pairwise hit parsing
# ---------------------------------------------------------------------------

def parse_pairwise_best_hits(
    mmseqs_path: str,
    query_records: List[Tuple[str, str, str, int]],
    unique_targets: List[str],
    mol_to_gt: Dict[str, Set[str]],
) -> Dict[Tuple[str, str], Dict[int, Dict[str, float]]]:
    """
    Parse mmseqs output and keep, for every (query_molecule, target_molecule,
    global_query_index) triple, the best hit (by effective identity).

    A hit from query q (belonging to molecule M) to target t (sequence S) is
    credited to every molecule M' that has S in its GT, excluding M'==M.

    Returns:
        best[(M, M')][global_query_idx] = {"identity": ..., "eff": ...}
    """
    gidx_to_smi: Dict[int, str] = {r[3]: r[1] for r in query_records}

    seq_to_mols: Dict[str, Set[str]] = defaultdict(set)
    for smi, gts in mol_to_gt.items():
        for s in gts:
            seq_to_mols[s].add(smi)

    target_idx_to_mols: Dict[int, Set[str]] = {}
    for idx, seq in enumerate(unique_targets):
        target_idx_to_mols[idx] = seq_to_mols.get(seq, set())

    best: Dict[Tuple[str, str], Dict[int, Dict[str, float]]] = defaultdict(dict)

    if not os.path.exists(mmseqs_path) or os.path.getsize(mmseqs_path) == 0:
        return {}

    with open(mmseqs_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            try:
                q = int(parts[0])
                t = int(parts[1])
            except ValueError:
                continue

            identity = _sf(parts[2], 0.0)
            if len(parts) >= 6:
                alnlen = _sf(parts[3], 0.0)
                tlen = _sf(parts[5], 0.0)
            elif len(parts) >= 5:
                alnlen = _sf(parts[3], 0.0)
                tlen = _sf(parts[4], 0.0)
            else:
                alnlen = 0.0
                tlen = 0.0
            t_cov = (alnlen / tlen) if tlen > 0 else 0.0
            eff = identity * t_cov

            if t < 0 or t >= len(unique_targets):
                continue

            query_smi = gidx_to_smi.get(q)
            if query_smi is None:
                continue

            for target_smi in target_idx_to_mols.get(t, set()):
                pair = (query_smi, target_smi)
                prev = best[pair].get(q)
                if prev is None or eff > prev["eff"]:
                    best[pair][q] = {"identity": identity, "eff": eff}

    return dict(best)


# ---------------------------------------------------------------------------
# Pairwise → per-molecule aggregation
# ---------------------------------------------------------------------------

def compute_pair_metrics(
    gen_data: Dict[int, Dict[str, float]],
    generation_gidxs: List[int],
) -> Dict[str, float]:
    """
    Given hits for one (M, M') pair and M's generation global indices,
    return accuracy / mean (balanced: 0 for missing generations).
    """
    ids: List[float] = []
    effs: List[float] = []
    for gi in generation_gidxs:
        hit = gen_data.get(gi)
        if hit:
            ids.append(hit["identity"])
            effs.append(hit["eff"])
        else:
            ids.append(0.0)
            effs.append(0.0)
    return {
        "accuracy": max(ids) if ids else 0.0,
        "mean_identity": sum(ids) / len(ids) if ids else 0.0,
        "eff_accuracy": max(effs) if effs else 0.0,
        "eff_mean": sum(effs) / len(effs) if effs else 0.0,
    }


def mol_gt_overlaps(
    query_smi: str,
    target_smi: str,
    mol_to_gt: Dict[str, Set[str]],
    overlap: Optional[Dict[str, Set[str]]],
) -> bool:
    """True if any protein in GT(target_smi) is identical to or within the
    overlap neighbourhood (eff > threshold) of any protein in GT(query_smi)."""
    if overlap is None:
        return False
    own_gt = mol_to_gt.get(query_smi, set())
    target_gt = mol_to_gt.get(target_smi, set())
    expanded = set(own_gt)
    for g in own_gt:
        expanded |= overlap.get(g, set())
    return bool(target_gt & expanded)


def aggregate(
    pairwise_best: Dict[Tuple[str, str], Dict[int, Dict[str, float]]],
    query_records: List[Tuple[str, str, str, int]],
    mol_to_gt: Dict[str, Set[str]],
    overlap: Optional[Dict[str, Set[str]]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict]:
    """
    Returns (per_mol_self, per_mol_olp, summary).
    """
    per_mol_selfonly: Dict[str, Dict[str, float]] = {}
    smi_to_gidxs: Dict[str, List[int]] = defaultdict(list)
    for _fn, smi, _lk, gidx in query_records:
        smi_to_gidxs[smi].append(gidx)
    for smi in smi_to_gidxs:
        smi_to_gidxs[smi].sort()

    all_query_smiles = sorted(smi_to_gidxs.keys())
    all_target_smiles = sorted(mol_to_gt.keys())

    per_mol_self: Dict[str, Dict[str, float]] = {}
    per_mol_olp: Dict[str, Dict[str, float]] = {}

    metric_keys = ["accuracy", "mean_identity", "eff_accuracy", "eff_mean"]

    for query_smi in all_query_smiles:
        gidxs = smi_to_gidxs[query_smi]
        vals_self: Dict[str, List[float]] = {k: [] for k in metric_keys}
        vals_olp: Dict[str, List[float]] = {k: [] for k in metric_keys}

        self_gen_data = pairwise_best.get((query_smi, query_smi), {})
        self_metrics = compute_pair_metrics(self_gen_data, gidxs)
        per_mol_selfonly[query_smi] = dict(self_metrics)
        per_mol_selfonly[query_smi]["n_compared"] = 1
        
        for target_smi in all_target_smiles:
            if target_smi == query_smi:
                continue
            pair = (query_smi, target_smi)
            gen_data = pairwise_best.get(pair, {})
            metrics = compute_pair_metrics(gen_data, gidxs)

            for k in metric_keys:
                vals_self[k].append(metrics[k])
            if not mol_gt_overlaps(query_smi, target_smi, mol_to_gt, overlap):
                for k in metric_keys:
                    vals_olp[k].append(metrics[k])

        per_mol_self[query_smi] = {
            k: (sum(v) / len(v) if v else 0.0) for k, v in vals_self.items()
        }
        per_mol_self[query_smi]["n_compared"] = len(vals_self["accuracy"])

        per_mol_olp[query_smi] = {
            k: (sum(v) / len(v) if v else 0.0) for k, v in vals_olp.items()
        }
        per_mol_olp[query_smi]["n_compared"] = len(vals_olp["accuracy"])

    def _global(d: Dict[str, Dict[str, float]], key: str) -> float:
        v = [x[key] for x in d.values()]
        return sum(v) / len(v) if v else 0.0

    summary = {
        "exclude_self_only": {
            "mean_accuracy": _global(per_mol_self, "accuracy"),
            "mean_identity": _global(per_mol_self, "mean_identity"),
            "mean_eff_accuracy": _global(per_mol_self, "eff_accuracy"),
            "mean_eff_mean": _global(per_mol_self, "eff_mean"),
            "n_molecules": len(per_mol_self),
        },
        "exclude_self_and_overlap": {
            "mean_accuracy": _global(per_mol_olp, "accuracy"),
            "mean_identity": _global(per_mol_olp, "mean_identity"),
            "mean_eff_accuracy": _global(per_mol_olp, "eff_accuracy"),
            "mean_eff_mean": _global(per_mol_olp, "eff_mean"),
            "n_molecules": len(per_mol_olp),
        },
        "self_only": {
          "mean_accuracy": _global(per_mol_selfonly, "accuracy"),
          "mean_identity": _global(per_mol_selfonly, "mean_identity"),
          "mean_eff_accuracy": _global(per_mol_selfonly, "eff_accuracy"),
          "mean_eff_mean": _global(per_mol_selfonly, "eff_mean"),
          "n_molecules": len(per_mol_selfonly),
        },
    }

    return per_mol_selfonly, per_mol_self, per_mol_olp, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Permutation GT MMseqs evaluation (pairwise).")
    ap.add_argument("--inference_folder", required=True)
    ap.add_argument("--gt_fasta_folder", required=True, help="output_folder_gt with folder_names.json")
    ap.add_argument("--overlap_pkl", default="", help="Optional pickle from permutation_GT_overlap.py")
    ap.add_argument("--mmseqs", default="mmseqs")
    ap.add_argument("--tmp_dir", default="")
    ap.add_argument(
        "--skip_mmseqs",
        action="store_true",
        help="Reuse existing mmseqs TSV under inference_folder/gt_mmseqs_permutation/",
    )
    args = ap.parse_args()

    meta_files = [f for f in os.listdir(args.inference_folder) if "molecule_input_metadata" in f]
    if not meta_files:
        raise FileNotFoundError("molecule_input_metadata*.json not found")
    with open(os.path.join(args.inference_folder, meta_files[0]), "r") as f:
        molecule_metadata = json.load(f)

    unique_targets, mol_to_gt, _ = collect_unique_gt_and_per_molecule(args.gt_fasta_folder)
    if not unique_targets:
        raise RuntimeError("No GT sequences found under gt_fasta_folder.")

    overlap: Optional[Dict[str, Set[str]]] = None
    if args.overlap_pkl:
        overlap = load_overlap_pkl(args.overlap_pkl)
        print(f"Loaded overlap map for {len(overlap)} sequences from {args.overlap_pkl}")

    target_fa = os.path.join(args.inference_folder, "_permutation_gt_targets.fasta")
    write_fasta_indexed(target_fa, unique_targets)

    query_fa, query_records = concat_inference_fastas(args.inference_folder, molecule_metadata)
    if not query_records:
        raise RuntimeError("No inference queries built.")

    out_dir = os.path.join(args.inference_folder, "gt_mmseqs_permutation")
    os.makedirs(out_dir, exist_ok=True)
    out_tab = os.path.join(out_dir, "mmseqs_all_vs_pooled_gt.m8")

    if not args.skip_mmseqs or not os.path.exists(out_tab):
        tmp = args.tmp_dir or tempfile.mkdtemp(prefix="mmseqs_perm_")
        if not args.tmp_dir:
            print(f"Using temp dir {tmp}")
        try:
            run_mmseqs(args.mmseqs, query_fa, target_fa, out_tab, tmp)
        finally:
            if not args.tmp_dir:
                import shutil
                shutil.rmtree(tmp, ignore_errors=True)
    else:
        print(f"Reusing {out_tab}")

    print("Parsing pairwise hits...")
    pairwise_best = parse_pairwise_best_hits(
        out_tab, query_records, unique_targets, mol_to_gt,
    )
    print(f"Parsed {len(pairwise_best)} (query_mol, target_mol) pairs with hits")

    print("Aggregating...")
    per_mol_selfonly, per_mol_self, per_mol_olp, summary = aggregate(
        pairwise_best, query_records, mol_to_gt, overlap,
    )

    summary_path = os.path.join(out_dir, "permutation_stats.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    per_path = os.path.join(out_dir, "per_molecule_stats.json")
    with open(per_path, "w") as f:
        json.dump(
            {
                "self_only": per_mol_selfonly,
                "exclude_self_only": per_mol_self,
                "exclude_self_and_overlap": per_mol_olp,
            },
            f,
            indent=2,
        )
    print(f"Wrote {summary_path} and {per_path}")


if __name__ == "__main__":
    main()
