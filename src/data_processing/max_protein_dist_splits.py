#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Algorithm:
Given a novelty per protein calculated as avg mmseqs distance to dataset, the novelty of each molecule is the avg of
 the novelty-scores of its associated proteins.

The novelty-score is the inverse of mmseqs identity of the protein scaled by the number of proteins associated to all
the molecules that it binds. This is the second level connectivity of the protein.

Then molecules are selected starting from the highest scoring ones and their cascading dependencies are included.

This version augments cascading to be protein-component aware:
Two proteins are in the same component if their "effective MMseqs identity" satisfies:
    eff_id = identity * (alnlen / max(qlen, tlen))  >=  eff_id_thresh
(identity is expected in [0,1] already; NOT divided by 100.)
Cascading expands via these protein components, not only exact sequence equality.

Missing proteins (no MMseqs identity found) are assigned identity based on protein-length Gaussian tail probability:
  - Compute μ, σ across dataset protein lengths
  - p_tail = 2 * (1 - Φ(|(len-μ)/σ|))
  - If p_tail > 0.10 -> in-distribution -> identity = 0.0
  - Else            -> outlier        -> identity = 1.0
"""

import argparse
import json
import random
from collections import defaultdict, deque
import numpy as np
import gc
import math
import os

try:
    from rdkit import Chem  # noqa: F401
except ImportError:
    print("Not loading RDKit")

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Create MMseqs/max-protein-distance molecule-level splits',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_json',
                        required=True,
                        type=str,
                        help='Path to JSON dataset')
    parser.add_argument('--dataset_fasta',
                        required=True,
                        type=str,
                        help='Path to FASTA with headers used in MMseqs output')
    parser.add_argument('--dataset_mmseqs',
                        default=None,
                        type=str,
                        help='Path to MMseqs results (optional). Expected columns: q, t, identity, alnlen, qlen, tlen (tab-separated).')
    parser.add_argument('--cache_dir',
                        default='./tmp_cache',
                        type=str,
                        help='Cache path')
    parser.add_argument('--seed',
                        default=0,
                        type=int,
                        help='Random seed')
    parser.add_argument('--val_size',
                        default=0.04,
                        type=float,
                        help='Proportion of molecules to include in the validation set')
    parser.add_argument('--test_ratio',
                        default=0.10,
                        type=float,
                        help='Proportion of molecules to include in the test set')
    parser.add_argument('--eff_id_thresh',
                        default=1.0,
                        type=float,
                        help='Threshold for effective identity to union proteins into the same component')
    parser.add_argument('--min_frac',
                        default=0.7,
                        type=float,
                        help='Minimum fraction of accepted cascade molecules that must be non-boundary to accept a killed cascade')
    parser.add_argument('--out_suffix',
                        default=None,
                        type=str,
                        help='Suffix for output file names')
    return parser.parse_args()


def correspondence_from_fasta(fasta_path):
    name_to_sequence = {}
    with open(fasta_path, 'r') as f:
        name = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:]
            else:
                name_to_sequence[name] = line
    return name_to_sequence


# =========================
# FO-sum with int IDs + CSR
# =========================
def compute_protein_connectivity_scores(molecule_to_proteins, protein_identity, protein_frequency):
    """
    FO-sum connectivity:
      penalty(p) = sum_{m in N(p)} deg(m)
    score[p] = (identity[p] * frequency[p]) * penalty(p)
    """
    print("Computing protein connectivity scores (FO-sum, int IDs + CSR)...", flush=True)

    proteins = list(protein_frequency.keys())
    P = len(proteins)
    prot2id = {p: i for i, p in enumerate(proteins)}

    molecules = list(molecule_to_proteins.keys())
    M = len(molecules)
    mol2id = {m: i for i, m in enumerate(molecules)}

    # per-molecule degree
    mol_deg = np.empty(M, dtype=np.int32)
    for m, protset in molecule_to_proteins.items():
        mol_deg[mol2id[m]] = len(protset)

    # first pass: counts per protein
    counts = np.zeros(P, dtype=np.int32)
    get_pid = prot2id.get
    for protset in molecule_to_proteins.values():
        for p in protset:
            pid = get_pid(p)
            if pid is not None:
                counts[pid] += 1

    # build CSR protein->molecules
    indptr = np.empty(P + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    E = int(indptr[-1])
    pro2mol = np.empty(E, dtype=np.int32)

    write_pos = indptr[:-1].copy()
    for m, protset in molecule_to_proteins.items():
        mid = mol2id[m]
        for p in protset:
            pid = prot2id.get(p)
            if pid is not None:
                idx = write_pos[pid]
                pro2mol[idx] = mid
                write_pos[pid] = idx + 1

    print(f"Graph sizes -> Proteins: {P:,} | Molecules: {M:,} | Pairs(E): {E:,}", flush=True)

    # FO-sum penalty (safe even when some proteins have 0 edges)
    values = mol_deg[pro2mol]
    acc = np.empty(E + 1, dtype=np.int64)
    acc[0] = 0
    if E:
        np.cumsum(values, out=acc[1:])
    penalty = acc[indptr[1:]] - acc[indptr[:-1]]
    has_edges = (np.diff(indptr) > 0)

    # base = identity * frequency
    iden = np.fromiter((protein_identity.get(p, 1.0) for p in proteins), dtype=np.float32, count=P)
    freq = np.fromiter((protein_frequency.get(p, 1) for p in proteins), dtype=np.int64, count=P)
    base = iden * freq

    scores = base * penalty
    scores = np.where(has_edges, scores, base)

    protein_connectivity_scores = {proteins[i]: float(scores[i]) for i in range(P)}
    print(f"Computed connectivity scores for {len(protein_connectivity_scores)} proteins", flush=True)
    return protein_connectivity_scores


# ----------------------
# Disjoint Set Union
# ----------------------
class DSU:
    __slots__ = ("p", "r")
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


def build_effective_identity_components(mmseqs_path, fasta_path, dataset_protein_set, eff_id_thresh=1.0):
    """
    eff_id = identity * (alnlen / max(qlen, tlen))
    identity is expected already in [0,1].
    Build components among ONLY proteins that appear in the dataset.
    """
    name_to_sequence = correspondence_from_fasta(fasta_path)

    protein_seqs = sorted(dataset_protein_set)
    protein_seq_to_id = {seq: i for i, seq in enumerate(protein_seqs)}
    dsu = DSU(len(protein_seqs))

    if mmseqs_path is None:
        return dsu, protein_seq_to_id

    total_lines = 0
    used_pairs = 0
    union_ops = 0

    with open(mmseqs_path, 'r') as f:
        for line in f:
            total_lines += 1
            parts = line.rstrip('\t\r\n ').split('\t')
            if len(parts) < 6:
                continue
            q, t = parts[0], parts[1]
            if q == t:
                continue

            try:
                identity = float(parts[2])
                alnlen   = float(parts[3])
                qlen     = float(parts[4])
                tlen     = float(parts[5])
            except ValueError:
                continue

            if not (math.isfinite(identity) and math.isfinite(alnlen) and math.isfinite(qlen) and math.isfinite(tlen)):
                continue
            if qlen <= 0 or tlen <= 0 or alnlen <= 0:
                continue

            q_seq = name_to_sequence.get(q)
            t_seq = name_to_sequence.get(t)
            if q_seq is None or t_seq is None:
                continue
            qi = protein_seq_to_id.get(q_seq)
            ti = protein_seq_to_id.get(t_seq)
            if qi is None or ti is None or qi == ti:
                continue
            # Normalize identity to [0,1] if needed (MMseqs often emits 0..100)
            if identity > 1.0:
                identity /= 100.0
            eff_id = identity * (alnlen / max(qlen, tlen))

            used_pairs += 1
            if eff_id >= eff_id_thresh:
                dsu.union(qi, ti)
                union_ops += 1

    print(f"MMseqs parsed: {total_lines} lines | usable pairs in-dataset: {used_pairs} | unions: {union_ops}", flush=True)
    return dsu, protein_seq_to_id


def build_component_mappings(molecule_to_proteins, protein_seq_to_id, dsu):
    protein_seq_to_comp = {}
    for seq, pid in protein_seq_to_id.items():
        protein_seq_to_comp[seq] = dsu.find(pid)

    component_to_molecules = defaultdict(set)
    molecule_to_components = defaultdict(set)

    def singleton_comp(seq):
        return - (abs(hash(seq)) % (10**9) + 1)

    for mol, prots in molecule_to_proteins.items():
        comps = set()
        for pseq in prots:
            pid = protein_seq_to_id.get(pseq)
            if pid is None:
                comp = singleton_comp(pseq)
            else:
                comp = dsu.find(pid)
            comps.add(comp)

        molecule_to_components[mol] = comps
        for comp in comps:
            component_to_molecules[comp].add(mol)

    return protein_seq_to_comp, component_to_molecules, molecule_to_components


def compute_molecule_degree_stats(component_to_molecules, molecule_to_components):
    comp2mols = component_to_molecules

    total_deg = 0
    min_deg = None
    max_deg = 0
    n = 0

    for mol, comps in molecule_to_components.items():
        if comps:
            neighbors = set().union(*(comp2mols[c] for c in comps))
            neighbors.discard(mol)
            d = len(neighbors)
        else:
            d = 0

        total_deg += d
        max_deg = d if d > max_deg else max_deg
        min_deg = d if (min_deg is None or d < min_deg) else min_deg
        n += 1

    mean_deg = (total_deg / n) if n else 0.0
    if min_deg is None:
        min_deg = 0
    return mean_deg, min_deg, max_deg


def cascade_closure_with_depth_components(
    start_mol, max_len_list, test_mols, blacklist_mols,
    component_to_molecules, molecule_to_components, molecule_to_proteins,
    min_frac=0.7, excluded_mols=frozenset(),
):
    """
    Level-wise cascade via protein components.

    We accept complete levels: {depth = 0, 1, ..., L}. If adding the next level (L+1)
    would exceed max_len_list (or max_depth is reached), we kill the not-added frontier.
    Killed molecules are computed from the raw frontier (neighbors via components minus only visited|test)
    and go to validation; the already-accepted molecules form the cascade and go to test if
    A/(A+B) ≥ min_frac, where A=|accepted so far|, B=|killed frontier|. Otherwise we reject (blacklist)
    and leave them for train. `excluded_mols` are never added (e.g., previously flagged boundary/validation).
    Returns:
      visited_mols  : accepted molecules of this cascade (test)
      accepted      : bool
      boundary_set  : killed frontier molecules (validation)
      unique_components_count, total_pairs, depth
    """
    assert not (start_mol in blacklist_mols or start_mol in test_mols or start_mol in excluded_mols)

    visited_mols = set()
    used_components = set()
    unique_components = set()
    total_pairs = 0
    depth = 0

    # Debug logging toggle (export CASCADE_DEBUG=1 to enable)
    debug_on = os.environ.get("CASCADE_DEBUG") == "1"

    def _log_decision(kind, killed_nodes):
        if not debug_on:
            return
        A = len(visited_mols)
        B = len(killed_nodes)
        denom = A + B
        ratio = (A / denom) if denom > 0 else float('nan')
        ok = (A > 0) and (ratio >= min_frac)
        print(f"[CASCADE] {kind}: A={A} B={B} ratio={ratio:.4f} min_frac={min_frac} accepted={ok}", flush=True)

    def add_level(level_nodes):
        nonlocal total_pairs
        for m in level_nodes:
            if m in visited_mols or m in test_mols or m in blacklist_mols or m in excluded_mols:
                continue
            visited_mols.add(m)
            total_pairs += len(molecule_to_proteins[m])
            unique_components.update(molecule_to_components[m])

    def compute_frontier(from_level):
        """Compute the FULL next frontier (raw neighbors via components minus visited|test)."""
        next_frontier = set()
        for m in from_level:
            for c in molecule_to_components[m]:
                if c in used_components:
                    continue
                used_components.add(c)
                next_frontier.update(component_to_molecules.get(c, ()))
        # Remove already-accepted and already-in-test
        next_frontier.difference_update(visited_mols | test_mols)
        return next_frontier

    def min_frac_test(killed_frontier):
        A = len(visited_mols)
        B = len(killed_frontier)
        if A == 0:
            return False, set()
        frac_test = A / (A + B) if (A + B) > 0 else 1.0
        return (frac_test >= min_frac), set(killed_frontier)

    frontier_full = {start_mol}  # full frontier for kill decisions at current level
    frontier = {start_mol}       # filtered frontier actually attempted to add at this level

    while frontier:
        # Would adding this level overflow?
        if len(visited_mols) + len(frontier) > max_len_list:
            _log_decision("current_overflow", frontier_full)
            ok, boundary = min_frac_test(frontier_full)  # boundary = killed frontier (full current level)
            if ok:
                return visited_mols, True, boundary, len(unique_components), total_pairs, depth
            else:
                return visited_mols, False, boundary, len(unique_components), total_pairs, depth  # To blacklist

        # Accept current level
        add_level(frontier)

        # Build next level (FULL frontier)
        next_frontier = compute_frontier(frontier)
        # Filter for actual addition attempt (exclude blacklist/excluded; visited/test are already removed)
        next_to_add = next_frontier - blacklist_mols - excluded_mols

        # If no candidates can be added:
        if not next_to_add:
            if debug_on:
                print("[CASCADE] finished: no addable nodes; accept; migrate full frontier to validation", flush=True)
            # Accept; migrate the full (raw) frontier to validation to preserve closure
            boundary = next_frontier
            return visited_mols, True, boundary, len(unique_components), total_pairs, depth

        # Would adding next level overflow?
        if len(visited_mols) + len(next_frontier) > max_len_list:
            _log_decision("next_overflow", next_frontier)
            ok, boundary = min_frac_test(next_frontier)
            if ok:
                return visited_mols, True, boundary, len(unique_components), total_pairs, depth
            else:
                return visited_mols, False, boundary, len(unique_components), total_pairs, depth  # To blacklist

        frontier_full = next_frontier
        frontier = next_to_add
        depth += 1

    return visited_mols, True, set(), len(unique_components), total_pairs, depth


# ---------- Gaussian tail probability helpers (no SciPy) ----------
SQRT2 = math.sqrt(2.0)
def std_normal_cdf(z_abs):
    return 0.5 * (1.0 + math.erf(z_abs / SQRT2))
def length_tail_probability(x, mu, sigma):
    if sigma <= 0:
        return 1.0
    z = abs((x - mu) / sigma)
    return 2.0 * (1.0 - std_normal_cdf(z))


def main():
    args = parse_args()
    random.seed(args.seed)

    # Load dataset
    print(args.dataset_json)
    raw_dataset = load_dataset(
        'json',
        data_files=args.dataset_json,
        cache_dir=args.cache_dir,
    )['train'].to_list()

    molecule_to_proteins = defaultdict(set)
    mol_encoding = 'Canonical Smiles'
    selfies = False
    protein_frequency = defaultdict(int)
    dataset_protein_set = set()

    for example in raw_dataset:
        try:
            mol = example['translation'][mol_encoding]
        except Exception:
            mol_encoding = 'SELFIES'
            selfies = True
            mol = example['translation'][mol_encoding]

        protein_sequence = example["translation"]["Protein sequence"]
        molecule_to_proteins[mol].add(protein_sequence)
        protein_frequency[protein_sequence] += 1
        dataset_protein_set.add(protein_sequence)

    print('molecule_to_proteins and protein frequency built', flush=True)

    # Length distribution
    protein_lengths = np.array([len(p) for p in dataset_protein_set], dtype=float)
    mu_len = float(protein_lengths.mean()) if protein_lengths.size else 0.0
    sigma_len = float(protein_lengths.std(ddof=0)) if protein_lengths.size else 0.0
    print(f"Protein length mean={mu_len:.2f}, std={sigma_len:.2f}", flush=True)
    print("No-match proteins are considered novel when above 10% probability of being correct", flush=True)
    alpha = 0.10

    # Load MMseqs identities -> effective identity per query (0..1), fill missing via length-probability
    protein_identity = {}
    if args.dataset_mmseqs:
        print("Loading MMseqs results for identity averages (using effective identity, skipping self-hits)...", flush=True)
        protein_identity_list = defaultdict(list)
        name_to_sequence = correspondence_from_fasta(args.dataset_fasta)

        with open(args.dataset_mmseqs, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 6:
                    continue
                q, t = parts[0], parts[1]
                if q == t:
                    continue
                try:
                    identity = float(parts[2])
                    alnlen   = float(parts[3])
                    qlen     = float(parts[4])
                    tlen     = float(parts[5])
                except ValueError:
                    continue
                if qlen <= 0 or tlen <= 0 or alnlen <= 0:
                    continue

                # Normalize identity to [0,1] if needed (MMseqs often emits 0..100)
                if identity > 1.0:
                    identity /= 100.0
                eff_id = identity * (alnlen / max(qlen, tlen))
                protein_seq = name_to_sequence.get(q)
                if protein_seq:
                    protein_identity_list[protein_seq].append(eff_id)

        protein_identity = {p: (float(np.mean(ids)) if ids else 0.0) for p, ids in protein_identity_list.items()}

        fill_counts = {"inlier": 0, "outlier": 0}
        for p in dataset_protein_set:
            if p not in protein_identity:
                p_tail = length_tail_probability(len(p), mu_len, sigma_len)
                if p_tail > alpha:
                    protein_identity[p] = 0.0
                    fill_counts["inlier"] += 1
                else:
                    protein_identity[p] = 1.0
                    fill_counts["outlier"] += 1
        print(f"Filled missing identities via length probability: inliers={fill_counts['inlier']} (id=0.0), outliers={fill_counts['outlier']} (id=1.0)", flush=True)
    else:
        print("No MMseqs file provided - assigning identities from length probability rule", flush=True)
        fill_counts = {"inlier": 0, "outlier": 0}
        for p in dataset_protein_set:
            p_tail = length_tail_probability(len(p), mu_len, sigma_len)
            if p_tail > alpha:
                protein_identity[p] = 0.0
                fill_counts["inlier"] += 1
            else:
                protein_identity[p] = 1.0
                fill_counts["outlier"] += 1
        print(f"Assigned identities (no MMseqs): inliers={fill_counts['inlier']} (id=0.0), outliers={fill_counts['outlier']} (id=1.0)", flush=True)

    identity_values = list(protein_identity.values()) or [0.0]
    print(f'Identity avg: {np.mean(identity_values):.4f} | Max: {np.max(identity_values):.4f} | Min: {np.min(identity_values):.4f}\n')

    # Connectivity scores (FO-sum + int IDs)
    protein_connectivity_scores = compute_protein_connectivity_scores(
        molecule_to_proteins, protein_identity, protein_frequency
    )

    # Molecule base ordering score
    mol_avg_connectivity_score = {}
    for mol, proteins in molecule_to_proteins.items():
        scores = [protein_connectivity_scores[p] for p in proteins if p in protein_connectivity_scores]
        if scores:
            mol_avg_connectivity_score[mol] = float(np.mean(scores))

    # Free some memory
    del protein_frequency
    del protein_identity
    del protein_connectivity_scores
    gc.collect()

    # Components BEFORE sorting to add a component-size risk term
    print("Building protein components via effective identity...", flush=True)
    dsu, protein_seq_to_id = build_effective_identity_components(
        args.dataset_mmseqs,
        args.dataset_fasta,
        dataset_protein_set,
        eff_id_thresh=args.eff_id_thresh
    )
    protein_seq_to_comp, component_to_molecules, molecule_to_components = build_component_mappings(
        molecule_to_proteins, protein_seq_to_id, dsu
    )

    for c in list(component_to_molecules):
        component_to_molecules[c] = tuple(component_to_molecules[c])

    comp_size = {c: len(ms) for c, ms in component_to_molecules.items()}

    def comp_risk(mol):
        comps = molecule_to_components[mol]
        if not comps:
            return 0.0
        return sum(math.log1p(comp_size.get(c, 1)) for c in comps) / len(comps)

    gamma = 0.7
    mol_score_adj = {mol: (base + gamma * comp_risk(mol)) for mol, base in mol_avg_connectivity_score.items()}

    # Order molecules by adjusted score (safer seeds first)
    sorted_mols = sorted(mol_score_adj.items(), key=lambda x: x[1])
    mol_ids_sorted = [m for m, _ in sorted_mols]

    total_molecules = len(mol_ids_sorted)
    test_target_mols = int(total_molecules * args.test_ratio)
    print(f'Dataset unique molecules: {total_molecules}')
    print(f'Target test molecules: {test_target_mols}')

    # Build test with component-aware cascading + min_frac; boundary -> validation
    test_mols = set()
    blacklist_mols = set()
    boundary_mols = set()     # boundary molecules (go to validation directly)
    max_test_size = int(test_target_mols * 2)  # cap per original design

    i = 0
    bl_count = 0
    test_count = 0

    while len(test_mols) < test_target_mols and i < len(mol_ids_sorted):
        if i % 100 == 0:
            print(f'Test size: {test_count} | Black list: {bl_count} | Boundary (to val): {len(boundary_mols)}',
                  flush=True)

        mol = mol_ids_sorted[i]
        i += 1
        if mol in test_mols or mol in blacklist_mols or mol in boundary_mols:
            continue  # If already fixed for test, train, or val

        remaining_allowed = max_test_size - len(test_mols)
        cascade, accepted, killed_frontier, unique_components_count, total_pairs, depth_reached = cascade_closure_with_depth_components(
            mol, remaining_allowed, test_mols, blacklist_mols,
            component_to_molecules, molecule_to_components, molecule_to_proteins,
            args.min_frac, excluded_mols=boundary_mols
        )

        if not accepted:
            blacklist_mols.update(cascade)
            bl_count += len(cascade)
            continue

        # Accept: test gets the accepted part; validation gets killed_frontier
        test_mols.update(cascade)
        boundary_mols.update(killed_frontier)
        # If any boundary molecules were previously blacklisted (train), migrate them to validation
        if killed_frontier & blacklist_mols:
            blacklist_mols.difference_update(killed_frontier)
        test_count += len(cascade)

        frac_test = (len(cascade) / (len(cascade) + len(killed_frontier))) if (len(cascade) + len(killed_frontier)) > 0 else 1.0
        print(f"Accepted cascade: test+={len(cascade)}, boundary->val={len(killed_frontier)}; "
              f"comps={unique_components_count}, pairs={total_pairs}, depth={depth_reached}, frac_test={frac_test:.3f}")

    print(f"Final test molecules: {len(test_mols)} (target {test_target_mols})")
    print(f"Boundary molecules diverted to validation: {len(boundary_mols)}")

    # Validation target based on test TARGET (not actual), then boundary top-up
    val_target_mols = int((total_molecules - test_target_mols) * args.val_size)
    print(f"Validation target molecules (before boundary): {val_target_mols}")

    # Remaining molecules after test and boundary
    remaining_mols = set(molecule_to_proteins.keys()) - test_mols - boundary_mols

    # Start validation with boundary molecules, then top up if needed
    val_mols = set(boundary_mols)
    if len(val_mols) < val_target_mols:
        remaining_list = list(remaining_mols)
        random.shuffle(remaining_list)
        need = val_target_mols - len(val_mols)
        val_mols.update(remaining_list[:need])
        remaining_mols.difference_update(val_mols)
        print(f"Topped up validation with {need} molecules to reach target.")
    else:
        print("Boundary ≥ validation target; no top-up added.")

    # Train = everything else
    train_mols = remaining_mols

    print(f"Train molecules: {len(train_mols)} | Validation molecules: {len(val_mols)} | Test molecules: {len(test_mols)}")

    # Generate output files
    if args.out_suffix is not None:
        suffix = f'_{args.out_suffix}'
    else:
        suffix=''
    base_name = args.dataset_json.replace('.json', f'_seed{args.seed}_optimal{suffix}')
    train_json = base_name + '_train.json'
    val_json = base_name + '_val.json'
    test_json = base_name + '_test.json'
    cascade_killed_json = base_name + '_cascade_killed.json'  # boundary audit

    print(f'Generating splits: {train_json}, {val_json}, {test_json}')
    if boundary_mols:
        print(f'Also generating boundary list (cascade_killed): {cascade_killed_json}')

    # Accumulate data + stats
    def stats_init():
        return {"pairs": 0, "uniq_mols": set(), "uniq_prots": set(), "mol2prots": defaultdict(set)}

    stats_train = stats_init()
    stats_val   = stats_init()
    stats_test  = stats_init()

    train_data = []
    val_data = []
    test_data = []
    cascade_killed_data = []

    for example in raw_dataset:
        mol = example['translation']['SELFIES'] if selfies else example['translation']['Canonical Smiles']
        prot = example["translation"]["Protein sequence"]

        if mol in test_mols:
            test_data.append(example)
            s = stats_test
        elif mol in val_mols:
            val_data.append(example)
            s = stats_val
        else:
            train_data.append(example)
            s = stats_train

        s["pairs"] += 1
        s["uniq_mols"].add(mol)
        s["uniq_prots"].add(prot)
        s["mol2prots"][mol].add(prot)

        if mol in boundary_mols:
            cascade_killed_data.append(example)

    # Write output files
    with open(train_json, 'w', encoding='utf-8') as f:
        for ex in train_data:
            json.dump(ex, f); f.write('\n')

    with open(val_json, 'w', encoding='utf-8') as f:
        for ex in val_data:
            json.dump(ex, f); f.write('\n')

    with open(test_json, 'w', encoding='utf-8') as f:
        for ex in test_data:
            json.dump(ex, f); f.write('\n')

    if cascade_killed_data:
        with open(cascade_killed_json, 'w', encoding='utf-8') as f:
            for ex in cascade_killed_data:
                json.dump(ex, f); f.write('\n')

    # ---- Final split stats ----
    def print_stats(title, s):
        avg_prot_per_mol = (np.mean([len(v) for v in s["mol2prots"].values()]) if s["mol2prots"] else 0.0)
        print(f"{title}:")
        print(f"  Unique molecules: {len(s['uniq_mols']):,}")
        print(f"  Unique proteins : {len(s['uniq_prots']):,}")
        print(f"  Total pairs     : {s['pairs']:,}")
        print(f"  Avg proteins/molecule: {avg_prot_per_mol:.3f}")

    print("\n=== Final split stats ===")
    print_stats("Train", stats_train)
    print_stats("Validation", stats_val)
    print_stats("Test", stats_test)

if __name__ == "__main__":
    main()
