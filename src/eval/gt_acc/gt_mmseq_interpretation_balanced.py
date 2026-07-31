"""
Balanced version of gt_mmseq_interpretation.py.

For each input target, computes the best GT identity per generation (0 if no match),
then averages across all generations for that target. Final statistics are the mean
and std of these per-target averages across all input targets.
Accuracy (max best-match identity across generations) is kept unchanged.
"""

import argparse
import numpy as np
import pandas as pd
import os
import json
import pickle
from collections import defaultdict

def _sf(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def read_fasta_sequences(fasta_path):
    """Read FASTA file and return dictionary of {header: sequence}"""
    sequences = {}
    current_header = None
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header is not None:
                    sequences[current_header] = ''.join(current_seq)
                current_header = line[1:]  # Remove '>'
                current_seq = []
            else:
                current_seq.append(line)

        # Don't forget the last sequence
        if current_header is not None:
            sequences[current_header] = ''.join(current_seq)

    return sequences


def _parse_identity(value):
    val = float(value)
    return val


def analyze_mmseqs_output(mmseqs_file, n_inference_seqs):
    """
    Per generation, keep the best identity (and best effective identity) across
    all GT hits.  Generations with no mmseqs hit receive 0.

    Returns:
      best_match, mean_identity, std_identity,
      best_eff,   mean_eff,      std_eff
    """
    per_query_best_id = {}
    per_query_best_eff = {}

    if os.path.getsize(mmseqs_file) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    with open(mmseqs_file, 'r') as infile:
        for line in infile:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue

            # Expected: query,target,fident,alnlen,qlen,tlen
            query_id = int(parts[0])
            identity = _parse_identity(parts[2])

            # Try to get coverage; if missing, use 0.0
            if len(parts) >= 6:
                alnlen = _sf(parts[3], 0.0)
                tlen   = _sf(parts[5], 0.0)
                t_cov  = (alnlen / tlen) if tlen > 0 else 0.0
            elif len(parts) >= 5:
                alnlen = _sf(parts[3], 0.0)
                tlen   = _sf(parts[4], 0.0)
                t_cov  = (alnlen / tlen) if tlen > 0 else 0.0
            else:
                t_cov = 0.0

            eff = identity * t_cov

            # Track best per query
            if (query_id not in per_query_best_id) or (identity > per_query_best_id[query_id]):
                per_query_best_id[query_id] = identity
            if (query_id not in per_query_best_eff) or (eff > per_query_best_eff[query_id]):
                per_query_best_eff[query_id] = eff

    # One value per generation: best identity (0 if no hit)
    best_ids  = [per_query_best_id.get(i, 0.0) for i in range(n_inference_seqs)]
    best_effs = [per_query_best_eff.get(i, 0.0) for i in range(n_inference_seqs)]

    best_match    = max(best_ids)
    mean_identity = float(np.mean(best_ids))
    std_identity  = float(np.std(best_ids))

    best_eff = max(best_effs)
    mean_eff = float(np.mean(best_effs))
    std_eff  = float(np.std(best_effs))

    return best_match, mean_identity, std_identity, best_eff, mean_eff, std_eff


def get_detailed_mmseqs_results_best_match(mmseqs_file, inference_sequences, gt_sequences):
    """
    Pick the best match per query by highest effective identity:
        effective_identity = identity * target_coverage
    where target_coverage = alnlen / tlen (requires --format-output to include alnlen,tlen).
    """
    detailed_results = {}

    if os.path.getsize(mmseqs_file) == 0:
        # No matches at all in this file
        for query_id in inference_sequences:
            detailed_results[query_id] = {
                'seq': inference_sequences[query_id],
                'closest_gt': '',
                'mmseqs_id': 'NaN',
                'target_coverage': 0.0,
                'effective_identity': 0.0,
            }
        return detailed_results

    # Store best (by effective_identity) for each query
    best_matches = {}  # query_id -> dict(target_name, identity, t_cov, eff)

    with open(mmseqs_file, 'r') as infile:
        for line in infile:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue

            query_id    = int(parts[0])
            target_name = parts[1]
            identity = _parse_identity(parts[2])

            if len(parts) >= 6:
                alnlen = _sf(parts[3], 0.0)
                tlen   = _sf(parts[5], 0.0)
                t_cov  = (alnlen / tlen) if tlen > 0 else 0.0
            elif len(parts) >= 5:
                alnlen = _sf(parts[3], 0.0)
                tlen   = _sf(parts[4], 0.0)
                t_cov  = (alnlen / tlen) if tlen > 0 else 0.0
            else:
                t_cov = 0.0

            eff = identity * t_cov

            prev = best_matches.get(query_id)
            if prev is None:
                best_matches[query_id] = {
                    'target_name': target_name,
                    'identity': identity,
                    'target_coverage': t_cov,
                    'effective_identity': eff,
                }
            else:
                if (eff > prev['effective_identity'] or
                    (eff == prev['effective_identity'] and identity > prev['identity']) or
                    (eff == prev['effective_identity'] and identity == prev['identity'] and t_cov > prev['target_coverage'])):
                    best_matches[query_id] = {
                        'target_name': target_name,
                        'identity': identity,
                        'target_coverage': t_cov,
                        'effective_identity': eff,
                    }

    # Build final detailed results
    for query_id_str, sequence in inference_sequences.items():
        qid = int(query_id_str)
        if qid in best_matches:
            bm = best_matches[qid]
            closest_gt = gt_sequences.get(bm['target_name'], '')
            identity   = bm['identity']
            t_cov      = bm['target_coverage']
            eff        = bm['effective_identity']
        else:
            closest_gt = ''
            identity   = 0.0
            t_cov      = 0.0
            eff        = 0.0

        detailed_results[query_id_str] = {
            'seq': sequence,
            'closest_gt': closest_gt,
            'mmseqs_id': identity,
            'target_coverage': t_cov,
            'effective_identity': eff,
        }

    return detailed_results



def get_all_matches_mmseqs_results(mmseqs_file, inference_sequences, gt_sequences):
    """Get all matches for each query sequence, including effective identity.

    effective_identity = identity * target_coverage
    where target_coverage = alnlen / tlen (requires --format-output to include alnlen,tlen).
    """
    all_matches_results = {}

    if os.path.getsize(mmseqs_file) == 0:
        for query_id in inference_sequences:
            all_matches_results[query_id] = {
                'seq': inference_sequences[query_id],
                'matching_gt': []
            }
        return all_matches_results

    all_matches = defaultdict(list)

    with open(mmseqs_file, 'r') as infile:
        for line in infile:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue

            query_id = int(parts[0])
            target_name = parts[1]
            identity = _parse_identity(parts[2])

            if len(parts) >= 6:
                alnlen = _sf(parts[3], 0.0)
                tlen   = _sf(parts[5], 0.0)
                t_cov  = (alnlen / tlen) if tlen > 0 else 0.0
            elif len(parts) >= 5:
                alnlen = _sf(parts[3], 0.0)
                tlen   = _sf(parts[4], 0.0)
                t_cov  = (alnlen / tlen) if tlen > 0 else 0.0
            else:
                t_cov = 0.0

            eff = identity * t_cov

            all_matches[query_id].append({
                'target_name': target_name,
                'identity': identity,
                'target_coverage': t_cov,
                'effective_identity': eff,
            })

    for query_id_str, sequence in inference_sequences.items():
        query_id = int(query_id_str)

        matching_gt = []
        if query_id in all_matches:
            for match in all_matches[query_id]:
                target_name = match['target_name']
                gt_sequence = gt_sequences.get(target_name, '')

                matching_gt.append({
                    'sequence': gt_sequence,
                    'mmseqs_id': match['identity'],
                    'effective_identity': match['effective_identity'],
                    'target_coverage': match['target_coverage'],
                })

        all_matches_results[query_id_str] = {
            'seq': sequence,
            'matching_gt': matching_gt
        }

    return all_matches_results


def main():
    parser = argparse.ArgumentParser(description="Analyze MMseqs output and compute sequence identity statistics (balanced).")
    parser.add_argument('--mmseqs_path', type=str, required=True, help="Path to the MMseqs output file")
    parser.add_argument('--n_inference_seqs', default=25, help="Number of sequences generated per molecule")
    parser.add_argument('--inference_folder', type=str, required=True,
                        help="Path to folder containing inference FASTA files")
    parser.add_argument('--gt_fasta_folder', type=str, required=True,
                        help="Path to folder containing GT fastas and folder_names.json (output_folder_gt in gt_inference_mmseq)")
    parser.add_argument('--molecule_metadata_file', type=str,
                        help="Path to molecule_input_metadata.json (optional, will auto-find if not provided)")

    args = parser.parse_args()

    # Load metadata files
    if args.molecule_metadata_file:
        molecule_metadata_path = args.molecule_metadata_file
    else:
        metadata_files = [f for f in os.listdir(args.inference_folder) if 'molecule_input_metadata' in f]
        if not metadata_files:
            raise FileNotFoundError("molecule_input_metadata.json not found in inference folder")
        molecule_metadata_path = os.path.join(args.inference_folder, metadata_files[0])

    print(f"Loading molecule metadata from {molecule_metadata_path}")
    with open(molecule_metadata_path, 'r') as f:
        molecule_metadata = json.load(f)

    folder_names_path = os.path.join(args.gt_fasta_folder, 'folder_names.json')
    print(f"Loading folder names from {folder_names_path}")
    with open(folder_names_path, 'r') as f:
        folder_names = json.load(f)

    mmseqs_files = os.listdir(args.mmseqs_path)
    mmseqs_files = [f for f in mmseqs_files if 'mmseqs' in f]

    best_match_vector = []
    mean_vector = []

    best_eff_vector = []
    mean_eff_vector = []

    detailed_results_json = {}
    per_input_stats = {}

    print(f'Calculating for {args.n_inference_seqs} sequences per molecule')

    for filename in mmseqs_files:
        if 'mmseqs' in filename:
            file_path = os.path.join(args.mmseqs_path, filename)

            safe_name = filename.replace('mmseqs_', '')

            reaction = None
            for smiles, sname in folder_names.items():
                if sname == safe_name:
                    reaction = smiles
                    break

            if reaction is None:
                print(f"Warning: Could not find reaction for safe_name {safe_name}")
                continue

            fasta_filename = None
            for fasta_path, smiles in molecule_metadata.items():
                if smiles == reaction:
                    fasta_filename = os.path.basename(fasta_path)
                    break

            if fasta_filename is None:
                print(f"Warning: Could not find FASTA file for reaction {reaction}")
                continue

            print(f"Processing {filename} -> {safe_name} -> {reaction} -> {fasta_filename}")

            inference_fasta_path = os.path.join(args.inference_folder, fasta_filename)
            if not os.path.exists(inference_fasta_path):
                print(f"Warning: Inference FASTA not found: {inference_fasta_path}")
                continue

            inference_sequences = read_fasta_sequences(inference_fasta_path)

            gt_fasta_path = os.path.join(args.gt_fasta_folder, safe_name, f"{safe_name}.fasta")
            if not os.path.exists(gt_fasta_path):
                print(f"Warning: GT FASTA not found: {gt_fasta_path}")
                continue

            gt_sequences = read_fasta_sequences(gt_fasta_path)

            n_seqs = len(inference_sequences)
            best_match, mean, _std, best_eff, mean_eff, _std_eff = analyze_mmseqs_output(file_path, n_seqs)
            best_match_vector.append(best_match)
            mean_vector.append(mean)

            best_eff_vector.append(best_eff)
            mean_eff_vector.append(mean_eff)

            per_input_stats[reaction] = {
                'accuracy': best_match,
                'mean_identity': mean,
                'effective_id_accuracy': best_eff,
                'effective_id_mean': mean_eff,
            }

            detailed_results = get_detailed_mmseqs_results_best_match(file_path, inference_sequences, gt_sequences)

            best_gt_seq = ''
            best_id_seen = -1.0
            for _qid, res in detailed_results.items():
                v = res['mmseqs_id']
                id_val = float(v) if v != 'NaN' else 0.0
                if id_val > best_id_seen:
                    best_id_seen = id_val
                    best_gt_seq = res['closest_gt']
            per_input_stats[reaction]['best_match_gt_seq'] = best_gt_seq

            fasta_results = {}
            for query_id, result in detailed_results.items():
                pdb_filename = f"top_{fasta_filename}_{query_id}.pdb"
                fasta_results[pdb_filename] = result

            detailed_results_json[fasta_filename] = fasta_results

    # Detailed results dictionary for all matches
    all_matches_results_json = {}

    print("\nProcessing all matches...")
    for filename in mmseqs_files:
        if 'mmseqs' in filename:
            file_path = os.path.join(args.mmseqs_path, filename)

            safe_name = filename.replace('mmseqs_', '')

            reaction = None
            for smiles, sname in folder_names.items():
                if sname == safe_name:
                    reaction = smiles
                    break

            if reaction is None:
                continue

            fasta_filename = None
            for fasta_path, smiles in molecule_metadata.items():
                if smiles == reaction:
                    fasta_filename = os.path.basename(fasta_path)
                    break

            if fasta_filename is None:
                continue

            inference_fasta_path = os.path.join(args.inference_folder, fasta_filename)
            if not os.path.exists(inference_fasta_path):
                continue

            inference_sequences = read_fasta_sequences(inference_fasta_path)

            gt_fasta_path = os.path.join(args.gt_fasta_folder, safe_name, f"{safe_name}.fasta")
            if not os.path.exists(gt_fasta_path):
                continue

            gt_sequences = read_fasta_sequences(gt_fasta_path)

            all_matches_results = get_all_matches_mmseqs_results(file_path, inference_sequences, gt_sequences)

            fasta_all_matches = {}
            for query_id, result in all_matches_results.items():
                pdb_filename = f"top_{fasta_filename}_{query_id}.pdb"
                fasta_all_matches[pdb_filename] = result

            all_matches_results_json[fasta_filename] = fasta_all_matches

    # --- Aggregate across input targets ---
    print("Best match vector:", best_match_vector)
    print("Mean vector:", mean_vector)

    accuracy      = float(np.mean(best_match_vector))
    mean_identity = float(np.mean(mean_vector))
    std_identity  = float(np.std(mean_vector))

    effective_id_accuracy = float(np.mean(best_eff_vector))
    effective_id_mean     = float(np.mean(mean_eff_vector))
    effective_id_std      = float(np.std(mean_eff_vector))

    output_file = f"{args.mmseqs_path}/stats_balanced.csv"
    df = pd.DataFrame({
        'Accuracy': [accuracy],
        'Mean Sequence Identity': [mean_identity],
        'STD Sequence Identity': [std_identity],
        'effective_id_accuracy': [effective_id_accuracy],
        'effective_id_mean': [effective_id_mean],
        'effective_id_std': [effective_id_std],
    })
    df.to_csv(output_file, index=False, float_format='%.4f')
    print(f"Stats saved to {output_file}")

    print("\nSummary statistics:")
    print(df.to_string(index=False))

    detailed_output_file = f"{args.mmseqs_path}/detailed_results.json"
    with open(detailed_output_file, 'w') as f:
        json.dump(detailed_results_json, f, indent=2)
    print(f"Detailed results saved to {detailed_output_file}")

    per_input_output_file = f"{args.mmseqs_path}/per_input_stats.pkl"
    with open(per_input_output_file, 'wb') as f:
        pickle.dump(per_input_stats, f)
    print(f"Per-input stats saved to {per_input_output_file}")

    all_matches_output_file = f"{args.mmseqs_path}/all_matches_results.json"
    with open(all_matches_output_file, 'w') as f:
        json.dump(all_matches_results_json, f, indent=2)
    print(f"All matches results saved to {all_matches_output_file}")


if __name__ == "__main__":
    main()
