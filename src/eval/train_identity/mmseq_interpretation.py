import argparse
import numpy as np
import pandas as pd
import os
import pickle
import json
from collections import defaultdict
import gzip


def _normalize_sequence(seq):
    if not isinstance(seq, str):
        return None
    stripped = seq.strip()
    if not stripped:
        return None
    return stripped.upper()


def _load_ground_truth_from_metadata(metadata_path):
    ground_truth = {}
    if not metadata_path:
        return ground_truth

    try:
        with open(metadata_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: metadata file {metadata_path} not found.")
        return ground_truth
    except Exception as e:
        print(f"Warning: could not read metadata file {metadata_path}: {e}")
        return ground_truth

    for fasta_path, meta in data.items():
        if not isinstance(meta, dict):
            continue
        seqs = meta.get('GT') or meta.get('Protein sequence') or meta.get('Sequence')
        if not seqs:
            continue
        if isinstance(seqs, str):
            seqs = [seqs]
        normalized = []
        for seq in seqs:
            norm = _normalize_sequence(seq)
            if norm:
                normalized.append(norm)
        if normalized:
            fasta_name = os.path.basename(fasta_path) if isinstance(fasta_path, str) else fasta_path
            if fasta_name:
                ground_truth[fasta_name] = sorted(set(normalized))

    return ground_truth


def _candidate_better(new_candidate, current_candidate):
    if current_candidate is None:
        return True
    if new_candidate[0] > current_candidate[0]:
        return True
    if new_candidate[0] == current_candidate[0] and new_candidate[1] > current_candidate[1]:
        return True
    if (new_candidate[0] == current_candidate[0] and
            new_candidate[1] == current_candidate[1] and
            new_candidate[2] > current_candidate[2]):
        return True
    return False


def _find_best_match_for_gt(matches, training_correspondence, gt_key):
    if not training_correspondence:
        return None
    best = None
    for match in matches:
        target_id = match['target_id']
        if target_id not in training_correspondence:
            continue
        seq = training_correspondence[target_id].get('sequence')
        if not isinstance(seq, str):
            continue
        seq_clean = seq.strip()
        key = _normalize_sequence(seq_clean)
        if key != gt_key:
            continue
        candidate = (match['effective_identity'], match['identity'], match['target_coverage'], seq_clean)
        if _candidate_better(candidate, best):
            best = candidate
    return best


def save_pickle(obj, path, compress=False):
    if compress:
        with gzip.open(path, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(path, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def count_fasta_sequences(fasta_path):
    """Count the number of sequences in a FASTA file."""
    count = 0
    with open(fasta_path, 'r') as fasta_file:
        for line in fasta_file:
            if line.startswith(">"):
                count += 1
    return count


def load_metadata(metadata_path):
    """Load the metadata mapping from pickle file."""
    try:
        with open(metadata_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Metadata file {metadata_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None


def load_sequence_correspondence(correspondence_path):
    """Load the sequence correspondence mapping from JSON file."""
    try:
        with open(correspondence_path, 'r') as f:
            correspondence = json.load(f)
            # Convert string keys to integers
            return {int(k): v for k, v in correspondence.items()}
    except FileNotFoundError:
        print(f"Sequence correspondence file {correspondence_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading sequence correspondence: {e}")
        return None


def read_target_sequences(mmseqs_path):
    """Read target sequences from MMseqs database if available."""
    # This is a placeholder - you might need to adjust based on your MMseqs setup
    # For now, we'll return None and use target IDs instead
    return None


def _parse_identity_auto(x: str) -> float:
    val = float(x)
    return val


def analyze_mmseqs_output(mmseqs_path, query_fasta, metadata_path=None, training_correspondence_path=None,
                          input_metadata_file=None):
    seq_identity = {}            # best identity per query (for reporting)
    seq_eff_identity = {}        # best effective identity per query
    alignment_lengths = []       # all alignment lengths (for info)
    best_targets = {}            # best-by-effective-identity target per query
    best_match_meta = {}         # store alnlen,tlen for the best-by-effective hit
    all_matches_by_query = defaultdict(list)  # collect all hits per query for top-match export

    # Get total number of query sequences
    total_queries = count_fasta_sequences(query_fasta)

    # Load metadata mapping if provided
    id_mapping = None
    if metadata_path and os.path.exists(metadata_path):
        id_mapping = load_metadata(metadata_path)
        print(f"Loaded metadata mapping with {len(id_mapping)} entries")
    else:
        print("No metadata file provided or found. Using numeric IDs only.")

    # Load sequence correspondence for query sequences
    correspondence_path = query_fasta.replace('.fasta', '_sequence_correspondence.json')
    sequence_correspondence = load_sequence_correspondence(correspondence_path)
    if sequence_correspondence:
        print(f"Loaded query sequence correspondence with {len(sequence_correspondence)} entries")

    # Load sequence correspondence for training sequences (from JSON file)
    training_correspondence = None
    if training_correspondence_path and os.path.exists(training_correspondence_path):
        training_correspondence = load_sequence_correspondence(training_correspondence_path)
        if training_correspondence:
            print(f"Loaded training sequence correspondence with {len(training_correspondence)} entries from {training_correspondence_path}")
    else:
        if training_correspondence_path is None:
            print("Warning: No training sequence correspondence file provided. Will use target IDs instead of sequences.")
        else:
            print(f"Warning: Training sequence correspondence file {training_correspondence_path} not found. Will use target IDs instead of sequences.")

    ground_truth_by_fasta = _load_ground_truth_from_metadata(input_metadata_file)
    if input_metadata_file:
        if ground_truth_by_fasta:
            print(f"Loaded ground-truth proteins for {len(ground_truth_by_fasta)} FASTA files from {input_metadata_file}")
        else:
            print(f"Warning: No ground-truth proteins found in {input_metadata_file}")
    warned_missing_training_for_gt = False

    # Read MMseqs output file
    mmseqs_output_path = os.path.join(mmseqs_path, 'mmseqs_output')
    with open(mmseqs_output_path, 'r') as infile:
        for line in infile:
            parts = line.strip().split('\t')
            # Expected: query, target, fident, alnlen, qlen, tlen
            if len(parts) < 6:
                continue

            query_id = int(parts[0])
            target_id = int(parts[1])
            identity = _parse_identity_auto(parts[2])           # → [0,1]
            alnlen = float(parts[3])
            # qlen = float(parts[4])  # not used for effective identity
            tlen = float(parts[5]) if float(parts[5]) > 0 else 0.0
            t_cov = (alnlen / tlen) if tlen > 0 else 0.0
            eff = identity * t_cov

            alignment_lengths.append(alnlen)

            # Track hit for later top-match export
            all_matches_by_query[query_id].append({
                'target_id': target_id,
                'identity': identity,
                'target_coverage': t_cov,
                'effective_identity': eff,
            })

            curr_best_eff = seq_eff_identity.get(query_id, -1.0)
            curr_best_id = seq_identity.get(query_id, -1.0)
            curr_best_cov = best_match_meta.get(query_id, {}).get("t_cov", -1.0)

            if (eff > curr_best_eff or
                    (eff == curr_best_eff and identity > curr_best_id) or
                    (eff == curr_best_eff and identity == curr_best_id and t_cov > curr_best_cov)):
                seq_eff_identity[query_id] = eff
                seq_identity[query_id] = identity  # identity of the eff-best hit
                best_targets[query_id] = target_id
                best_match_meta[query_id] = {"alnlen": alnlen, "tlen": tlen, "t_cov": t_cov, "eff": eff}

    # 1) Do we actually map target ids → sequences?
    if training_correspondence is not None:
        all_tids = set(m['target_id'] for hits in all_matches_by_query.values() for m in hits)
        covered = sum(1 for t in all_tids if t in training_correspondence)
        if all_tids:
            print(f"[diag] target-id coverage in training_correspondence: "
                f"{covered}/{len(all_tids)} ({100.0*covered/len(all_tids):.2f}%)")
        else:
            print("[diag] No targets found in mmseqs_output?")
    else:
        print("[diag] training_correspondence not loaded")

    # 2) Do query FASTA names match GT metadata keys?
    if sequence_correspondence and ground_truth_by_fasta:
        some = next(iter(sequence_correspondence.values()))
        print(f"[diag] example query fasta name: {some['fasta']}")
        print(f"[diag] GT metadata example keys: {list(ground_truth_by_fasta.keys())[:5]}")

    
    # Ensure all query sequences are accounted for (missing ones get 0)
    rest = total_queries - len(seq_identity.keys())
    print(f'{len(seq_identity.keys())}/{total_queries} output proteins were found in the database')

    # Identity stats (as you had)
    id_vals = list(seq_identity.values())
    id_vals_full = id_vals + [0.0] * rest
    id_vec = np.array(id_vals, dtype=float)
    id_vec_full = np.array(id_vals_full, dtype=float)
    mean_identity = np.mean(id_vec) if id_vec.size else 0.0
    std_identity = np.std(id_vec) if id_vec.size else 0.0
    mean_all = np.mean(id_vec_full) if id_vec_full.size else 0.0
    std_all = np.std(id_vec_full) if id_vec_full.size else 0.0
    non_perf = id_vec_full[id_vec_full < 1.0]
    novelty = (len(non_perf) / len(id_vec_full) * 100) if id_vec_full.size else 0.0
    mean_non_perfect = float(np.mean(non_perf)) if non_perf.size else 0.0
    std_non_perfect = float(np.std(non_perf)) if non_perf.size else 0.0
    mean_align_len = np.mean(alignment_lengths) if alignment_lengths else 0.0

    # Effective-identity stats (mirroring the same computations)
    eff_vals = list(seq_eff_identity.values())
    eff_vals_full = eff_vals + [0.0] * rest
    eff_vec = np.array(eff_vals, dtype=float)
    eff_vec_full = np.array(eff_vals_full, dtype=float)
    eff_mean = np.mean(eff_vec) if eff_vec.size else 0.0
    eff_std = np.std(eff_vec) if eff_vec.size else 0.0
    eff_mean_all = np.mean(eff_vec_full) if eff_vec_full.size else 0.0
    eff_std_all = np.std(eff_vec_full) if eff_vec_full.size else 0.0
    eff_non_perf = eff_vec_full[eff_vec_full < 1.0]
    eff_novelty = (len(eff_non_perf) / len(eff_vec_full) * 100) if eff_vec_full.size else 0.0
    eff_mean_non_perf = float(np.mean(eff_non_perf)) if eff_non_perf.size else 0.0
    eff_std_non_perf = float(np.std(eff_non_perf)) if eff_non_perf.size else 0.0

    # Save main results → two rows: identity, effective_identity
    output_file = f"{mmseqs_path}/novelty.csv"
    df = pd.DataFrame([
        {
            'Metric': 'identity',
            'Mean Sequence Identity (matches)': mean_identity,
            'Std Sequence Identity': std_identity,
            'Non-Perfect Matches (%)': novelty,
            'Mean all': mean_all,
            'Std all': std_all,
            'Mean Non-Perfect Identity': mean_non_perfect,
            'Std Non-Perfect Identity': std_non_perfect,
            'Non-Matched sequences': f"{rest}/{total_queries}",
            'Mean Alignment Length': mean_align_len
        },
        {
            'Metric': 'effective_identity',
            'Mean Sequence Identity (matches)': eff_mean,
            'Std Sequence Identity': eff_std,
            'Non-Perfect Matches (%)': eff_novelty,
            'Mean all': eff_mean_all,
            'Std all': eff_std_all,
            'Mean Non-Perfect Identity': eff_mean_non_perf,
            'Std Non-Perfect Identity': eff_std_non_perf,
            'Non-Matched sequences': f"{rest}/{total_queries}",
            'Mean Alignment Length': mean_align_len
        }
    ])
    df.to_csv(output_file, index=False, float_format='%.4f')

    # Save sequence-specific results with original identifiers (identity only, unchanged)
    sequence_results_file = f"{mmseqs_path}/sequence_identities.csv"
    sequence_data = []
    for numeric_id in range(0, total_queries):
        original_id = str(numeric_id)
        if id_mapping and numeric_id in id_mapping:
            original_id = id_mapping[numeric_id]
        identity = seq_identity.get(numeric_id, 0.0)
        sequence_data.append({
            'Numeric_ID': numeric_id,
            'Original_Identifier': original_id,
            'Sequence_Identity': identity
        })
    pd.DataFrame(sequence_data).to_csv(sequence_results_file, index=False, float_format='%.4f')

    # Create detailed results for PDB files (now including coverage + effective_identity)
    detailed_results = {}
    all_matches_results = {}
    gt_total_sequences = 0
    gt_present_initial = 0
    gt_appended_after_search = 0
    gt_missing_after_search = 0
    gt_no_candidate_found = 0
    prev_fasta_name = None
    pdb_idx = 0
    if sequence_correspondence:
        for numeric_id in range(0, total_queries):
            if numeric_id in sequence_correspondence:
                fasta_name = sequence_correspondence[numeric_id]['fasta']
                if fasta_name != prev_fasta_name:
                    pdb_idx = 0
                    prev_fasta_name = fasta_name
                sequence_string = sequence_correspondence[numeric_id]['sequence']
                pdb_file = f"{fasta_name}_{pdb_idx}.pdb"

                # MMseqs info for the best-by-identity hit
                target_id = best_targets.get(numeric_id, None)
                mmseqs_identity = seq_identity.get(numeric_id, 0.0)
                if numeric_id in best_match_meta:
                    t_cov = best_match_meta[numeric_id]['t_cov']
                    eff_id = best_match_meta[numeric_id]['eff']
                else:
                    t_cov = 0.0
                    eff_id = 0.0

                # Training sequence string if known
                if target_id is not None and training_correspondence and target_id in training_correspondence:
                    closest_train_seq = training_correspondence[target_id]['sequence']
                elif target_id is not None:
                    closest_train_seq = f"target_id_{target_id}"
                else:
                    closest_train_seq = "No match"

                detailed_results[pdb_file] = {
                    'seq': sequence_string,
                    'closest_train_seq': closest_train_seq,
                    'mmseqs_id': mmseqs_identity,
                    'target_coverage': t_cov,
                    'effective_identity': eff_id
                }
                
                # Build matching GT list for top effective-identity hits
                MAX_MATCHES_TOP = 30        # final cap in output
                TIER2_THRESHOLD = 2 * MAX_MATCHES_TOP  # when we tighten tol to 0.0
                INITIAL_TOL = 0.01
                TIGHT_TOL = 0.001
                matches_for_query = all_matches_by_query.get(numeric_id, [])
                if matches_for_query:
                    best_eff = max(m['effective_identity'] for m in matches_for_query)

                    # Progressive tolerance tightening
                    tol = INITIAL_TOL
                    candidates_by_seq = {}  # map normalized sequence → tuple (eff, id, cov, seq)
                    count = 0

                    for m in matches_for_query:
                        eff = m['effective_identity']
                        if eff >= best_eff - tol:
                            target_id = m['target_id']
                            seq_raw = None
                            if training_correspondence and target_id in training_correspondence:
                                seq_raw = training_correspondence[target_id].get('sequence')
                            seq_clean = seq_raw.strip() if isinstance(seq_raw, str) else None
                            if not seq_clean:
                                seq_clean = f"target_id_{target_id}"
                            key = _normalize_sequence(seq_clean) or f"TARGET_ID_{target_id}"

                            candidate = (eff, m['identity'], m['target_coverage'], seq_clean)
                            prev = candidates_by_seq.get(key)
                            if _candidate_better(candidate, prev):
                                if prev is None:
                                    count += 1
                                    # Tighten tolerance as we accrue many unique candidates
                                    if count == MAX_MATCHES_TOP:
                                        tol = TIGHT_TOL
                                    elif count == TIER2_THRESHOLD:
                                        tol = 0.0
                                candidates_by_seq[key] = candidate

                    # Sort by effective identity desc, then identity desc, then coverage desc
                    candidates_all = list(candidates_by_seq.values())
                    candidates_all.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
                    top_candidates = candidates_all[:MAX_MATCHES_TOP]

                    final_candidates = list(top_candidates)
                    final_keys = {(_normalize_sequence(c[3]) or c[3]) for c in final_candidates}

                    gt_sequences = ground_truth_by_fasta.get(fasta_name, [])
                    if gt_sequences:
                        if not training_correspondence and not warned_missing_training_for_gt:
                            print("Warning: Ground-truth proteins provided but training correspondence unavailable; "
                                  "cannot match against GT sequences.")
                            warned_missing_training_for_gt = True
                        elif training_correspondence:
                            for gt_seq in gt_sequences:
                                gt_key = _normalize_sequence(gt_seq)
                                if not gt_key:
                                    continue
                                gt_total_sequences += 1
                                already_present = gt_key in final_keys
                                candidate_found = False
                                candidate = None
                                if already_present:
                                    gt_present_initial += 1
                                else:
                                    candidate = candidates_by_seq.get(gt_key)  # is gt_key in the matches list?
                                    if candidate is None:
                                        candidate = _find_best_match_for_gt(matches_for_query, training_correspondence, gt_key)
                                    if candidate is not None:
                                        candidate_found = True
                                    if candidate and gt_key not in final_keys:
                                        final_candidates.append(candidate)
                                        final_keys.add(gt_key)
                                        gt_appended_after_search += 1
                                    if gt_key not in final_keys:
                                        gt_missing_after_search += 1
                                        if not candidate_found:
                                            gt_no_candidate_found += 1
                            if len(final_candidates) > len(top_candidates):
                                final_candidates.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)

                    # Convert to compact columnar structure with arrays for numeric fields
                    if final_candidates:
                        effs = np.array([c[0] for c in final_candidates], dtype=np.float32)
                        ids_ = np.array([c[1] for c in final_candidates], dtype=np.float32)
                        covs = np.array([c[2] for c in final_candidates], dtype=np.float32)
                        seqs = [c[3] for c in final_candidates]

                        all_matches_results[pdb_file] = {
                            'seq': sequence_string,
                            'matching_gt': {
                                'sequence': seqs,
                                'mmseqs_id': ids_,
                                'target_coverage': covs,
                                'effective_identity': effs,
                            }
                        }
                    else:
                        all_matches_results[pdb_file] = {
                            'seq': sequence_string,
                            'matching_gt': {
                                'sequence': [],
                                'mmseqs_id': np.array([], dtype=np.float32),
                                'target_coverage': np.array([], dtype=np.float32),
                                'effective_identity': np.array([], dtype=np.float32),
                            }
                        }
                else:
                    all_matches_results[pdb_file] = {
                        'seq': sequence_string,
                        'matching_gt': {
                            'sequence': [],
                            'mmseqs_id': np.array([], dtype=np.float32),
                            'target_coverage': np.array([], dtype=np.float32),
                            'effective_identity': np.array([], dtype=np.float32),
                        }
                    }

                pdb_idx += 1

    # Save detailed results
    detailed_results_file = f"{mmseqs_path}/detailed_results.json"
    with open(detailed_results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)

    all_matches_file = f"{mmseqs_path}/all_matches_results.pkl"
    # Set compress=True to gzip (smaller but a bit slower); False is already compact & fast
    save_pickle(all_matches_results, all_matches_file, compress=False)

    if gt_total_sequences:
        gt_present_total = gt_present_initial + gt_appended_after_search
        present_pct = (gt_present_total / gt_total_sequences) * 100.0
        initial_pct = (gt_present_initial / gt_total_sequences) * 100.0
        appended_pct = (gt_appended_after_search / gt_total_sequences) * 100.0
        missing_pct = (gt_missing_after_search / gt_total_sequences) * 100.0
        print(
            f"GT coverage in all_matches_results: {gt_present_total}/{gt_total_sequences} "
            f"({present_pct:.2f}%)"
        )
        print(
            f"  - From top hits: {gt_present_initial}/{gt_total_sequences} ({initial_pct:.2f}%), "
            f"appended after search: {gt_appended_after_search}/{gt_total_sequences} ({appended_pct:.2f}%)"
        )
        print(
            f"  - Missing after search: {gt_missing_after_search}/{gt_total_sequences} "
            f"({missing_pct:.2f}%)"
        )
        if gt_missing_after_search:
            no_candidate_pct = (gt_no_candidate_found / gt_missing_after_search) * 100.0
            print(
                f"    - No candidate found: {gt_no_candidate_found}/{gt_missing_after_search} "
                f"({no_candidate_pct:.2f}%)"
            )
    elif ground_truth_by_fasta:
        print("GT coverage in all_matches_results: 0 ground-truth sequences processed.")

    # Print summary DataFrame to console
    print("\nSummary statistics:")
    print(df.to_string(index=False))
    print(f"Novelty analysis saved to {output_file}")
    print(f"Sequence-specific identities saved to {sequence_results_file}")
    print(f"Detailed results saved to {detailed_results_file}")
    print(f"Top-match results saved to {all_matches_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze MMseqs output and compute sequence identity statistics.")
    parser.add_argument('--mmseqs_path', type=str, required=True, help="Path to the MMseqs output file")
    parser.add_argument('--query_fasta', type=str, required=True, help="Path to the query FASTA file")
    parser.add_argument('--metadata_file', type=str, help="Path to the metadata file containing ID mappings")
    parser.add_argument('--training_correspondence', type=str,
                        help="Path to the training sequence correspondence JSON file")
    parser.add_argument('--input_metadata_file', type=str,
                        help="Path to molecule_gt-input_metadata.json with ground-truth protein sequences")
    args = parser.parse_args()

    analyze_mmseqs_output(args.mmseqs_path, args.query_fasta, args.metadata_file,
                          args.training_correspondence, args.input_metadata_file)

if __name__ == "__main__":
    main()
