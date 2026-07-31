#!/usr/bin/env python3
import json
import argparse
from typing import Dict, Any, Tuple, List, Optional
import os
import re
import math
import csv
import colorsys
import numpy as np
import pickle
import zlib

# Only import matplotlib if we actually plot (so script works on headless systems by default)
def _lazy_import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _load_fingerprint_store(fp_store_path: str) -> Tuple[Dict[str, bytes], Dict[str, Any]]:
    """
    Load precomputed fingerprint store.
    Returns (store_dict, metadata_dict).
    store_dict maps RAW_SMILES → zlib-compressed bitstring payload.
    """
    if not os.path.exists(fp_store_path):
        raise FileNotFoundError(f"Fingerprint store not found: {fp_store_path}")
    
    with open(fp_store_path, "rb") as f:
        container = pickle.load(f)
    
    meta = container.get("meta", {})
    store = container.get("data", {})
    
    print(f"Loaded fingerprint store from {fp_store_path}")
    print(f"  Metadata: {meta}")
    print(f"  Cached fingerprints: {len(store)} (raw-SMILES keyed)")
    
    return store, meta


def _get_fingerprint_from_store(smiles: str, fp_store: Dict[str, bytes], gen) -> Optional[Any]:
    """
    Retrieve fingerprint from store for the given SMILES, using the RAW string as key.
    Returns ExplicitBitVect or None if not found.
    """
    from rdkit.DataStructs import CreateFromBitString

    payload = fp_store.get(smiles)  # <— no canonicalization; raw key
    if payload is None:
        return None

    try:
        bs = zlib.decompress(payload).decode("ascii")
        return CreateFromBitString(bs)  # faster than manual bit loop
    except Exception as e:
        print(f"Warning: failed to decompress fingerprint for raw key: {e}")
        return None


def _compute_fingerprint_online(smiles: str, gen) -> Optional[Any]:
    """
    Compute fingerprint on-the-fly for given SMILES.
    """
    from rdkit import Chem
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return gen.GetFingerprint(mol)


def _load_train_container(train_path: str):
    """
    Load training matches container.
    Supports: pickle (preferred for .pkl) and legacy JSON.
    """
    # Try pickle first (works even if extension isn't .pkl)
    try:
        with open(train_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        pass

    # Fallback: JSON
    with open(train_path, "r") as f:
        return json.load(f)


def _matching_gt_as_list(mg: Any) -> List[Dict[str, Any]]:
    """
    Normalize matching_gt to a list[dict], regardless of format.
    - If mg is already a list[dict] (legacy JSON), return as-is.
    - If mg is the new columnar dict with arrays, convert to list[dict].
    """
    if isinstance(mg, list):
        # Legacy JSON: already list of dicts
        return [m for m in mg if isinstance(m, dict)]

    if isinstance(mg, dict):
        seqs = mg.get("sequence", [])
        ids  = mg.get("mmseqs_id", [])
        covs = mg.get("target_coverage", [])
        effs = mg.get("effective_identity", [])

        # Convert numpy arrays to Python lists when needed
        if hasattr(seqs, "tolist"): seqs = seqs.tolist()
        if hasattr(ids,  "tolist"): ids  = ids.tolist()
        if hasattr(covs, "tolist"): covs = covs.tolist()
        if hasattr(effs, "tolist"): effs = effs.tolist()

        n = min(len(seqs), len(ids), len(covs), len(effs))
        out = []
        for i in range(n):
            out.append({
                "sequence": seqs[i],
                "mmseqs_id": float(ids[i]),
                "target_coverage": float(covs[i]),
                "effective_identity": float(effs[i]),
            })
        return out

    return []


def _stddev(values: List[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    m = sum(values) / len(values)
    var = sum((v - m) * (v - m) for v in values) / len(values)  # population std
    return math.sqrt(var)


def _is_leaf(d: Any) -> bool:
    return isinstance(d, dict) and ("seq" in d or "mmseqs_id" in d or "closest_gt" in d or "closest_train_seq" in d)


def _flatten_gt(d: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Flattens GT JSON which has structure:
    { "output_topk15_file-34.fasta": { "top_output_topk15_file-34.fasta_0.pdb": {...leaf...}, ... }, ... }
    Returns { "top_output_topk15_file-34.fasta_0.pdb": {...leaf...} }.
    """
    out = {}
    for fasta_key, fasta_content in d.items():
        if isinstance(fasta_content, dict):
            for pdb_key, pdb_content in fasta_content.items():
                if _is_leaf(pdb_content):
                    out[pdb_key] = pdb_content
    return out

def _flatten_train(d):
    """
    Flattens train JSON. Supports either already-flat dicts or nested dicts of the form
    { "output_topk15_file-34.fasta": { "top_output_topk15_file-34.fasta_0.pdb": {...leaf...}, ... }, ... }.
    Returns { "top_output_topk15_file-34.fasta_0.pdb": {...leaf...} }.
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            if _is_leaf(v):
                out[k] = v
            else:
                for inner_k, inner_v in v.items():
                    if _is_leaf(inner_v):
                        out[inner_k] = inner_v
        elif _is_leaf(v):
            out[k] = v
    return out


def _to_float_or_zero(x: Any) -> float:
    if x in (None, "", "NaN", "nan"):
        return 0.0
    if isinstance(x, (int, float)):
        return 0.0 if (isinstance(x, float) and math.isnan(x)) else float(x)
    if isinstance(x, str):
        try:
            val = float(x)
            return 0.0 if math.isnan(val) else val
        except ValueError:
            return 0.0
    return 0.0


def _presence(val: Any, numeric: bool = False) -> bool:
    """
    Considered 'found' if:
      - numeric: value is a number (including 0.0) and not NaN
      - text: non-empty string
    """
    if numeric:
        if isinstance(val, (int, float)):
            return not (isinstance(val, float) and math.isnan(val))
        # allow numeric strings
        if isinstance(val, str) and val != "":
            try:
                v = float(val)
                return not math.isnan(v)
            except ValueError:
                return False
        return False
    else:
        return isinstance(val, str) and val != ""


def _load_inference_smiles_map(out_dir: str, explicit_meta: Optional[str] = None) -> Dict[str, str]:
    """
    Returns mapping from FASTA basename (e.g. 'output_topk15_file-34.fasta') → SMILES.
    Looks for molecule_input_metadata.json (or the explicit path) in out_dir.
    Falls back to molecule_gt-input_metadata.json if present.
    """
    candidates: List[str] = []
    if explicit_meta:
        candidates.append(explicit_meta)
    else:
        for fname in os.listdir(out_dir):
            if fname.startswith("molecule_input_metadata") and fname.endswith(".json"):
                candidates.append(os.path.join(out_dir, fname))
            elif fname.startswith("molecule_gt-input_metadata") and fname.endswith(".json"):
                candidates.append(os.path.join(out_dir, fname))

    print(f"Using as metadata: {candidates}")
    for path in candidates:
        try:
            with open(path, "r") as f:
                raw = json.load(f)
            # raw maps absolute fasta path → smiles
            # Convert to basename → smiles
            out = {}
            for k, v in raw.items():
                if not isinstance(v, str):
                    # skip non-file entries like Greedy_Perplexity in gt metadata
                    continue
                base = os.path.basename(k)
                out[base] = v
            if out:
                print(f"Loaded inference SMILES map from {path} with {len(out)} entries")
                return out
        except Exception as e:
            print(f"Warning: could not read metadata {path}: {e}")
    return {}


def _iter_dataset_records(json_path: str):
    """
    Iterate dataset entries yielding (molecule, protein) pairs.
    Supports JSON lines or a single JSON array file.
    """
    def extract(entry: Dict[str, Any]):
        t = entry.get("translation", {}) if isinstance(entry, dict) else {}
        mol = t.get("Canonical Smiles") or t.get("Reaction")
        prot = t.get("Protein sequence") or t.get("Sequence")
        if isinstance(mol, str) and isinstance(prot, str):
            return mol, prot
        return None

    try:
        with open(json_path, "r", buffering=1024*1024) as f:
            for line in f:
                if line == "\n":
                    continue
                try:
                    entry = json.loads(line)
                    pair = extract(entry)
                    if pair:
                        yield pair
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Warning: training dataset JSON not found: {json_path}")
        return
    

def _build_protein_to_molecules_map(train_dataset_json: Optional[str]) -> Dict[str, List[str]]:
    """
    Build mapping protein sequence → list of associated molecules (unique, sorted).
    Caches result as prot2mol_<training_dataset_basename>.pkl in same directory.
    Returns empty dict if json path is None or cannot be read.
    """
    if not train_dataset_json:
        return {}

    # Derive pickle path
    base = os.path.basename(train_dataset_json)
    pkl_name = f"prot2mol_{base}.pkl"
    pkl_path = os.path.join(os.path.dirname(train_dataset_json), pkl_name)

    # Try to load from cache first
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, dict):
                print(f"Loaded cached protein→molecules map from {pkl_path} ({len(cached)} proteins).")
                return cached
        except Exception as e:
            print(f"Warning: could not load {pkl_path}: {e}. Rebuilding...")

    # Otherwise build from scratch
    prot_to_mols: Dict[str, set] = {}
    n = 0
    try:
        for mol, prot in _iter_dataset_records(train_dataset_json):
            n += 1
            s = prot_to_mols.setdefault(prot, set())
            s.add(mol)
    except Exception as e:
        print(f"Error while reading {train_dataset_json}: {e}")
        return {}

    print(f"Built protein→molecules map from {train_dataset_json}: "
          f"{len(prot_to_mols)} proteins, {n} pairs scanned")

    # Convert sets to sorted lists for JSON stability
    result = {k: sorted(list(v)) for k, v in prot_to_mols.items()}

    # Cache to pickle for next runs
    try:
        with open(pkl_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Cached protein→molecules map to {pkl_path}.")
    except Exception as e:
        print(f"Warning: could not write cache {pkl_path}: {e}")

    return result


def combine(
    gt_json_path: str,
    train_json_path: str,
    *,
    out_dir: Optional[str] = None,
    train_dataset_json: Optional[str] = None,
    inference_metadata_path: Optional[str] = None,
    inference_path: Optional[str] = None,
    fp_store_path: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    print("Load GT MMSeqs results JSON",flush=True)
    with open(gt_json_path, "r") as f:
        gt_raw = json.load(f)
    print("Load train MMSeqs pkl",flush=True)
    tr_raw = _load_train_container(train_json_path)

    gt = _flatten_gt(gt_raw)
    tr = _flatten_train(tr_raw)

    # Load fingerprint store
    print("Load fingerprint store", flush=True)
    fp_store, fp_meta = _load_fingerprint_store(fp_store_path)

    # Optional: build helper maps for extended reporting
    fasta2smiles: Dict[str, str] = {}
    prot2mols: Dict[str, List[str]] = {}
    if out_dir:
        print("Load out file name to SMILES input",flush=True)
        fasta2smiles = _load_inference_smiles_map(inference_path, explicit_meta=inference_metadata_path)
        print(f"Loaded inference SMILES map entries: {len(fasta2smiles)}")
    if train_dataset_json:
        print("Create / Load prot2mol map",flush=True)
        prot2mols = _build_protein_to_molecules_map(train_dataset_json)

    print(f"DEBUG: GT keys sample (before remap): {list(gt.keys())[:3]}", flush=True)
    print(f"DEBUG: Train keys sample: {list(tr.keys())[:3]}", flush=True)

    gt_remapped = {k.replace('top_output', 'output'): v for k, v in gt.items()}

    print(f"DEBUG: GT keys sample (after remap): {list(gt_remapped.keys())[:3]}", flush=True)
    print(f"DEBUG: Key overlap: {len(set(gt_remapped.keys()) & set(tr.keys()))}", flush=True)

    all_keys = set(gt_remapped.keys()) | set(tr.keys())
    combined_entries: List[Tuple[str, Dict[str, Any], float]] = []

    n_total = len(all_keys)
    field_found = {
        "seq": 0,
        "closest_gt": 0,
        "gt_mmseqs_id": 0,
        "closest_train_seq": 0,
        "train_mmseqs_id": 0,
        "gt_effective_id": 0,
        "train_effective_id": 0,
    }
    keys_missing_either = 0
    keys_missing_gt_side = 0
    keys_missing_train_side = 0

    matching_gt_sizes: List[int] = []  # For printing matching sequences in train (pre-dedup)
    matching_gt_sizes_dedup: List[int] = []  # Track post-dedup lengths

    # Counters for fingerprint cache usage
    fp_cache_hits = 0
    fp_cache_misses = 0
    fp_computed_online = 0
    
    for key in all_keys:
        gt_entry = gt_remapped.get(key, {}) or {}
        tr_entry = tr.get(key, {}) or {}

        has_gt_side = bool(gt_entry)
        has_tr_side = bool(tr_entry)
        if not (has_gt_side and has_tr_side):
            keys_missing_either += 1
            if not has_gt_side: keys_missing_gt_side += 1
            if not has_tr_side: keys_missing_train_side += 1

        seq = gt_entry.get("seq") or tr_entry.get("seq") or ""
        closest_gt = gt_entry.get("closest_gt", "")

        # numeric identities
        gt_mmseqs_id = _to_float_or_zero(gt_entry.get("mmseqs_id", ""))
        train_mmseqs_id = _to_float_or_zero(tr_entry.get("mmseqs_id", ""))

        # coverages if available
        gt_cov = _to_float_or_zero(gt_entry.get("target_coverage", 0.0))
        tr_cov = _to_float_or_zero(tr_entry.get("target_coverage", 0.0))

        # prefer precomputed effective_identity if present, else compute
        gt_eff = _to_float_or_zero(gt_entry.get("effective_identity",
                                                gt_mmseqs_id * gt_cov))
        tr_eff = _to_float_or_zero(tr_entry.get("effective_identity",
                                                train_mmseqs_id * tr_cov))

        # Normalize matching_gt to a list[dict] regardless of source format
        matching_gt_raw = tr_entry.get("matching_gt", []) if isinstance(tr_entry, dict) else []
        matching_gt_list: List[Dict[str, Any]] = _matching_gt_as_list(matching_gt_raw)
        pre_dedup_len = len(matching_gt_list)
        matching_gt_sizes.append(pre_dedup_len)  # report pre-dedup length
        post_dedup_len = pre_dedup_len

        best_identity = 0.0

        if matching_gt_list:  # A list of the N highest matching sequences from train
            # 1) normalize numerics
            for m in matching_gt_list:
                m["mmseqs_id"] = _to_float_or_zero(m.get("mmseqs_id", 0.0))
                m["target_coverage"] = _to_float_or_zero(m.get("target_coverage", 0.0))
                m["effective_identity"] = _to_float_or_zero(m.get("effective_identity", 0.0))

            # 2) sort by desirability (effective id, then raw id, then coverage), DESC
            matching_gt_list.sort(
                key=lambda m: (m["effective_identity"], m["mmseqs_id"], m["target_coverage"]),
                reverse=True
            )

            # 3) deduplicate by sequence, keeping the first (best-ranked) occurrence
            seen_seqs = set()
            deduped = []
            for m in matching_gt_list:
                seq_key = m.get("sequence") or ""
                if not seq_key:
                    continue
                if seq_key in seen_seqs:
                    continue
                seen_seqs.add(seq_key)
                deduped.append(m)

            matching_gt_list = deduped
            post_dedup_len = len(matching_gt_list)

            # 4) top match drives summary fields
            top = matching_gt_list[0]
            best_identity = float(top["mmseqs_id"])
        else:
            post_dedup_len = 0

        matching_gt_sizes_dedup.append(post_dedup_len)


        # Sequences in Dict to a List
        best_sequences: List[str] = [m.get("sequence", "") for m in matching_gt_list if isinstance(m, dict) and m.get("sequence")] 

        if matching_gt_list:
            closest_train_seq = top.get("sequence", "") or tr_entry.get("closest_train_seq", "")
            best_train_cov = float(top.get("target_coverage", 0.0))
            best_train_eff = float(top.get("effective_identity", 0.0))
        else:
            closest_train_seq = tr_entry.get("closest_train_seq", "")
            best_train_cov = tr_cov
            best_train_eff = tr_eff


        entry_out: Dict[str, Any] = {
            "seq": seq,
            "closest_gt": closest_gt,
            "gt_mmseqs_id": gt_mmseqs_id,
            "gt_target_coverage": gt_cov,
            "gt_effective_id": gt_eff,

            "closest_train_seq": closest_train_seq,
            "train_mmseqs_id": best_identity if matching_gt_list else train_mmseqs_id,
            "train_target_coverage": best_train_cov,
            "train_effective_id": best_train_eff,
        }
        # Extended reporting: summarize molecules only for sequences tied at best identity
        # (n_train_matches, train_best_match_mol, train_best_match_fp_sim, train_match_fp_sim_std)
        n_train_matches = len(best_sequences) if best_sequences else (1 if closest_train_seq else 0)

        # Build the mapping (each mol associated to matching seqs) -> (matching seq)
        mol_to_srcseq: Dict[str, str] = {}
        if prot2mols and best_sequences:
            for seq_key in best_sequences:
                for smi in prot2mols.get(seq_key, []):  # all smiles of matching seq
                    if isinstance(smi, str) and smi not in mol_to_srcseq:
                        mol_to_srcseq[smi] = seq_key  # keeps the first one that had this mol (highest sim)
        elif prot2mols and closest_train_seq:
            print('WARNING: No train matches read, falling back to single match')
            # Fallback if matching list not present
            for smi in prot2mols.get(closest_train_seq, []):
                if isinstance(smi, str) and smi not in mol_to_srcseq:
                    mol_to_srcseq[smi] = closest_train_seq

        # Input SMILES from metadata (for similarity computation)
        fasta_name = _extract_fasta_name(key)
        fasta_name_norm = fasta_name[4:] if fasta_name.startswith("top_") else fasta_name
        inp_smiles = fasta2smiles.get(fasta_name_norm)
        entry_out["input_molecule_smiles"] = inp_smiles or ""

        best_mol = ""
        best_sim = 0.0
        best_seq_for_best_mol = ""
        all_sims: List[float] = []

        if inp_smiles and mol_to_srcseq:
            try:
                from rdkit import Chem, DataStructs
                from rdkit.Chem import rdFingerprintGenerator

                morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
                
                # Get query fingerprint (try cache first, fallback to online computation)
                qfp = _get_fingerprint_from_store(inp_smiles, fp_store, morgan_gen)
                if qfp is None:
                    fp_computed_online += 1
                    qfp = _compute_fingerprint_online(inp_smiles, morgan_gen)
                    if qfp is not None:
                        print(f"Warning: query SMILES not in cache, computed online: {inp_smiles}")
                else:
                    fp_cache_hits += 1
                
                if qfp is not None:
                    for smi, src_seq in mol_to_srcseq.items():
                        # Get target fingerprint from cache (required)
                        fp = _get_fingerprint_from_store(smi, fp_store, morgan_gen)
                        if fp is None:
                            fp_cache_misses += 1
                            print(f"Warning: target SMILES not found in fingerprint cache: {smi}")
                            continue
                        
                        fp_cache_hits += 1
                        sim = float(DataStructs.TanimotoSimilarity(qfp, fp))
                        all_sims.append(sim)
                        if sim > best_sim:
                            best_sim = sim
                            best_mol = smi
                            best_seq_for_best_mol = src_seq
            except Exception as e:
                print(f"RDKit similarity computation failed for {key}: {e}")

        # Store new compact fields (std is population std; change _stddev if you want sample std)
        entry_out["n_train_matches"] = n_train_matches
        entry_out["train_best_match_mol"] = best_mol
        entry_out["train_best_match_fp_sim"] = float(best_sim)
        entry_out["train_match_fp_sim_std"] = _stddev(all_sims)
        entry_out["train_best_match_sequence"] = best_seq_for_best_mol

        combined_entries.append((key, entry_out, float(gt_eff)))

        # presence counters
        if _presence(seq, numeric=False): field_found["seq"] += 1
        if _presence(closest_gt, numeric=False): field_found["closest_gt"] += 1
        if _presence(gt_mmseqs_id, numeric=True): field_found["gt_mmseqs_id"] += 1
        if _presence(closest_train_seq, numeric=False): field_found["closest_train_seq"] += 1
        if _presence(train_mmseqs_id, numeric=True): field_found["train_mmseqs_id"] += 1
        if _presence(gt_eff, numeric=True): field_found["gt_effective_id"] += 1
        if _presence(tr_eff, numeric=True): field_found["train_effective_id"] += 1

    combined_entries.sort(key=lambda item: (item[2], item[0]))
    combined: Dict[str, Any] = {key: entry for key, entry, _ in combined_entries}

    stats = {
        "n_total_keys": n_total,
        "field_found": field_found,
        "keys_missing_either": keys_missing_either,
        "keys_missing_gt_side": keys_missing_gt_side,
        "keys_missing_train_side": keys_missing_train_side,
        "gt_mmseqs_missing": n_total - field_found["gt_mmseqs_id"],
        "train_mmseqs_missing": n_total - field_found["train_mmseqs_id"],
        # optional: effective-id missing
        "gt_effective_missing": n_total - field_found["gt_effective_id"],
        "train_effective_missing": n_total - field_found["train_effective_id"],
        # fingerprint cache stats
        "fp_cache_hits": fp_cache_hits,
        "fp_cache_misses": fp_cache_misses,
        "fp_computed_online": fp_computed_online,
    }
    
    if matching_gt_sizes:
        avg_matches = sum(matching_gt_sizes) / len(matching_gt_sizes)
        max_matches = max(matching_gt_sizes)
    else:
        avg_matches = 0.0
        max_matches = 0

    if matching_gt_sizes_dedup:
        avg_matches_dedup = sum(matching_gt_sizes_dedup) / len(matching_gt_sizes_dedup)
        max_matches_dedup = max(matching_gt_sizes_dedup)
    else:
        avg_matches_dedup = 0.0
        max_matches_dedup = 0

    stats["matching_gt_pre_avg"] = avg_matches
    stats["matching_gt_pre_max"] = max_matches
    stats["matching_gt_post_avg"] = avg_matches_dedup
    stats["matching_gt_post_max"] = max_matches_dedup

    # Print as requested
    print(f"Average #matches in matching_gt_list per entry (pre-dedup): {avg_matches:.2f}")
    print(f"Maximum #matches in matching_gt_list (pre-dedup): {max_matches}")
    print(f"Average #matches in matching_gt_list per entry (post-dedup): {avg_matches_dedup:.2f}")
    print(f"Maximum #matches in matching_gt_list (post-dedup): {max_matches_dedup}")
    print(f"\nFingerprint cache statistics:")
    print(f"  Cache hits: {fp_cache_hits}")
    print(f"  Cache misses: {fp_cache_misses}")
    print(f"  Computed online (query): {fp_computed_online}")
    
    return combined, stats


_fasta_re = re.compile(r"(.+)_\d+\.pdb$", re.IGNORECASE)


def _extract_fasta_name(pdb_key: str) -> str:
    """
    Given 'my_fasta_name_24.pdb' → 'my_fasta_name'.
    If pattern doesn't match, returns basename without .pdb.
    """
    m = _fasta_re.match(pdb_key)
    if m:
        return m.group(1)
    # Fallback: strip extension
    if pdb_key.lower().endswith(".pdb"):
        return pdb_key[:-4]
    return pdb_key


def _color_for_fasta(fasta_name: str):
    """
    Deterministic, non-repeating color per FASTA using hash→HSV.
    Hue from 0..1 based on hash, with fixed S,V.
    """
    # Stable hash across runs: use a custom FNV-1a-like hash
    h = 2166136261
    for ch in fasta_name.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    hue = (h % 360) / 360.0
    s = 0.65
    v = 0.95
    r, g, b = colorsys.hsv_to_rgb(hue, s, v)
    return (r, g, b)

_PAPER_HIST_COLOR = "#97D3CD"


def _paper_rcparams(scale: float = 1.5) -> Dict[str, Any]:
    import matplotlib as mpl

    base = mpl.rcParams.get("font.size", 10.0)
    try:
        base_f = float(base)
    except (TypeError, ValueError):
        base_f = 10.0

    font_size = base_f * float(scale)
    return {
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "figure.titlesize": font_size,
    }

def _colorbar_minmax_topbottom(cbar, *, title: str = "Train Id.", vmin: float = 0.0, vmax: float = 1.0):
    """
    Replace right-side tick labels with centered min/max labels above/below the bar,
    plus an optional title above the max label.
    """
    cbar.set_label("")
    cbar.set_ticks([])
    cbar.ax.tick_params(left=False, right=False, labelleft=False, labelright=False)
    cbar.ax.text(
        0.5,
        -0.05,
        f"{vmin:.1f}",
        transform=cbar.ax.transAxes,
        ha="center",
        va="top",
        clip_on=False,
    )
    cbar.ax.text(
        0.5,
        1.02,
        f"{vmax:.1f}",
        transform=cbar.ax.transAxes,
        ha="center",
        va="bottom",
        clip_on=False,
    )
    if title:
        cbar.ax.text(
            0.5,
            1.18,
            title,
            transform=cbar.ax.transAxes,
            ha="center",
            va="bottom",
            clip_on=False,
        )


def plot_scatter(combined: Dict[str, Any], out_dir: str):
    plt = _lazy_import_matplotlib()
    import matplotlib as mpl
    import numpy as np  # safe local import

    x_o, y_o, c_o = [], [], []   # GT eff > 0 : circles
    x_x, y_x, c_x = [], [], []   # GT eff == 0 : X markers

    for pdb_key, entry in combined.items():
        x = float(entry.get("gt_effective_id", 0.0))
        y = float(entry.get("train_effective_id", 0.0))
        fasta = _extract_fasta_name(pdb_key)
        col = _color_for_fasta(fasta)
        if x == 0.0:
            x_x.append(x); y_x.append(y); c_x.append(col)
        else:
            x_o.append(x); y_o.append(y); c_o.append(col)

    x_all = np.array(x_o + x_x, dtype=float)
    y_all = np.array(y_o + y_x, dtype=float)

    fig = plt.figure(figsize=(7.0, 7.0))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[4.0, 1.2],
        height_ratios=[1.2, 4.0],
        hspace=0.05, wspace=0.05
    )
    ax_histx = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[1, 0], sharex=ax_histx)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

    if len(x_o):
        ax_scatter.scatter(x_o, y_o, c=c_o, s=18, edgecolors="none")
    if len(x_x):
        ax_scatter.scatter(x_x, y_x, c=c_x, s=28, marker='x', linewidths=1.0)

    ax_scatter.set_xlabel("GT Effective Identity")
    ax_scatter.set_ylabel("Train Effective Identity")
    ax_scatter.set_xlim(0.0, 1.0)
    ax_scatter.set_ylim(0.0, 1.0)

    bins = np.linspace(0.0, 1.0, 31)
    ax_histx.hist(x_all, bins=bins, edgecolor="none")
    ax_histx.set_xlim(0.0, 1.0)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histx.spines["right"].set_visible(False)
    ax_histx.spines["top"].set_visible(False)

    ax_histy.hist(y_all, bins=bins, orientation="horizontal", edgecolor="none")
    ax_histy.set_ylim(0.0, 1.0)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.spines["right"].set_visible(False)
    ax_histy.spines["top"].set_visible(False)

    out_path = os.path.join(out_dir, "accuracy_vs_novelty.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Successfully plotted: {len(x_all)}")
    print(f"Plot saved to: {out_path}")

    paper_out_path = os.path.join(out_dir, "accuracy_vs_novelty_paper.pdf")
    with mpl.rc_context(_paper_rcparams(scale=1.5)):
        fig = plt.figure(figsize=(7.0, 7.0))
        gs = fig.add_gridspec(
            nrows=1, ncols=2,
            width_ratios=[4.0, 1.2],
            wspace=0.05
        )
        ax_scatter = fig.add_subplot(gs[0, 0])
        ax_histy = fig.add_subplot(gs[0, 1], sharey=ax_scatter)

        if len(x_o):
            ax_scatter.scatter(x_o, y_o, c=c_o, s=18, edgecolors="none")
        if len(x_x):
            ax_scatter.scatter(x_x, y_x, c=c_x, s=28, marker='x', linewidths=1.0)

        ax_scatter.set_xlabel("GT Effective Identity")
        ax_scatter.set_ylabel("Train Effective Identity")
        ax_scatter.set_xlim(0.0, 1.0)
        ax_scatter.set_ylim(0.0, 1.0)

        bins = np.linspace(0.0, 1.0, 31)
        ax_histy.hist(y_all, bins=bins, orientation="horizontal", edgecolor="none", color=_PAPER_HIST_COLOR)
        ax_histy.set_ylim(0.0, 1.0)
        ax_histy.tick_params(axis="y", labelleft=False)
        ax_histy.spines["right"].set_visible(False)
        ax_histy.spines["top"].set_visible(False)

        fig.savefig(paper_out_path, bbox_inches="tight")
        plt.close(fig)

    print(f"Paper plot saved to: {paper_out_path}")


def plot_gt_vs_max_similarity(combined: Dict[str, Any], out_dir: str):
    plt = _lazy_import_matplotlib()
    import matplotlib as mpl
    import numpy as np
    from matplotlib import colors

    xs: List[float] = []
    ys: List[float] = []
    cs: List[float] = []

    for entry in combined.values():
        x = float(entry.get("gt_effective_id", 0.0))
        y = entry.get("train_best_match_fp_sim", None)
        if y is None:
            continue
        y = float(y)
        c = float(entry.get("train_effective_id", 0.0))
        xs.append(x)
        ys.append(y)
        cs.append(c)

    if not xs:
        print("Warning: no points available for GT vs max Tanimoto similarity plot")
        return

    fig = plt.figure(figsize=(7.0, 7.0))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[4.0, 1.2],
        height_ratios=[1.2, 4.0],
        hspace=0.05, wspace=0.05
    )
    ax_histx = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[1, 0], sharex=ax_histx)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

    try:
        cmap = plt.get_cmap("BrBG")
    except ValueError:
        from matplotlib import cm
        cmap = cm.get_cmap("inferno")

    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    scatter = ax_scatter.scatter(xs, ys, c=cs, cmap=cmap, norm=norm, s=18, edgecolors="none")

    ax_scatter.set_xlabel("GT Effective Identity")
    ax_scatter.set_ylabel("Max Train Tanimoto Similarity")
    ax_scatter.set_xlim(0.0, 1.0)
    ax_scatter.set_ylim(0.0, 1.0)

    bins = np.linspace(0.0, 1.0, 31)
    ax_histx.hist(xs, bins=bins, edgecolor="none")
    ax_histx.set_xlim(0.0, 1.0)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histx.spines["right"].set_visible(False)
    ax_histx.spines["top"].set_visible(False)

    ax_histy.hist(ys, bins=bins, orientation="horizontal", edgecolor="none")
    ax_histy.set_ylim(0.0, 1.0)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.spines["right"].set_visible(False)
    ax_histy.spines["top"].set_visible(False)

    cbar = fig.colorbar(scatter, ax=ax_scatter, fraction=0.05, pad=0.02)
    cbar.set_label("Train Effective Identity")

    out_path = os.path.join(out_dir, "gt_vs_max_tanimoto_similarity.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Successfully plotted GT vs max Tanimoto similarity: {len(xs)}")
    print(f"Plot saved to: {out_path}")

    paper_out_path = os.path.join(out_dir, "gt_vs_max_tanimoto_similarity_paper.pdf")
    with mpl.rc_context(_paper_rcparams(scale=1.5)):
        fig = plt.figure(figsize=(7.0, 7.0))
        gs = fig.add_gridspec(
            nrows=1, ncols=2,
            width_ratios=[4.0, 1.2],
            wspace=0.05
        )
        ax_scatter = fig.add_subplot(gs[0, 0])
        ax_histy = fig.add_subplot(gs[0, 1], sharey=ax_scatter)

        scatter = ax_scatter.scatter(xs, ys, c=cs, cmap=cmap, norm=norm, s=18, edgecolors="none")

        ax_scatter.set_xlabel("GT Effective Identity")
        ax_scatter.set_ylabel("Max Train Mol Tanimoto Similarity")
        ax_scatter.set_xlim(0.0, 1.0)
        ax_scatter.set_ylim(0.0, 1.0)

        bins = np.linspace(0.0, 1.0, 31)
        ax_histy.hist(ys, bins=bins, orientation="horizontal", edgecolor="none", color=_PAPER_HIST_COLOR)
        ax_histy.set_ylim(0.0, 1.0)
        ax_histy.tick_params(axis="y", labelleft=False)
        ax_histy.spines["right"].set_visible(False)
        ax_histy.spines["top"].set_visible(False)

        cbar = fig.colorbar(scatter, ax=ax_scatter, fraction=0.05, pad=0.02)
        _colorbar_minmax_topbottom(cbar, title="Train Id.")

        fig.savefig(paper_out_path, bbox_inches="tight")
        plt.close(fig)

    print(f"Paper plot saved to: {paper_out_path}")


def _print_stats(stats: Dict[str, Any]):
    n_total = stats["n_total_keys"]
    print("\n=== Coverage Summary ===")
    for field, n_found in stats["field_found"].items():
        print(f"{field}: {n_found} / {n_total} found")

    print("\n=== Cross-JSON Coverage ===")
    print(f"Keys missing from either JSON: {stats['keys_missing_either']} / {n_total}")
    print(f"  - Missing GT side: {stats['keys_missing_gt_side']}")
    print(f"  - Missing Train side: {stats['keys_missing_train_side']}")

    print("\n=== MMseqs identity field missing counts ===")
    print(f"gt_mmseqs_id missing: {stats['gt_mmseqs_missing']}")
    print(f"train_mmseqs_id missing: {stats['train_mmseqs_missing']}")


def main():
    p = argparse.ArgumentParser(
        description="Combine detailed MMseqs results vs GT and vs Training; optional scatter plot.")
    p.add_argument("--gt_json", required=True, help="Path to JSON with detailed results vs ground truth")
    p.add_argument("--train_json", required=True,
                   help="Path to .pkl (or JSON in older versions) with training matches (e.g., all_matches_results.pkl)")
    p.add_argument("--out_path", required=True, help="Output directory path for joined JSON and optional plot")
    p.add_argument("--fp_store", required=True,
                   help="Path to precomputed fingerprint store .pkl (required)")
    p.add_argument(
        "--plot",
        action="store_true",
        help="Also save plot PDFs (including *_paper.pdf variants)",
    )
    p.add_argument("--train_dataset_json", required=False,
                   help="Path to the training dataset JSON (to map proteins→molecules)")
    p.add_argument("--inference_metadata", required=False,
                   help="Optional path to molecule_input_metadata.json. If not set, auto-detect in out_path.")
    p.add_argument("--inference_path", required=False,
                   help="Directory where inference outputs/metadata live (used to find molecule_input_metadata.json)")
    p.add_argument("--ligand_novelty_min_gt_id", type=float, default=1.0,
                   help="Minimum gt_effective_id to include in ligand_novelty.csv (default: 1.0).")
    args = p.parse_args()

    combined, stats = combine(
        args.gt_json,
        args.train_json,
        out_dir=args.out_path,
        train_dataset_json=args.train_dataset_json,
        inference_metadata_path=args.inference_metadata,
        inference_path=args.inference_path,
        fp_store_path=args.fp_store,
    )

    os.makedirs(args.out_path, exist_ok=True)
    out_json = os.path.join(args.out_path, "joined_mmseqs.json")
    with open(out_json, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Combined file written to {out_json} with {len(combined)} entries.")

    # Also write only the entries where GT generalizes beyond train (gt_effective_id > train_effective_id)
    gen_examples = {
        k: v for k, v in combined.items()
        if float(v.get("gt_effective_id", 0.0)) > float(v.get("train_effective_id", 0.0))
    }
    
    high_acc_examples = {
        k: v for k, v in combined.items()
        if float(v.get("gt_effective_id", 0.0)) > 0.8
    }
    n_acc_pass = len(high_acc_examples)
    # Sort items (list of (k,v)) by ascending train_best_match_fp_sim, then take top-k
    high_acc_items = sorted(
        high_acc_examples.items(),
        key=lambda kv: float(kv[1].get("train_best_match_fp_sim", 1.0))
    )
    k_select = max(10, int(math.floor(0.25 * n_acc_pass)))
    gen_examples_mol = dict(high_acc_items[:k_select])

    gen_mol_path = os.path.join(args.out_path, "generalization_examples_molecule.json")
    with open(gen_mol_path, "w") as f:
        json.dump(gen_examples_mol, f, indent=2)
    print(f"Generalization-by-molecule examples written to {gen_mol_path} with {len(gen_examples_mol)} entries. "
          f"(Passed GT>0.8: {n_acc_pass})")

    # Collect high-GT matches (gt_effective_id >= threshold) and summarize ligand novelty
    threshold = float(args.ligand_novelty_min_gt_id)
    perfect_matches = {
        k: v for k, v in combined.items()
        if float(v.get("gt_effective_id", 0.0)) >= threshold
    }
    sims = [float(v.get("train_best_match_fp_sim", 0.0)) for v in perfect_matches.values()]
    avg_sim = sum(sims) / len(sims) if sims else 0.0
    std_sim = _stddev(sims)

    ligand_novelty_csv = os.path.join(args.out_path, f"ligand_novelty_th{threshold}.csv")
    with open(ligand_novelty_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pdb_key", "train_best_match_fp_sim"])
        for k, v in perfect_matches.items():
            writer.writerow([k, float(v.get("train_best_match_fp_sim", 0.0))])
        writer.writerow([])
        writer.writerow(["count", len(perfect_matches)])
        writer.writerow(["avg_fp_sim", avg_sim])
        writer.writerow(["std_fp_sim", "" if std_sim is None else std_sim])

    print(f"Ligand novelty (GT>={threshold:.2f}) → count: {len(perfect_matches)}, "
          f"avg fp_sim: {avg_sim:.4f}, std: {0.0 if std_sim is None else std_sim:.4f}")
    print(f"Ligand novelty CSV written to {ligand_novelty_csv}")
    
    gen_path = os.path.join(args.out_path, "generalization_examples_protein.json")
    with open(gen_path, "w") as f:
        json.dump(gen_examples, f, indent=2)
    print(f"Generalization-by-protein examples written to {gen_path} with {len(gen_examples)} entries.")

    _print_stats(stats)

    if args.plot:
        plot_scatter(combined, args.out_path)
        plot_gt_vs_max_similarity(combined, args.out_path)


if __name__ == "__main__":
    main()
