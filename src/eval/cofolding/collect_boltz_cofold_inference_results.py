#!/usr/bin/env python3
"""Aggregate Boltz-2 cofolding outputs from boltz_inference_cofold runs.

Reads cofolding_results/manifest.json and per-target boltz_output trees, extracts
confidence_*.json and affinity_*.json metrics, writes summary CSV/JSON, mean±std,
histogram of affinity scores, and a JSON map fasta→target, sequences, scores.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np


def _find_first_json(pred_dir: str, pattern: str) -> Optional[str]:
    hits = sorted(glob.glob(os.path.join(pred_dir, pattern)))
    return hits[0] if hits else None


def load_confidence_affinity(pred_dir: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """pred_dir = .../boltz_output/predictions/<stem>/"""
    conf_path = _find_first_json(pred_dir, "confidence_*.json")
    aff_path = _find_first_json(pred_dir, "affinity_*.json")
    conf: dict[str, Any] = {}
    aff: dict[str, Any] = {}
    if conf_path and os.path.isfile(conf_path):
        with open(conf_path, encoding="utf-8") as f:
            conf = json.load(f)
    if aff_path and os.path.isfile(aff_path):
        with open(aff_path, encoding="utf-8") as f:
            aff = json.load(f)
    return conf, aff


def discover_prediction_dir(boltz_out: str, stem: str) -> Optional[str]:
    """Find the folder containing confidence_*.json / affinity_*.json.

    Boltz-2 writes: out_dir/boltz_results_<input>/predictions/<input>/
    so for --out_dir .../boltz_output and input target_00000.yaml:
      boltz_output/boltz_results_target_00000/predictions/target_00000/

    We search multiple layouts for robustness.
    """
    if not os.path.isdir(boltz_out):
        return None

    # Build candidate prediction roots: predictions/ at various depths
    pred_roots = [
        os.path.join(boltz_out, "predictions"),
    ]
    # boltz_results_<stem>/predictions/  (observed Boltz-2 layout)
    br = os.path.join(boltz_out, f"boltz_results_{stem}", "predictions")
    pred_roots.append(br)
    # Any boltz_results_*/predictions/ directory
    for entry in os.listdir(boltz_out):
        if entry.startswith("boltz_results"):
            cand = os.path.join(boltz_out, entry, "predictions")
            if cand not in pred_roots:
                pred_roots.append(cand)

    for pred_root in pred_roots:
        if not os.path.isdir(pred_root):
            continue
        direct = os.path.join(pred_root, stem)
        if os.path.isdir(direct):
            return direct
        subs = [d for d in os.listdir(pred_root) if os.path.isdir(os.path.join(pred_root, d))]
        if len(subs) == 1:
            return os.path.join(pred_root, subs[0])

    # Flat: confidence_*.json directly in boltz_out
    if glob.glob(os.path.join(boltz_out, "confidence_*.json")):
        return boltz_out

    # Walk one level for any confidence_*.json
    for entry in os.listdir(boltz_out):
        cand = os.path.join(boltz_out, entry)
        if os.path.isdir(cand) and glob.glob(os.path.join(cand, "confidence_*.json")):
            return cand

    return None


def summarize_metrics(conf: dict[str, Any], aff: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for k in (
        "confidence_score",
        "ptm",
        "iptm",
        "ligand_iptm",
        "protein_iptm",
        "complex_plddt",
        "complex_iplddt",
    ):
        if k in conf:
            row[f"boltz_{k}"] = conf[k]
    for k in (
        "affinity_pred_value",
        "affinity_probability_binary",
        "affinity_pred_value1",
        "affinity_probability_binary1",
    ):
        if k in aff:
            row[f"boltz_{k}"] = aff[k]
    return row


def _ensemble_head_values(aff: dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    """Return (head0, head1) binary probabilities from the raw affinity dict.

    Boltz-2 outputs an ensemble average as ``affinity_probability_binary`` and
    one individual head as ``affinity_probability_binary1``.  The other head is
    derived: head0 = 2*avg - head1.
    """
    avg = aff.get("affinity_probability_binary")
    h1 = aff.get("affinity_probability_binary1")
    if avg is None or h1 is None:
        return (None, None)
    try:
        avg_f, h1_f = float(avg), float(h1)
    except (TypeError, ValueError):
        return (None, None)
    h0_f = 2.0 * avg_f - h1_f
    return (h0_f, h1_f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run_dir",
        required=True,
        help="Inference run folder (must contain cofolding_results/manifest.json)",
    )
    ap.add_argument(
        "--cofolding_subdir",
        default="cofolding_results",
        help="Subfolder of run_dir with manifest and runs/ (default: cofolding_results)",
    )
    ap.add_argument(
        "--binding_histogram_key",
        default="affinity_probability_binary",
        choices=(
            "affinity_pred_value",
            "affinity_probability_binary",
            "affinity_pred_value1",
            "affinity_probability_binary1",
        ),
        help="Which affinity field to histogram (default: binary binder probability)",
    )
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    cf_root = os.path.join(run_dir, args.cofolding_subdir)
    manifest_path = os.path.join(cf_root, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(manifest_path)

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    targets = manifest.get("targets") or []
    per_target: list[dict[str, Any]] = []
    raw_affinities: list[dict[str, Any]] = []
    fasta_map: dict[str, Any] = {}
    hist_values: list[float] = []

    for t in targets:
        stem = t["stem"]
        ti = int(t["target_index"])
        boltz_out = os.path.join(cf_root, "runs", stem, "boltz_output")
        pred_dir = discover_prediction_dir(boltz_out, stem)
        if pred_dir is None:
            contents = os.listdir(boltz_out) if os.path.isdir(boltz_out) else "<dir not found>"
            print(f"[warn] missing predictions under {boltz_out}  (contents: {contents})")
            conf, aff = {}, {}
        else:
            conf, aff = load_confidence_affinity(pred_dir)

        metrics = summarize_metrics(conf, aff)
        row = {
            "target_index": ti,
            "stem": stem,
            "fasta_path": t.get("fasta_path"),
            "generation_index": t.get("generation_index"),
            "selection_mean_plddt_esmfold": t.get("selection_mean_plddt_esmfold"),
            "smiles": t.get("smiles"),
            "protein_sequence": t.get("protein_sequence"),
            **metrics,
        }
        row["boltz_success"] = bool(conf or aff)
        per_target.append(row)
        raw_affinities.append(aff)

        fasta_key = os.path.basename(t.get("fasta_path") or "")
        fasta_map[fasta_key] = {
            "target_index": ti,
            "smiles": t.get("smiles"),
            "protein_sequence_selected": t.get("protein_sequence"),
            "generation_index": t.get("generation_index"),
            "selection_mean_plddt_esmfold": t.get("selection_mean_plddt_esmfold"),
            "boltz_confidence": conf,
            "boltz_affinity": aff,
        }

        hk = f"boltz_{args.binding_histogram_key}"
        if hk in row and row[hk] is not None:
            try:
                hist_values.append(float(row[hk]))
            except (TypeError, ValueError):
                pass

    # CSV
    import csv

    csv_path = os.path.join(cf_root, "boltz_cofold_summary.csv")
    if per_target:
        fieldnames = sorted({k for r in per_target for k in r.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in per_target:
                w.writerow(r)
    print(f"Wrote {csv_path}")

    # Aggregate stats for numeric boltz_* columns
    stats: dict[str, dict[str, float]] = {}
    numeric_keys = []
    for r in per_target:
        for k, v in r.items():
            if not k.startswith("boltz_"):
                continue
            if k == "boltz_success":
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                numeric_keys.append(k)
    numeric_keys = sorted(set(numeric_keys))
    for k in numeric_keys:
        vals = []
        for r in per_target:
            v = r.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                vals.append(float(v))
        if vals:
            a = np.array(vals, dtype=float)
            stats[k] = {"mean": float(a.mean()), "std": float(a.std(ddof=0)), "n": float(len(a))}

    # Affinity threshold percentages
    n_with_aff = 0
    n_binary_gt_05 = 0
    n_binary_gt_07 = 0
    n_optimistic_gt_07 = 0
    n_combined = 0
    for aff in raw_affinities:
        avg_raw = aff.get("affinity_probability_binary")
        if avg_raw is None:
            continue
        try:
            avg_f = float(avg_raw)
        except (TypeError, ValueError):
            continue
        n_with_aff += 1
        if avg_f > 0.5:
            n_binary_gt_05 += 1
        if avg_f > 0.7:
            n_binary_gt_07 += 1
        h0, h1 = _ensemble_head_values(aff)
        any_head_gt_07 = (h0 is not None and h0 > 0.7) or (h1 is not None and h1 > 0.7)
        if any_head_gt_07:
            n_optimistic_gt_07 += 1
        if any_head_gt_07 and avg_f > 0.5:
            n_combined += 1

    threshold_stats: dict[str, Any] = {}
    if n_with_aff > 0:
        threshold_stats = {
            "n_cofoldings_with_affinity": n_with_aff,
            "pct_affinity_binary_gt_0.5": round(100.0 * n_binary_gt_05 / n_with_aff, 2),
            "pct_affinity_binary_gt_0.7": round(100.0 * n_binary_gt_07 / n_with_aff, 2),
            "pct_optimistic_any_head_gt_0.7": round(100.0 * n_optimistic_gt_07 / n_with_aff, 2),
            "pct_combined_any_head_gt_0.7_and_avg_gt_0.5": round(100.0 * n_combined / n_with_aff, 2),
            "count_affinity_binary_gt_0.5": n_binary_gt_05,
            "count_affinity_binary_gt_0.7": n_binary_gt_07,
            "count_optimistic_any_head_gt_0.7": n_optimistic_gt_07,
            "count_combined_any_head_gt_0.7_and_avg_gt_0.5": n_combined,
        }
        stats["affinity_threshold_analysis"] = threshold_stats
        print("\n=== Affinity threshold analysis ===")
        print(f"  Cofoldings with affinity data: {n_with_aff}")
        print(f"  affinity_probability_binary > 0.5:                          "
              f"{n_binary_gt_05}/{n_with_aff}  ({threshold_stats['pct_affinity_binary_gt_0.5']:.2f}%)")
        print(f"  affinity_probability_binary > 0.7:                          "
              f"{n_binary_gt_07}/{n_with_aff}  ({threshold_stats['pct_affinity_binary_gt_0.7']:.2f}%)")
        print(f"  optimistic (any head > 0.7):                                "
              f"{n_optimistic_gt_07}/{n_with_aff}  ({threshold_stats['pct_optimistic_any_head_gt_0.7']:.2f}%)")
        print(f"  combined (any head > 0.7 AND avg > 0.5):                    "
              f"{n_combined}/{n_with_aff}  ({threshold_stats['pct_combined_any_head_gt_0.7_and_avg_gt_0.5']:.2f}%)")
    else:
        print("[warn] No affinity data available for threshold analysis.")

    stats_path = os.path.join(cf_root, "boltz_cofold_aggregate_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {stats_path}")

    hist_key = f"boltz_{args.binding_histogram_key}"
    hist_export = {
        "histogram_metric": hist_key,
        "values": hist_values,
    }
    with open(os.path.join(cf_root, "boltz_cofold_histogram_values.json"), "w", encoding="utf-8") as f:
        json.dump(hist_export, f, indent=2)

    # Histogram
    if hist_values:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(hist_values, bins=min(40, max(10, len(hist_values) // 2)), color="#4C72B0", edgecolor="black", alpha=0.85)
        ax.set_xlabel(args.binding_histogram_key)
        ax.set_ylabel("count")
        ax.set_title(f"Boltz affinity metric ({args.binding_histogram_key}), n={len(hist_values)}")
        fig.tight_layout()
        hist_path = os.path.join(cf_root, "boltz_cofold_binding_histogram.png")
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {hist_path}")
    else:
        print("[warn] No values for histogram; check affinity JSON outputs.")

    map_path = os.path.join(cf_root, "boltz_cofold_fasta_to_scores.json")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(fasta_map, f, indent=2)
    print(f"Wrote {map_path}")


if __name__ == "__main__":
    main()
