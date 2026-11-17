# Mol2Pro

Code for **Generalise or Memorise? Benchmarking Ligand-Conditioned Protein Generation from Sequence-Only Data**: curated JSON data, split utilities, a small **T5 example** (tokenize, train, [`src/example_t5/inference_t5.py`](src/example_t5/inference_t5.py)), and **evaluation** scripts that assume you already have generated FASTAs and sidecar metadata.

This repository does **not** rebuild the raw database or perform large-scale JSON curation. Use the public dataset on the Hugging Face Hub and/or your own exports that match the expected record shape (see below).

## Dataset

Download or load the curated dataset from:

[https://huggingface.co/datasets/contributor-anonymous/Mol2Pro-Binder-Dataset](https://huggingface.co/datasets/contributor-anonymous/Mol2Pro-Binder-Dataset)

Records are expected to carry a top-level **`translation`** object with molecule and protein fields, for example:

- **SMILES-style**: `Canonical Smiles` or `SELFIES`, plus `Protein sequence` or `Sequence`
- **Reaction-style**: `Reaction` plus `Sequence` (and related keys as in your export)

Splits and [`src/example_t5/tokenize_dataset.py`](src/example_t5/tokenize_dataset.py) assume this layout. For details on loading (Hub-first vs local JSON/JSONL), see [`src/example_t5/README.md`](src/example_t5/README.md).

## Installation

- **Python**: 3.10+ recommended (3.12 is used in development).
- **Dependencies**: from the repository root:

  ```bash
  pip install -r requirements.txt
  ```

  [`requirements.txt`](requirements.txt) pins **PyTorch CUDA 12.1** wheels via `--extra-index-url https://download.pytorch.org/whl/cu121`. You need a GPU driver compatible with that stack; for CPU-only installs, adjust the index / packages manually.

- **RDKit**: required for some paths (e.g. novelty joining and max-protein-distance splits). The PyPI `rdkit` package works on many platforms; on Windows, use a [conda-forge](https://conda-forge.org/) environment if pip fails.

### System tools (not installed via pip)

| Tool | Used for |
|------|-----------|
| **[MMseqs2](https://github.com/soedinglab/MMseqs2)** | Ground-truth identity, train-set identity, permutation controls (`src/eval/gt_acc/`, `src/eval/train_identity/`). Install separately and ensure `mmseqs` is on `PATH`, or pass `--mmseq_path` / `MMSEQS_BIN_DIR` where supported. |
| **Bash** | Orchestration scripts under [`src/example_t5/eval_scripts/`](src/example_t5/eval_scripts/) (Git Bash, WSL, or Linux). |
| **Boltz** (optional) | Cofolding pipeline in [`src/eval/cofolding/`](src/eval/cofolding/). |
| **ESMFold** (optional) | pLDDT path in [`src/eval/plddt/run-esm.py`](src/eval/plddt/run-esm.py); set `ESMFOLD_PATH` when using [`run_inference_and_eval.sh`](src/example_t5/eval_scripts/run_inference_and_eval.sh) with `RUN_PLDDT=1`. |

## Repository layout

| Path | Role |
|------|------|
| [`src/data_processing/`](src/data_processing/) | Split generators: random molecule-level splits and MMseqs-aware max-protein-distance splits. |
| [`src/eval/`](src/eval/) | Metrics and analysis on **existing** inference outputs (FASTAs + JSON metadata). No model weights loaded here. |
| [`src/example_t5/`](src/example_t5/) | T5-oriented workflow: tokenize, train, `inference_t5.py`, and bash wrappers that call `src/eval/`. |
| [`tests/`](tests/) | Offline tests (see [`pytest.ini`](pytest.ini) and project testing rules). |

## Splits (`src/data_processing/`)

### Random splits (no SMILES leakage across train/val/test)

```bash
python src/data_processing/random_splits.py --dataset_json /path/to/dataset.json --output_path /path/to/out_dir
```

Writes `train.jsonl`, `val.jsonl`, `test.jsonl`, and `split_summary.json` under `--output_path`.

### Max-protein-distance / MMseqs-aware splits

```bash
python src/data_processing/max_protein_dist_splits.py \
  --dataset_json /path/to/dataset.json \
  --dataset_fasta /path/to/proteins.fasta \
  [--dataset_mmseqs /path/to/mmseqs_results.tsv] \
  ...
```

See `--help` for thresholds and output naming. This path expects MMseqs-related inputs in addition to the JSON corpus.

## Evaluation (`src/eval/`)

**Contract:** every script here assumes an **inference run directory** already containing:

- Generated **`*.fasta`** files
- **`molecule_input_metadata.json`** (FASTA path → input SMILES/reaction)
- **`molecule_gt-input_metadata.json`** (per-run GT and perplexity sidecars where applicable)

Those files are produced by [`src/example_t5/inference_t5.py`](src/example_t5/inference_t5.py) (or any tool that mirrors the same layout). **`src/eval/` does not run neural inference.**

### Ground-truth accuracy (`src/eval/gt_acc/`)

Compares generated sequences to held-out ground-truth proteins with MMseqs.

| Script | Purpose |
|--------|---------|
| [`gt_inference_mmseq.py`](src/eval/gt_acc/gt_inference_mmseq.py) | Align generations to GT; writes MMseqs inputs/outputs under the inference folder. |
| [`gt_mmseq_interpretation_balanced.py`](src/eval/gt_acc/gt_mmseq_interpretation_balanced.py) | Aggregates MMseqs results into balanced GT metrics. |
| [`gt_permutation_mmseq_eval.py`](src/eval/gt_acc/gt_permutation_mmseq_eval.py) | Wrong-target / permutation-style controls on MMseqs outputs. |

Typical pattern: run `gt_inference_mmseq.py` with `--inference_folder`, `--dataset_json` (test split), `--output_folder_gt`, then run `gt_mmseq_interpretation_balanced.py` on the resulting `gt_mmseqs` tree. See each file’s `--help` for full arguments.

### Train-set identity (`src/eval/train_identity/`)

Measures how similar generations are to proteins in the training reference.

| Script | Purpose |
|--------|---------|
| [`mmseq_file_preparation.py`](src/eval/train_identity/mmseq_file_preparation.py) | Builds training FASTA and correspondence files from the train JSON. |
| [`mmseq_interpretation.py`](src/eval/train_identity/mmseq_interpretation.py) | Parses MMseqs search against the training database. |

The bash helper [`src/example_t5/eval_scripts/run_train_identity_mmseqs.sh`](src/example_t5/eval_scripts/run_train_identity_mmseqs.sh) wires these together with `mmseqs easy-search`.

### Novelty / join analysis (`src/eval/novelty_analysis/`)

| Script | Purpose |
|--------|---------|
| [`join_gt_train_mmseqs.py`](src/eval/novelty_analysis/join_gt_train_mmseqs.py) | Joins GT MMseqs outputs with train-identity MMseqs outputs; optional ligand novelty logic and plots. Expects paths to prior step artifacts (`detailed_results.json`, `all_matches_results.pkl`, etc.). |

[`src/example_t5/eval_scripts/join_mmseqs_results.sh`](src/example_t5/eval_scripts/join_mmseqs_results.sh) demonstrates one set of environment defaults (including optional fingerprint stores).

### pLDDT / folding (`src/eval/plddt/`)

| Script | Purpose |
|--------|---------|
| [`run-esm.py`](src/eval/plddt/run-esm.py) | Runs folding / pLDDT extraction given a FASTA folder and ESMFold installation. |
| [`plot_plddt.py`](src/eval/plddt/plot_plddt.py) | Summarizes pLDDT outputs under a results folder. |

### Cofolding (`src/eval/cofolding/`)

| Script | Purpose |
|--------|---------|
| [`boltz_inference_cofold_prepare.py`](src/eval/cofolding/boltz_inference_cofold_prepare.py) | Prepares Boltz YAML inputs from an inference run (often using prior folding outputs). |
| [`collect_boltz_cofold_inference_results.py`](src/eval/cofolding/collect_boltz_cofold_inference_results.py) | Aggregates Boltz prediction artifacts. |

[`src/example_t5/eval_scripts/run_boltz2_cofolding.sh`](src/example_t5/eval_scripts/run_boltz2_cofolding.sh) loops Boltz over a manifest.

### One-shot evaluation wrapper

[`src/example_t5/eval_scripts/run_inference_and_eval.sh`](src/example_t5/eval_scripts/run_inference_and_eval.sh) runs `inference_t5.py`, GT MMseqs, train-identity MMseqs, the join step, and optionally pLDDT. Required environment variables are documented at the top of that script (`MODEL_PATH`, `TOKENIZER_*`, `INFERENCE_INPUT`, `TEST_DATASET`, `TRAIN_DATASET`, …).

## T5 example

See **[`src/example_t5/README.md`](src/example_t5/README.md)** for tokenization (Hub-first and local JSON), training, inference, and how the eval scripts are invoked in order.

## Citation

If you find this work useful, please cite:

```bibtex
@article{VicenteSola2026Generalise,
  title   = {Generalise or Memorise? Benchmarking Ligand-Conditioned Protein Generation from Sequence-Only Data},
  author  = {Vicente-Sola, Alex and Dornfeld, Lars and Coines, Joan and Ferruz, Noelia},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.02.06.704305},
}