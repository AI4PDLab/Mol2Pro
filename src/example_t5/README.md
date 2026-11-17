# T5 example (`example_t5`)

End-to-end placeholder flow: **tokenize** curated Mol2Pro JSON → **train** a seq2seq model → **infer** with [`inference_t5.py`](inference_t5.py) → **evaluate** with scripts under [`../eval/`](../eval/) (those scripts expect FASTAs + metadata already on disk).

## Dataset loading (Hub-first, local fallback)

[`tokenize_dataset.py`](tokenize_dataset.py) uses Hugging Face [`datasets.load_dataset`](https://huggingface.co/docs/datasets/) in both cases:

| Source | `--dataset` value | Behavior |
|--------|-------------------|----------|
| **Hub (default)** | `contributor-anonymous/Mol2Pro-Binder-Dataset` | `load_dataset(dataset_id, …)` |
| **Local file or folder** | Path that exists on disk | `load_dataset("json", data_files=…)` |

- If `--dataset` is a **file** (`.json` / `.jsonl`), that file is loaded as JSON lines or array JSON.
- If `--dataset` is a **directory**, split files are inferred when named like `train.jsonl`, `val.jsonl`, `test.jsonl` (same style as [`../data/random_splits.py`](../data/random_splits.py)).
- Override inferred splits with repeated `--data_files split=relative_or_absolute_path` (e.g. `--data_files train=train.jsonl --data_files validation=val.jsonl`).

Optional Hub flags: `--revision`, `--trust_remote_code`, `--split`, `--cache_dir`.

Outputs:

- **Tokenized dataset**: `--output_dir` contains a `DatasetDict` saved via `save_to_disk` (includes `dataset_dict.json`).
- **Metadata**: `tokenize_metadata.json` next to it (tokenizer paths, inferred fields).

## One-shot train wrapper

[`run_train.sh`](run_train.sh) runs `tokenize_dataset.py` when `TOKENIZED_DIR/dataset_dict.json` is missing (or when `FORCE_TOKENIZE=1`), then [`train.py`](train.py).

Required environment variables:

- `TOKENIZER_AA`, `TOKENIZER_MOL` — same as inference.
- `MODEL_NAME_OR_PATH` — base checkpoint (e.g. `google/t5-small`).

Optional:

- `DATASET` — Hub id (default `contributor-anonymous/Mol2Pro-Binder-Dataset`).
- `LOCAL_DATASET_PATH` — if set, used instead of `DATASET` for `--dataset` (local path).
- `TOKENIZED_DIR` — default `./example_t5/tokenized`.
- `OUTPUT_DIR` — training output; default `./example_t5/training_output`.
- `HF_REVISION`, `TRUST_REMOTE_CODE`, `FORCE_TOKENIZE`, `TRAIN_EXTRA_ARGS`.

Example (Git Bash):

```bash
export TOKENIZER_AA=/path/to/aa_tokenizer
export TOKENIZER_MOL=/path/to/mol_tokenizer
export MODEL_NAME_OR_PATH=google/t5-small
bash example_t5/run_train.sh
```

Local splits instead of Hub:

```bash
export LOCAL_DATASET_PATH=/path/to/splits_dir   # contains train.jsonl, val.jsonl, ...
bash example_t5/run_train.sh
```

## Scripts overview

| Script / path | Role |
|---------------|------|
| [`tokenize_dataset.py`](tokenize_dataset.py) | Hub-first / local JSON → tokenized `DatasetDict` on disk. |
| [`train.py`](train.py) | Minimal `Seq2SeqTrainer` on that `DatasetDict` (extend as needed). |
| [`inference_t5.py`](inference_t5.py) | Generates FASTAs + `molecule_*_metadata.json` for eval. |
| [`eval_scripts/run_inference_and_eval.sh`](eval_scripts/run_inference_and_eval.sh) | Inference + MMseqs / join / optional pLDDT. |

## Notes

- **Tokenizers**: Use Hugging Face `AutoTokenizer` checkpoints only (local dirs or Hub ids). `tokenize_dataset.py` and `inference_t5.py` load them via `transformers.AutoTokenizer.from_pretrained`.
- **`train.py`** loads `AutoTokenizer` from the `tokenizer_aa` path recorded in `tokenize_metadata.json`.
