Oficcial implementation of "Generalise or Memorise? Benchmarking Ligand-Conditioned Protein
Generation from Sequence-Only Data"

# Mol2Pro

Inference script to generate protein binder sequences from ligand SMILES using a T5-style encoder–decoder model.

## Requirements

- Python 3.10+
- `torch`, `transformers`, `numpy`

## Pretrained Models

The prretrained models presented in the paper are available at https://huggingface.co/collections/AI4PD/mol2pro-family

## Running inference

**Input:** a text file with one SMILES per line (or a JSON/JSONL file with translation entries).

**Example — model and tokenizers on Hugging Face (single repo):**

```bash
python src/inference.py \
  --model_path AI4PD/Mol2Pro-base \
  --tokenizer_aa AI4PD/Mol2Pro-tokenizer \
  --tokenizer_mol AI4PD/Mol2Pro-tokenizer \
  --input_file data/smiles.txt \
  --output_folder fastas \
  --top_k 15 \
  --seed 0
```


Outputs: one FASTA per input SMILES (25 sequences per molecule by default), plus `inference_metadata.json` (per-FASTA SMILES, perplexities, optional ground-truth sequences; run-level greedy perplexity stats) and `generation_parameters.json` in `--output_folder`.