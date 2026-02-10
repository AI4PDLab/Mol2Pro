# Mol2Pro
### _Code for the paper:_ 
### [Generalise or Memorise? Benchmarking Ligand-Conditioned Protein Generation from Sequence-Only Data](https://doi.org/10.64898/2026.02.06.704305)

Generate protein binder sequences from ligand SMILES using an encoder–decoder Protein Language Model.

## Requirements

- Python 3.10+
- `torch`, `transformers`, `numpy`

## Pretrained Models

The pretrained models presented in the paper are available at [Hugging Face](https://huggingface.co/collections/AI4PD/mol2pro-family)

## Running inference

**Input:** a text file with one SMILES per line (or a JSON/JSONL file with translation entries).

**Example — model and tokenizers on Hugging Face:**

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