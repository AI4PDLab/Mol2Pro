"""Inference on a SMILES txt. Saved as fastas.
Previously called generate_comparison"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
import argparse
import os
import torch
import json
import numpy as np
import torch.nn.functional as F
from datetime import datetime
from collections import defaultdict
from pathlib import Path


def compute_transition_scores(sequences, scores, pad_token_id, normalize_by_length=True):
    batch_size = sequences.size(0)
    num_scores = len(scores)

    # Handle sequence/score alignment
    predicted_tokens = sequences[:, 1:1 + num_scores]  # First one is bos or pad

    actual_length = min(predicted_tokens.size(1), num_scores)
    predicted_tokens = predicted_tokens[:, :actual_length]

    if actual_length == 0 or not scores:
        return [0.0] * batch_size

    # Stack scores: (T, B, V)
    logits_tensor = torch.stack(scores[:actual_length], dim=0)

    # Apply log_softmax
    log_probs = F.log_softmax(logits_tensor, dim=-1)  # (T, B, V)

    # Prepare token indices for gather: (T, B, 1)
    target_tokens = predicted_tokens[:, :actual_length].transpose(0, 1).unsqueeze(-1)

    # Return probs of the selected token at each position: (T, B)
    token_logprobs = torch.gather(log_probs, dim=-1, index=target_tokens).squeeze(-1)

    # Create mask for valid (non-inf, non-nan) values
    valid_mask = ~(torch.isinf(token_logprobs) | torch.isnan(token_logprobs))

    # Replace invalid values with 0 (safe for summation)
    token_logprobs = token_logprobs.masked_fill(~valid_mask, 0.0)

    # Compute per-sequence scores
    sum_logprobs = token_logprobs.sum(dim=0)  # (B,)
    valid_lengths = valid_mask.sum(dim=0)     # (B,)

    # Avoid division by zero
    valid_lengths_safe = valid_lengths.clone()
    valid_lengths_safe[valid_lengths_safe == 0] = 1  # Prevent divide-by-zero

    if normalize_by_length:
        transition_scores_tensor = sum_logprobs / valid_lengths_safe
    else:
        transition_scores_tensor = sum_logprobs

    # Replace scores where valid_lengths was 0 with -10.0
    zero_mask = valid_lengths == 0
    transition_scores_tensor[zero_mask] = -10.0

    # Convert to list of floats
    return transition_scores_tensor.tolist()


@torch.no_grad()
def generated_sequence_nll(model, sequences, input_ids, attention_mask=None):
    """Mean negative log-likelihood per generated token, for each row of T5 ``generate()`` output.

    ``generate()`` starts every row with the decoder start token. It is dropped before the row is
    used as ``labels``: T5 shifts labels right and prepends that token itself, so keeping it would
    condition the decoder on two start tokens. Positions after the first EOS (padding) are not scored.
    """
    nlls = []
    for row in sequences:
        labels = row[1:].unsqueeze(0).clone()  # drop the decoder start token
        is_eos = (labels == model.config.eos_token_id).long()
        labels[(is_eos.cumsum(dim=1) - is_eos) > 0] = -100  # positions after the first EOS
        nlls.append(model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss.item())
    return nlls


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Mol2Pro inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input_file', default='../inference/random_smiles2.txt', type=str,
                        help='File with the input molecule SMILES')
    parser.add_argument('--model_path', default='./output03/checkpoint-60000', type=str,
                        help='Path to model to load')
    parser.add_argument('--tokenizer_aa', type=str, required=True,
                        help='Path or Hugging Face repo ID for amino acid tokenizer')
    parser.add_argument('--tokenizer_mol', type=str, required=True,
                        help='Path or Hugging Face repo ID for SMILES tokenizer')
    parser.add_argument('--tokenizer_aa_subfolder', type=str, default="aa",
                        help='Subfolder in tokenizer_aa repo (e.g. tokenizer_aa when both tokenizers live in model repo)')
    parser.add_argument('--tokenizer_mol_subfolder', type=str, default="smiles",
                        help='Subfolder in tokenizer_mol repo (e.g. tokenizer_mol when both tokenizers live in model repo)')
    parser.add_argument('--top_k', default=15, type=int, help='K for top-k sampling')
    parser.add_argument('--seed', default=0, type=int, help='Random seed for reproducibility')
    parser.add_argument('--output_folder', default='fastas', type=str,
                        help='Folder for saving results')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load protein tokenizer
    def load_tokenizer(arg, subfolder=None):
        if Path(arg).exists():
            return AutoTokenizer.from_pretrained(arg)
        else:
            return AutoTokenizer.from_pretrained(arg, subfolder=subfolder)

    tokenizer_aa = load_tokenizer(args.tokenizer_aa, subfolder=args.tokenizer_aa_subfolder)
    tokenizer_mol = load_tokenizer(args.tokenizer_mol, subfolder=args.tokenizer_mol_subfolder)

    # Load data
    smiles_list = set()
    gt_dict = None

    def read_entry(entry):
        rxn = entry["translation"].get("Reaction") or entry["translation"].get("Canonical Smiles")
        gt_prot = entry["translation"].get("Sequence") or entry["translation"].get("Protein sequence")
        smiles_list.add(rxn)
        if isinstance(gt_prot, list):
            for seq in gt_prot:
                gt_dict[rxn].append({"Sequence": seq})
        else:
            gt_dict[rxn].append({"Sequence": gt_prot})

    if 'json' in args.input_file:
        gt_dict = defaultdict(list)
        with open(args.input_file) as f:
            text = f.read().strip()
            try:
                arr = json.loads(text)
                # assume arr is a list
                for entry in arr:
                    read_entry(entry)
            except json.JSONDecodeError:
                # fallback: JSON lines
                f.seek(0)
                for line in f:
                    if line.strip():
                        read_entry(json.loads(line))
    else:
        with open(args.input_file, 'r') as input_file:
            for line in input_file:
                smiles_list.add(line.strip())

    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    model.eval()
    model.to(device)
    print('Model loaded')

    pad_token_id = (
        tokenizer_aa.pad_token_id
        if tokenizer_aa.pad_token_id is not None
        else tokenizer_aa.convert_tokens_to_ids(tokenizer_aa.eos_token)
    )
    print(f'Using {pad_token_id} as pad_token_id from AA tokenizer')

    molecule_gt_json = {}
    ppxt_avg = []

    print(f'Generating for {len(smiles_list)} inputs')

    for index, smiles in enumerate(smiles_list):
        with torch.no_grad():
            input_ids = tokenizer_mol(smiles, return_tensors="pt").input_ids.to(device)
            encoder_attention_mask = (input_ids != tokenizer_mol.pad_token_id).long().to(device)
            if tokenizer_mol.pad_token_id is None:
                encoder_attention_mask = torch.ones_like(input_ids, device=device)

            print(f'Generating for {smiles}')

            # top_k = Choose at random from the first K tokens (weigthed by softmax score)
            # num_return_sequences = The number of independently computed returned sequences for each element in the batch.
            outputs_dict = model.generate(
                input_ids,
                top_k=args.top_k,
                repetition_penalty=1.2,
                max_length=1024,
                do_sample=True,
                num_return_sequences=25,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
            )
            greedy_outputs = model.generate(
                input_ids,
                max_length=1024,
                do_sample=False,
                use_cache=True,
            )

        outputs = outputs_dict.sequences
        transition_scores = compute_transition_scores(
            sequences=outputs,
            scores=outputs_dict.scores,
            pad_token_id=pad_token_id,
            normalize_by_length=True,
        )
        ppxt_samples = [np.exp(-s) for s in transition_scores]

        sequences = [tokenizer_aa.decode(o, skip_special_tokens=True) for o in outputs]

        # Greedy perplexity: teacher-forced NLL of the greedy sequence (see generated_sequence_nll)
        nll = generated_sequence_nll(model, greedy_outputs, input_ids, encoder_attention_mask)[0]

        # Filename of the output fasta
        filename = f'{args.output_folder}/output_topk{args.top_k}_file-{index}.fasta'
        os.makedirs(args.output_folder, exist_ok=True)

        with open(filename, 'w') as fn:
            for idx, seq in enumerate(sequences):
                fn.write(f">{idx}\n{seq}\n")
        # Register per-FASTA metadata (SMILES, perplexities, optional GT)
        molecule_gt_json[filename] = {'fasta_perplexity': ppxt_samples, 'Smiles': smiles, 'Perplexity_greedy': np.exp(nll)}
        if gt_dict is not None and smiles in gt_dict:
            molecule_gt_json[filename]['GT'] = [gt['Sequence'] for gt in gt_dict[smiles]]
        ppxt_avg.append(np.exp(nll))

    # Print ppxt statistics
    ppxt_mean = np.mean(ppxt_avg)
    ppxt_std = np.std(ppxt_avg)
    print(f'Avg Perplexity greedy decoding: {ppxt_mean} +- {ppxt_std}')
    molecule_gt_json['Greedy_Perplexity'] = {'Mean': ppxt_mean, 'STD': ppxt_std}

    os.makedirs(args.output_folder, exist_ok=True)
    inference_metadata_path = os.path.join(args.output_folder, 'inference_metadata.json')
    with open(inference_metadata_path, 'w') as f:
        json.dump(molecule_gt_json, f, indent=4)
    print(f"Metadata written to {inference_metadata_path}")

    run_metadata_path = os.path.join(args.output_folder, 'generation_parameters.json')
    run_metadata = {
        'generation_date': datetime.now().isoformat(),
        'model_path': args.model_path,
        'tokenizer_aa': args.tokenizer_aa,
        'tokenizer_mol': args.tokenizer_mol,
        'input_file': args.input_file,
        'top_k': args.top_k,
        'seed': args.seed,
    }
    with open(run_metadata_path, 'w') as f:
        json.dump(run_metadata, f, indent=4)
    print(f"Run metadata written to {run_metadata_path}")
