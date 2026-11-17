"""T5 sequence generation for the Mol2ProBench example workflow.

Reads SMILES / reaction / benchmark JSON inputs, runs the model, and writes
FASTA files plus `molecule_input_metadata.json` and
`molecule_gt-input_metadata.json` in the layout expected by `eval/*` scripts.

Evaluation under `eval/` assumes this step has already produced those FASTAs;
it does not run model inference.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
import argparse
import os
import torch
import json
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
from datetime import datetime


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
    transition_scores = transition_scores_tensor.tolist()

    return transition_scores

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

    parser = argparse.ArgumentParser(description='Mol2ProBench T5 example inference',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_file', required=True, type=str,
                        help='File with the input molecule SMILES')
    parser.add_argument('--model_path', required=True, type=str, help='Path to model to load')
    parser.add_argument('--tokenizer_aa',
                        required=True, type=str,
                        help='Path to amino acid tokenizer')
    parser.add_argument('--tokenizer_mol',
                        required=True, type=str,
                        help='Path to SMILES tokenizer')
    parser.add_argument('--top_k',
                        default=15,type=int,
                        help='K for top-k sampling')
    parser.add_argument('--temperature',
                        default=1.0, type=float,
                        help='Sampling temperature (only used when do_sample=True).')
    parser.add_argument('--seed',
                        default=0,type=int, help='Random seed for reproducibility')
    parser.add_argument('--output_folder', default='fastas', type=str, help='Folder for saving results')
    parser.add_argument('--translation_key', default='translation', type=str,
                        help='Top-level key containing molecule/protein fields in JSON inputs')
    parser.add_argument('--input_key', default='Reaction', type=str,
                        help='Primary molecule/reaction field in JSON inputs')
    parser.add_argument('--input_fallback_key', default='Canonical Smiles', type=str,
                        help='Fallback molecule field in JSON inputs')
    parser.add_argument('--protein_key', default='Sequence', type=str,
                        help='Primary protein sequence field in JSON inputs')
    parser.add_argument('--protein_fallback_key', default='Protein sequence', type=str,
                        help='Fallback protein sequence field in JSON inputs')
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load protein tokenizer (Hugging Face only)
    tokenizer_aa = AutoTokenizer.from_pretrained(args.tokenizer_aa)

    # Load molecule tokenizer
    tokenizer_mol = AutoTokenizer.from_pretrained(args.tokenizer_mol)

    # Load data
    smiles_list = set()
    gt_dict = None
    def read_entry(entry):

        translation = entry.get(args.translation_key, {})
        rxn = translation.get(args.input_key) or translation.get(args.input_fallback_key)
        gt_prot = translation.get(args.protein_key) or translation.get(args.protein_fallback_key)

        if rxn is None:
            return

        smiles_list.add(rxn)
        """if type(gt_prot) is list:
            gt_dict[rxn].extend(gt_prot)
        else:
            gt_dict[rxn].append(gt_prot)"""
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
                        entry = json.loads(line)
                        read_entry(entry)

    else:
        with open(args.input_file, 'r') as input_file:
            for line in input_file:
                smiles_list.add(line.strip())

    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device)
    model.eval()
    print('Model Loaded')

    molecule_json = {}
    molecule_gt_json = {}
    all_perplexities = []
    pad_token_id = tokenizer_aa.pad_token_id if tokenizer_aa.pad_token_id is not None else tokenizer_aa.convert_tokens_to_ids(
        tokenizer_aa.eos_token)
    print(f'Using {pad_token_id} as pad_token_id from AA tokenizer')
    ppxt_avg = []


    print(f'Generating for {len(smiles_list)} inputs')
    for index,smiles in enumerate(smiles_list):
        sequences=[]
        with torch.no_grad():
            input_ids = tokenizer_mol(smiles, return_tensors="pt").input_ids.to(device)
            print(f'Generating for {smiles} (input ids: {input_ids})')

            # top_k = Choose at random from the first K tokens (weigthed by softmax score)
            # num_return_sequences = The number of independently computed returned sequences for each element in the batch.

            outputs_dict = model.generate(input_ids, top_k=args.top_k, repetition_penalty=1.2, max_length=1024,
                                          do_sample=True, temperature=args.temperature, num_return_sequences=25,
                                          return_dict_in_generate=True, output_scores=True)
            greedy_outputs = model.generate(input_ids, max_length=1024, do_sample=False)

            outputs = outputs_dict.sequences
            transition_scores = compute_transition_scores(
                sequences=outputs,
                scores=outputs_dict.scores, # optional, if your model outputs logits
                pad_token_id=tokenizer_aa.pad_token_id,
                normalize_by_length=True
            )
            ppxt_samples = [np.exp(-score) for score in transition_scores]  # perplexity per sequence

            sequences = [tokenizer_aa.decode(output, skip_special_tokens=True) for output in outputs]

            # Greedy perplexity: teacher-forced NLL of the greedy sequence (see generated_sequence_nll)
            nll = generated_sequence_nll(model, greedy_outputs, input_ids)[0]


            # Filename of the output fasta
            filename = f'{args.output_folder}/output_topk{args.top_k}_file-{index}.fasta'

            if not os.path.exists(args.output_folder):
                os.makedirs(args.output_folder)

            with open(filename, 'w') as fn:
                for idx, seq in enumerate(sequences):
                    fn.write(f">{idx}\n{seq}\n")
                # Register sequence perplexity
                molecule_gt_json[filename] = {'fasta_perplexity':ppxt_samples}
            # Store molecule name
            molecule_json[filename] = smiles
            if gt_dict is not None:
                gt_seqs = [gt['Sequence'] for gt in gt_dict[smiles]]
                molecule_gt_json[filename].update({'Smiles': smiles, 'GT':gt_seqs, 'Perplexity_greedy':np.exp(nll)})
            else:
                molecule_gt_json[filename].update({'Smiles': smiles, 'Perplexity_greedy': np.exp(nll)})
            ppxt_avg.append(np.exp(nll))  # exponentiate to go from NLL to perplexity

    # Print ppxt statistics
    ppxt_mean = np.mean(ppxt_avg)
    ppxt_std = np.std(ppxt_avg)
    print(f'Avg Perplexity greedy decoding: {ppxt_mean} +- {ppxt_std}')
    molecule_gt_json['Greedy_Perplexity'] = {'Mean': ppxt_mean, 'STD': ppxt_std}

    # Save metadata
    os.makedirs(args.output_folder, exist_ok=True)
    metadata_path = os.path.join(args.output_folder, 'molecule_input_metadata.json')
    try:
        with open(metadata_path, 'w') as json_file:
            json.dump(molecule_json, json_file, indent=4)
        print(f"Metadata successfully written to {metadata_path}")

        gt_metadata_path = os.path.join(args.output_folder, 'molecule_gt-input_metadata.json')
        with open(gt_metadata_path, 'w') as json_file:
            json.dump(molecule_gt_json, json_file, indent=4)
        print(f"Metadata successfully written to {gt_metadata_path}")
    except Exception as e:
        print(f"An error occurred while writing to JSON: {e}")

    try:
        run_metadata_path = os.path.join(args.output_folder, 'generation_parameters.json')
        run_metadata = {
            'generation_date': datetime.now().isoformat(),
            'model_path': args.model_path,
            'model_class': model.__class__.__name__,
            'tokenizer_aa': args.tokenizer_aa,
            'tokenizer_mol': args.tokenizer_mol,
            'input_file': args.input_file,
            'top_k': args.top_k,
            'seed': args.seed
        }
        with open(run_metadata_path, 'w') as json_file:
            json.dump(run_metadata, json_file, indent=4)
        print(f"Metadata successfully written to {run_metadata_path}")
    except Exception as e:
        print(f"An error occurred while writing run metadata JSON: {e}")
