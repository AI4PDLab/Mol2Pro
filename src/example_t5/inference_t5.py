"""T5 (and related) sequence generation for the Mol2ProBench example workflow.

Reads SMILES / reaction / benchmark JSON inputs, runs the model, and writes
FASTA files plus `molecule_input_metadata.json` and
`molecule_gt-input_metadata.json` in the layout expected by `eval/*` scripts.

Evaluation under `eval/` assumes this step has already produced those FASTAs;
it does not run model inference.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, \
    set_seed  # T5ForConditionalGeneration
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

    if 'gatgpt' in args.model_path.lower():
        GNN = True
        print('Graph data mode')
    else:
        GNN = False
        print('SMILES/SELFIES data mode')

    # Load protein tokenizer (Hugging Face only)
    tokenizer_aa = AutoTokenizer.from_pretrained(args.tokenizer_aa)

    # Load molecule tokenizer
    if GNN:
        tokenizer_mol = None
    else:
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
    def peek_model_type(model_dir: str) -> str | None:
        cfg_path = os.path.join(model_dir, "config.json")
        try:
            with open(cfg_path, "r") as f:
                return json.load(f).get("model_type", None)
        except Exception:
            return None


    dec_only = False


    use_cache = True  # KV cache for speed, not supported in custom model
    if GNN:
        from transformers import GPT2Config, Trainer
        from models import GATGPT2Config, GATGPT2
        from torch_geometric.data import Batch, Data

        config = GATGPT2Config.from_pretrained(args.model_path)

        # Load model weights
        model = GATGPT2.from_pretrained(args.model_path, config=config)

        model.eval()
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        try:
            print('Attempt Seq2Seq model load... ')
            loaded = False
            mtype = peek_model_type(args.model_path)

            if mtype == "custom_encoder_decoder":
                use_cache = True  # KV cache for speed, not supported in custom model
                from transformers.modeling_utils import load_state_dict
                from models import create_custom_encoder_decoder_model

                # Load State Dict
                if os.path.exists(os.path.join(args.model_path, "pytorch_model.bin")):
                    wfile = os.path.join(args.model_path, "pytorch_model.bin")
                else:
                    wfile = os.path.join(args.model_path, "model.safetensors")

                sd = load_state_dict(wfile)
                # Infer vocab size from checkpoint (robust to key variations)
                cfg_path = os.path.join(args.model_path, "config.json")
                cfg = json.load(open(cfg_path))

                vocab_size = cfg["vocab_size"]                          # Joint vocab size (encoder in my implementations)
                dec_vocab_size = cfg["decoder_config"]["vocab_size"]
                dec_eos = cfg["eos_token_id"]
                assert dec_eos == tokenizer_aa.eos_token_id, f"custom_encoder_decoder eos ({dec_eos}) is not the same as the tokenizer's ({tokenizer_aa.eos_token})"

                # Load model config
                cfg_path = os.path.join(args.model_path, "config.json")
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)

                enc_type = cfg.get("encoder_type", "t5")
                dec_type = cfg.get("decoder_type", "t5")

                enc_cfg = cfg.get("encoder_config") or {}
                dec_cfg = cfg.get("decoder_config") or {}

                # HF usually stores the origin checkpoint here
                enc_name = enc_cfg.get("_name_or_path") or enc_cfg.get("name_or_path")
                dec_name = dec_cfg.get("_name_or_path") or dec_cfg.get("name_or_path") or enc_name

                if enc_name is None or dec_name is None:
                    raise ValueError(
                        "custom_encoder_decoder checkpoint missing encoder/decoder '_name_or_path' in config.json. "
                        "Fix saving: set encoder_config._name_or_path and decoder_config._name_or_path (or add explicit fields)."
                    )

                pad_id = cfg.get("pad_token_id")
                if pad_id is None:
                    print("WARNING: Setting PAD token ID as the Tokenizer AA one")
                    pad_id = tokenizer_aa.pad_token_id if tokenizer_aa.pad_token_id is not None else 0
                pad_id = int(pad_id)

                dec_start = cfg.get("decoder_start_token_id")
                if dec_start is None:
                    print("WARNING: Setting Dec_Start token ID as the PAD one")
                    dec_start = pad_id
                dec_start = int(dec_start)

                # Build the *same* architecture as training, then load weights
                model = create_custom_encoder_decoder_model(
                    encoder_type=enc_type,
                    decoder_type=dec_type,
                    encoder_name_or_path=enc_name,
                    decoder_name_or_path=dec_name,
                    train_from_scratch=False,
                    freeze_encoder=False,
                    freeze_decoder=False,
                    vocab_size=vocab_size,
                    decoder_vocab_size=dec_vocab_size,
                    pad_token_id=pad_id,
                    decoder_start_token_id=dec_start,
                    eos_token_id=dec_eos,
                    eager_att=True  # Avoid KV-cache + SDPA attention bug
                ).cuda()
                
                # Load weights
                missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)

                print("[custom_encoder_decoder] Loaded via factory + state_dict")
                if missing_keys:
                    print(f"[custom_encoder_decoder] missing keys (first 30): {missing_keys[:30]}")
                if unexpected_keys:
                    print(f"[custom_encoder_decoder] unexpected keys (first 30): {unexpected_keys[:30]}")

                # Safety (prevents generation crashes)
                if getattr(model.config, "decoder_start_token_id", None) is None:
                    model.config.decoder_start_token_id = pad_id
                    print(f"WARNING: Inference code set decoder_start_token_id to PAD_ID {model.config.decoder_start_token_id}")
                if getattr(model.config, "pad_token_id", None) is None:
                    model.config.pad_token_id = pad_id
                    print(f"WARNING: Inference code set pad_token_id to PAD_ID {model.config.pad_token_id}")


                model.eval()
                loaded = True


            if not loaded:
                model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).cuda()

        except Exception as e:
            print(e)
            print('Attempt CausalLM model load... ')
            model = AutoModelForCausalLM.from_pretrained(args.model_path).cuda()
            model.config.eos_token_id = tokenizer_mol.eos_token_id
            model.config.pad_token_id = tokenizer_mol.pad_token_id
            print(
                f"Set `eos_token_id` to {tokenizer_mol.eos_token_id} and `pad_token_id` to {tokenizer_mol.pad_token_id}.")
            dec_only = True
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
            if GNN:
                from build_tokenized_dataset import convert_smiles_to_graph

                node_feats, edge_index, edge_feats = convert_smiles_to_graph(smiles)
                node_feats_tensor = torch.tensor(node_feats, dtype=torch.float, device=device)
                edge_index_tensor = torch.tensor(edge_index, dtype=torch.long, device=device).T.contiguous()
                edge_feats_tensor = torch.tensor(edge_feats, dtype=torch.float, device=device)

                # Input to decoder is only bos
                start_token = tokenizer_aa.bos_token_id or tokenizer_aa.convert_tokens_to_ids("▁")  # fallback to the space which is always appended by our tokenizer
                text_input_ids = torch.tensor([[start_token]], dtype=torch.long, device=device)

                input_ids = {
                    "graph_node_feats": node_feats_tensor,  # shape (N, 3)
                    "graph_edge_index": edge_index_tensor,  # shape (2, E)
                    "graph_edge_feats": edge_feats_tensor,  # shape (E, 2)
                    "batch": torch.full((len(node_feats),), 0, dtype=torch.long, device=device),  # shape (N,)
                    "input_ids": text_input_ids
                }

            else:
                input_ids = tokenizer_mol(smiles, return_tensors="pt").input_ids.to(device="cuda")
            if not GNN:
                print(f'Generating for {smiles} (input ids: {input_ids})')
            else:
                print(f'Generating for {smiles}')

            # top_k = Choose at random from the first K tokens (weigthed by softmax score)
            # num_return_sequences = The number of independently computed returned sequences for each element in the batch.

            if dec_only:
                attention_mask = torch.ones_like(input_ids).cuda()
                outputs_dict = model.generate(input_ids, top_k=args.top_k, attention_mask = attention_mask, repetition_penalty=1.2,
                                              max_length=1024, do_sample=True, temperature=args.temperature,
                                              num_return_sequences=25, return_dict_in_generate=True, output_scores=True)
                greedy_outputs = model.generate(input_ids, attention_mask = attention_mask, max_length=1024, do_sample=False)

            else:
                outputs_dict = model.generate(input_ids, top_k=args.top_k,repetition_penalty=1.2, max_length=1024,
                                         do_sample=True, temperature=args.temperature, num_return_sequences=25,
                                         return_dict_in_generate=True, output_scores=True, use_cache=use_cache)
                greedy_outputs = model.generate(input_ids, max_length=1024, do_sample=False, use_cache=use_cache)

            outputs = outputs_dict.sequences
            transition_scores = compute_transition_scores(
                sequences=outputs,
                scores=outputs_dict.scores, # optional, if your model outputs logits
                pad_token_id=tokenizer_aa.pad_token_id,
                normalize_by_length=True
            )
            ppxt_samples = [np.exp(-score) for score in transition_scores]  # perplexity per sequence

            sequences = [tokenizer_aa.decode(output, skip_special_tokens=True) for output in outputs]

            # Model forward for greedy perplexity
            g_labels = greedy_outputs.clone()
            g_labels[g_labels == pad_token_id] = -100  # ignore pad tokens in loss
            forward_outputs = model(input_ids=input_ids, attention_mask=attention_mask if dec_only else None,
                        labels=g_labels)
            nll = forward_outputs.loss.item()  # Greedy NLL. Normalised per out length


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
