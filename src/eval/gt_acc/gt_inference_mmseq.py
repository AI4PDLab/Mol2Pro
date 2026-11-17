"""
In output_folder_gt it saves the GT fastas and a folder_names.json that later allows mapping mmseqs to sequence
If AA_binding exists in dataset, also creates pocket versions in output_folder_gt_pocket
"""
if __name__ == '__main__':
    import argparse
    import json
    import re
    from collections import defaultdict
    import os
    import subprocess
    from datasets import load_dataset

    parser = argparse.ArgumentParser(description='Prepare GT FASTA files and run MMseqs against generated proteins',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_file', type=str, help='File with the input molecule SMILES for inference')
    parser.add_argument('--inference_folder', required=True, type=str, help='Folder containing fastas from inference')
    parser.add_argument('--dataset_json', required=True, type=str, help='Dataset JSON/JSONL containing ground-truth proteins')
    parser.add_argument('--top_k', default=15, type=int, help='K for top-k sampling')
    parser.add_argument('--output_folder_gt', required=True, type=str, help='Folder for saving results')
    parser.add_argument('--cache_path', default="./tmp_cache", type=str,
                        help='MMSeqs cache')
    parser.add_argument('--mmseq_path', default="", type=str,
                        help='Path to folder containing MMSeqs executable; empty uses mmseqs from PATH')
    parser.add_argument('--translation_key', default='translation', type=str,
                        help='Top-level key containing molecule/protein fields in dataset records')
    parser.add_argument('--protein_key', default='Protein sequence', type=str,
                        help='Primary ground-truth protein sequence field')
    parser.add_argument('--protein_fallback_key', default='Sequence', type=str,
                        help='Fallback ground-truth protein sequence field')
    parser.add_argument('--molecule_key', default='Canonical Smiles', type=str,
                        help='Primary molecule/reaction field')
    parser.add_argument('--molecule_fallback_key', default='Reaction', type=str,
                        help='Fallback molecule/reaction field')
    parser.add_argument('--binding_key', default='AA_binding', type=str,
                        help='Optional residue-binding annotation field')
    args = parser.parse_args()

    input_metadata_file = [f for f in os.listdir(args.inference_folder) if 'molecule_input_metadata' in f]
    if len(input_metadata_file) != 0:
        print("Reading molecule to output protein file correspondences from metadata")
        metadata_path = os.path.join(args.inference_folder, input_metadata_file[0])
        input_metadata = json.load(open(metadata_path))
        smiles_list = list(input_metadata.values())
        input_metadata_by_basename = {}
        for path, smi in input_metadata.items():
            base = os.path.basename(path)
            if base not in input_metadata_by_basename:
                input_metadata_by_basename[base] = smi
    else:
        print("Reading inputs from input .txt file ")
        input_metadata = None
        input_metadata_by_basename = None
        input_file = open(args.input_file)
        smiles_list = []
        for line in input_file:
            smiles_list.append(line.strip())

    assert (args.input_file is not None or input_metadata is not None)

    # Step 1: Extract unique SMILES and sanitize for valid folder names
    inference_unique_smiles = sorted(set(smiles_list))
    smiles_to_safe_name = {smiles: re.sub(r'[^a-zA-Z0-9_-]', '_', smiles) for smiles in inference_unique_smiles}

    # Step 2: Create directories for each SMILES in inference inputs
    os.makedirs(args.output_folder_gt, exist_ok=True)
    output_folder_gt_pocket = f"{args.output_folder_gt}_pocket"

    print(
        f'Create folders for {len(inference_unique_smiles)} unique molecules from the original {len(smiles_list)} list')
    idx = 0
    global_correspondence_dict = {}
    for smiles in inference_unique_smiles:
        safe_name = smiles_to_safe_name[smiles]
        max_len = 50
        safe_name = f'{idx}_' + safe_name[:max_len]  # Add index to avoid identical ones
        folder_path = os.path.join(args.output_folder_gt, safe_name)
        os.makedirs(folder_path, exist_ok=True)

        if not os.path.exists(os.path.join(folder_path, f"{safe_name}.fasta")):
            metadata_path = os.path.join(folder_path, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump({"molecule": smiles}, f, indent=4)

        global_correspondence_dict[smiles] = safe_name
        print(f"Storing in global_correspondence_dict: {smiles} -> {safe_name}")
        idx += 1

    del smiles_to_safe_name  # The correspondence is not valid anymore

    if len(global_correspondence_dict) > 0:
        print('Write folder_names.json')
        global_dict_path = os.path.join(args.output_folder_gt, 'folder_names.json')
        with open(global_dict_path, "w") as f:
            json.dump(global_correspondence_dict, f, indent=4)
    else:
        print('No new folders added, reusing folder_names.json')

    # Step 3: Load the dataset to extract GTs
    print(f'Loading {args.dataset_json} ...')
    extension = args.dataset_json.split(".")[-1]
    raw_dataset = load_dataset('json', data_files=args.dataset_json, cache_dir=args.cache_path, streaming=True,
                               split='train')
    print('Loaded')

    # Step 4: Collect protein sequences for each molecule (0-based AA_binding expected)
    from collections import defaultdict

    # For each SMILES: protein -> aa_binding (or None)
    protein_bindings = defaultdict(dict)  # Dict[str, Dict[str, Optional[List[int]]]]
    has_aa_binding = False

    print('Inference molecules:')
    print(inference_unique_smiles)

    mols_read = 0
    for example in raw_dataset:
        tr = example.get(args.translation_key, {})
        if not tr:
            continue

        protein = tr.get(args.protein_key)
        if protein is None and args.protein_fallback_key:
            protein = tr.get(args.protein_fallback_key)

        molecule = tr.get(args.molecule_key)
        if molecule is None and args.molecule_fallback_key:
            molecule = tr.get(args.molecule_fallback_key)

        molecules = molecule if isinstance(molecule, (list, tuple)) else [molecule]
        proteins = protein if isinstance(protein, (list, tuple)) else [protein]
        if not any(m in inference_unique_smiles for m in molecules) or all(p is None for p in proteins):
            mols_read += 1
            continue

        aa_binding = tr.get(args.binding_key, None)
        if isinstance(aa_binding, list) and len(aa_binding) > 0:
            has_aa_binding = True
        else:
            aa_binding = None

        # Keep first seen binding per protein (stable), ignore subsequent conflicting ones
        for mol in molecules:
            if mol not in inference_unique_smiles:
                continue
            for prot in proteins:
                if prot is None:
                    continue
                if prot not in protein_bindings[mol]:
                    protein_bindings[mol][prot] = aa_binding

        mols_read += 1

    print(f'{mols_read} molecules scanned from dataset')

    # Step 5: Write protein (and pocket) FASTAs deterministically with aligned seq_i
    with open(os.path.join(args.output_folder_gt, 'folder_names.json')) as f:
        global_correspondence_dict = json.load(f)

    if has_aa_binding:
        os.makedirs(output_folder_gt_pocket, exist_ok=True)
        # mirror mapping for interpreter
        with open(os.path.join(output_folder_gt_pocket, 'folder_names.json'), "w") as f:
            json.dump(global_correspondence_dict, f, indent=4)

    print(f'Write the {len(protein_bindings.items())} GT pairs to {args.output_folder_gt}')
    for smiles, prot2bind in protein_bindings.items():
        safe_name = global_correspondence_dict[smiles]
        folder_path = os.path.join(args.output_folder_gt, safe_name)
        os.makedirs(folder_path, exist_ok=True)

        # Deterministic order for both files
        proteins_sorted = sorted(prot2bind.keys())

        # Full GT
        gt_fasta_path = os.path.join(folder_path, f"{safe_name}.fasta")
        if not os.path.exists(gt_fasta_path):
            with open(gt_fasta_path, 'w') as fasta_file:
                for i, p in enumerate(proteins_sorted):
                    fasta_file.write(f">seq_{i}\n{p}\n")
        else:
            print(f'{gt_fasta_path} already exists')

        # Pocket GT (same seq_i; write only entries that have binding)
        if has_aa_binding:
            folder_path_pocket = os.path.join(output_folder_gt_pocket, safe_name)
            os.makedirs(folder_path_pocket, exist_ok=True)
            pocket_fasta_path = os.path.join(folder_path_pocket, f"{safe_name}.fasta")

            if not os.path.exists(pocket_fasta_path):
                wrote_any = False
                with open(pocket_fasta_path, 'w') as fasta_file:
                    for i, p in enumerate(proteins_sorted):
                        bind = prot2bind[p]
                        if bind:
                            # 0-based indices; clamp to protein bounds just in case
                            lo = max(0, min(bind))
                            hi = min(len(p) - 1, max(bind))
                            if lo <= hi:
                                pocket_seq = p[lo:hi + 1]
                                fasta_file.write(f">seq_{i}\n{pocket_seq}\n")
                                wrote_any = True
                if not wrote_any:
                    os.remove(pocket_fasta_path)  # avoid empty pocket files
            else:
                print(f'{pocket_fasta_path} already exists')

    # Step 6: List all inference FASTA files in inference_folder
    inference_fastas = [f for f in os.listdir(args.inference_folder) if f.endswith('.fasta')]
    fasta_pattern = re.compile(r'_file-(\d+)\.fasta')

    # Step 7: Process each inference FASTA
    for fasta_file in inference_fastas:
        match = fasta_pattern.search(fasta_file)
        if not match:
            continue  # Skip files that do not match the expected pattern

        idx = int(match.group(1))  # Extract index

        if input_metadata is not None:
            fasta_file_path = os.path.join(args.inference_folder, fasta_file)
            original_smiles = input_metadata.get(fasta_file_path)
            if original_smiles is None and input_metadata_by_basename is not None:
                original_smiles = input_metadata_by_basename.get(fasta_file)
            if original_smiles is None:
                print(f"Warning: Missing metadata for {fasta_file_path}. Skipping...")
                continue
        else:
            if idx >= len(smiles_list):
                print(f"Warning: Index {idx} out of range for SMILES list.")
                continue
            original_smiles = smiles_list[idx]  # Retrieve the original SMILES

        print(f'File {fasta_file} with index {idx} corresponds to {original_smiles}')
        safe_name = global_correspondence_dict[original_smiles]

        # Process regular GT
        gt_folder = os.path.join(args.output_folder_gt, safe_name)
        if not os.path.exists(gt_folder):
            print(f"Warning: GT folder {gt_folder} not found for index {idx}. Skipping...")
            continue

        # Locate GT FASTA file
        if not f"{safe_name}.fasta" in os.listdir(gt_folder):
            print(f"Warning: No GT FASTA file found in {gt_folder}. Skipping...")
            continue

        gt_fasta = os.path.join(gt_folder, f"{safe_name}.fasta")

        # Paths for MMseqs
        inference_fasta_path = os.path.join(args.inference_folder, fasta_file)
        out_path = os.path.join(args.inference_folder, "gt_mmseqs")
        out_file = os.path.join(out_path, f"mmseqs_{safe_name}")
        tmp_folder = args.cache_path

        os.makedirs(tmp_folder, exist_ok=True)
        os.makedirs(out_path, exist_ok=True)

        # MMseqs command execution
        mmseqs_cmd = [
            os.path.join(args.mmseq_path, "mmseqs"),
            "easy-search",
            inference_fasta_path,
            gt_fasta,
            out_file,
            tmp_folder,
            "--min-seq-id", "0.0",
            "--format-output", "query,target,fident,alnlen,qlen,tlen",
            "-c", "0.3",  # require >=30% coverage …
            "--cov-mode", "0"  # … on BOTH query AND target
        ]

        print(f"Running MMseqs for {fasta_file} vs {gt_fasta} ...")
        subprocess.run(mmseqs_cmd, check=True)

        # Process pocket GT if it exists
        if has_aa_binding:
            gt_folder_pocket = os.path.join(output_folder_gt_pocket, safe_name)
            if os.path.exists(gt_folder_pocket) and f"{safe_name}.fasta" in os.listdir(gt_folder_pocket):
                gt_fasta_pocket = os.path.join(gt_folder_pocket, f"{safe_name}.fasta")

                out_path_pocket = os.path.join(args.inference_folder, "gt_mmseqs_pocket")
                out_file_pocket = os.path.join(out_path_pocket, f"mmseqs_{safe_name}")

                os.makedirs(out_path_pocket, exist_ok=True)

                mmseqs_cmd_pocket = [
                    os.path.join(args.mmseq_path, "mmseqs"),
                    "easy-search",
                    inference_fasta_path,
                    gt_fasta_pocket,
                    out_file_pocket,
                    tmp_folder,
                    "--min-seq-id", "0.0",
                    "--format-output", "query,target,fident,alnlen,qlen,tlen",
                    "-c", "0.3",
                    "--cov-mode", "0"
                ]

                print(f"Running MMseqs for {fasta_file} vs {gt_fasta_pocket} (pocket) ...")
                subprocess.run(mmseqs_cmd_pocket, check=True)

    print("MMseqs searches completed.")
