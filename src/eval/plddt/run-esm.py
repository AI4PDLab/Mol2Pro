import torch
import re
import gc
import os
import argparse
import glob
from transformers import EsmForProteinFolding

parser = argparse.ArgumentParser(description='Run ESMfold',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fasta_path',
                    default='', type=str,
                    help='Path to FASTA database')
parser.add_argument('--output_path',
                    default='./', type=str,
                    help='Path to output directory')
parser.add_argument('--max_len',
                    default=800, type=int,
                    help='Max sequence length to pass to ESMfold')
parser.add_argument('--batch_size',
                    default=1, type=int,
                    help='Number of sequences to process in a batch')
parser.add_argument('--esmfold_path',
                    default="facebook/esmfold_v1", type=str,
                    help='Optional path to local ESMfold')

args = parser.parse_args()


model = EsmForProteinFolding.from_pretrained(args.esmfold_path)
model = model.eval().cuda()

from transformers.models.esm.openfold_utils import residue_constants

# Try to use OpenFold's own supported alphabet
ALLOWED = None
if hasattr(residue_constants, "restypes_with_x"):
    ALLOWED = set(residue_constants.restypes_with_x)
else:
    # fallback: standard 20 + X (common)
    ALLOWED = set(residue_constants.restypes) | {"X"}

def find_invalid(seq: str):
    seq_u = re.sub(r"\s+", "", seq).upper()
    bad = sorted(set(ch for ch in seq_u if ch not in ALLOWED))
    return seq_u, bad

#model.set_chunk_size(128)

# Define max length
max_len = args.max_len

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

# Function to parse a multi-sequence FASTA file
def parse_fasta(fasta_file):
    with open(fasta_file, 'r') as file:
        sequences = {}
        sequence_id = None
        sequence = []
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if sequence_id is not None:
                    sequences[sequence_id] = ''.join(sequence)
                sequence_id = line[1:].split()[0]
                sequence = []
            else:
                sequence.append(line)
        if sequence_id is not None:
            sequences[sequence_id] = ''.join(sequence)
    return sequences

# Read and parse the fasta file containing multiple sequences
fastafiles = glob.glob(os.path.join(args.fasta_path, '*.fasta'))
print(f'Found {len(fastafiles)} fasta files')

# Process each sequence
bad_counter = 0
for fastafile in fastafiles:
    sequences = parse_fasta(fastafile)
    sequence_names = list(sequences.keys())
    sequence_values = list(sequences.values())

    # Process sequences in batches
    for i in range(0, len(sequence_values), args.batch_size):
        batch_names = sequence_names[i:i + args.batch_size]
        batch_sequences = sequence_values[i:i + args.batch_size]

        # Filter out sequences that are too long or have existing output files
        valid_names = []
        valid_sequences = []
        for name, sequence in zip(batch_names, batch_sequences):
            seq_u, bad = find_invalid(sequence)
            if bad:
                print(f"[INVALID] {fastafile} :: {name} has invalid chars: {bad}")
                print(seq_u)
                bad_counter += 1
                continue
            
            if len(sequence) > max_len:
                print(f"Sequence too long: {name}")
                continue

            output_file = f"top_{fastafile.split('/')[-1]}_{name}.pdb"
            output_file = os.path.join(args.output_path, output_file)

            if os.path.exists(output_file):
                print(f"{output_file} already exists. Skipping inference.")
                continue

            valid_names.append(name)
            valid_sequences.append(seq_u)

        # Skip if no valid sequences in the batch
        if not valid_sequences:
            continue

        # Infer PDBs for the batch
        with torch.no_grad():
            outputs = model.infer_pdbs(valid_sequences)

        # Save each PDB output
        for name, output in zip(valid_names, outputs):
            output_file = f"top_{fastafile.split('/')[-1]}_{name}.pdb"
            output_file = os.path.join(args.output_path, output_file)
            print(f"Saving {output_file}...")
            with open(output_file, "w") as f:
                f.write(output)
                print("Saved")

        # Free up CUDA memory
        del outputs
        torch.cuda.empty_cache()
        gc.collect()
    
print(f"{bad_counter} were ignored because wrong tokens")
print(f'Finished folding')
