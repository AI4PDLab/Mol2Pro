"""Modified from plot_multicsv"""

import pandas as pd
import matplotlib.pyplot as plt
import itertools
import glob
import os
import argparse
import numpy as np
import re

parser = argparse.ArgumentParser(description='Lot pLDDT',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--results_folder',
                    default=None, type=str,
                    help='Path to SMILES tokenizer')
parser.add_argument('--top_k',
                    default=15, type=int,
                    help='To load the correct top_k results')
parser.add_argument('--output_per_file',
                    action='store_true',
                    help='Give an output per csv instead of global one')
args = parser.parse_args()

# Generate CSV with avg pLDDT
from Bio.PDB import PDBParser

def compute_average_b_factor(pdb_file):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('', pdb_file)
        b_factors = [atom.get_bfactor() for atom in structure.get_atoms()]
        if b_factors:
            return sum(b_factors) / len(b_factors)
        else:
            return None
    except Exception as e:
        print(f"Error parsing {pdb_file}: {e}")
        return None

def extract_file_indices(directory, top_k):
    """Extract unique idx values from filenames in the directory"""
    pdb_files = os.listdir(directory)

    # Pattern for new naming format: top_output_topk{top_k}_file-{idx}.fasta_{idx2}.pdb
    new_pattern = rf"top_output_topk{top_k}_file-(\d+)\.fasta_\d+\.pdb"

    # Pattern for old naming format: top_top_{idx}_smiles_results.fasta_{idx2}.pdb
    old_pattern = r"top_top_(\d+)_smiles_results\.fasta_\d+\.pdb"

    indices = set()

    # Try new pattern first
    for filename in pdb_files:
        match = re.match(new_pattern, filename)
        if match:
            indices.add(int(match.group(1)))

    # If no matches with new pattern, try old pattern
    if not indices:
        print("No files found with new naming format, trying old naming format")
        for filename in pdb_files:
            match = re.match(old_pattern, filename)
            if match:
                indices.add(int(match.group(1)))

    return sorted(list(indices))

def process_pdb_files(directory, idx):
    pdb_files = os.listdir(directory)
    file_name = f"top_output_topk{args.top_k}_file-{idx}.fasta_"
    pdb_files = [pdb for pdb in pdb_files if file_name in pdb and '.pdb' in pdb]

    if len(pdb_files) == 0:
        print("No files found, trying old naming format")
        pdb_files = os.listdir(directory)
        old_file_name = f"top_top_{idx}_smiles_results.fasta_"
        pdb_files = [pdb for pdb in pdb_files if old_file_name in pdb and '.pdb' in pdb]
        if len(pdb_files) == 0:
            print('No Files found')
            return None
        else:
            print(f'{len(pdb_files)} sequences for input {idx}')

    results = []

    for pdb_file in pdb_files:
        pdb_path = os.path.join(directory, pdb_file)
        avg_b_factor = compute_average_b_factor(pdb_path)
        if avg_b_factor is not None:
            results.append({'file': pdb_file, 'average_b_factor': avg_b_factor})

    df = pd.DataFrame(results)
    return df

def save_to_csv(df, output_csv):
    df.to_csv(output_csv, index=False)

# Extract actual indices from filenames instead of hardcoded 0-99
available_indices = extract_file_indices(args.results_folder, args.top_k)

for index in available_indices:
    # Output CSV file
    output_csv_file = f'average_plddt_{index}.csv'

    # Process the PDB files and compute the average B-factors
    df = process_pdb_files(args.results_folder, index)
    # Save the results to a CSV file
    if df is not None:
        save_to_csv(df, os.path.join(args.results_folder, output_csv_file))
        print(f'Average B-factors have been saved to {output_csv_file}')


# List of CSV files to load
#csv_files = glob.glob('*.csv')
results_folder = args.results_folder  #'../results/pLDDT/1_alldata'
log_file_path = os.path.join(results_folder, 'plddt_statistics.out') if results_folder else None
if log_file_path:
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        log_file.write('')
csv_files = glob.glob(os.path.join(results_folder,'*.csv'))

# Colors for different datasets
colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])

# Initialize the plot
plt.figure(figsize=(10, 6))

def log_output():
    global all_values
    messages = []
    if max(all_values) > 1:
        threshold_plddt = 70
        scaling = 1
    else:
        threshold_plddt = 0.7
        scaling = 100
    over70 = sum([1 for v in all_values if float(v) >= threshold_plddt])
    perc = 100 * over70 / len(all_values)
    messages.append(f'Number of pLDDT values over 70: {over70}/{len(all_values)} ({perc:.2f}%)')
    mean = np.array(all_values).mean() * scaling
    std = np.array(all_values).std() * scaling
    messages.append(f'Mean: {mean:.2f} STD: {std:.2f}')
    for line in messages:
        print(line)
    if log_file_path:
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            for line in messages:
                log_file.write(f"{line}\n")
    return []  # reset all_values

# Loop through the CSV files and plot the data
all_values = []
for csv_file, color in zip(csv_files, colors):
    # Load the CSV file into a DataFrame
    if os.stat(csv_file).st_size == 1:
        continue
    df = pd.read_csv(csv_file)

    # Plot the data as a scatter plot
    values = df['average_b_factor']
    all_values += values.to_list()
    plt.scatter(range(1, len(df) + 1), values, color=color, label=csv_file.split('_')[-1][:8])
    if args.output_per_file:
        print(f'Value for: {csv_file}')
        all_values = log_output()

if not args.output_per_file:
    log_output()
# Add labels and title
plt.xlabel('PDB File Index')
plt.ylabel('Average pLDDT')

# Set x-axis ticks to show fewer indices
plt.xticks(range(1, len(df) + 1, max(1, len(df) // 10)))  # Show 10 or fewer ticks

# Add legend
plt.legend()

# Display the plot
plt.tight_layout()

# Save the plot to an image file
output_image_file = os.path.join(results_folder, 'average_scatter_plot.png')
plt.savefig(output_image_file)

print(f'The plot has been saved to {output_image_file}')
