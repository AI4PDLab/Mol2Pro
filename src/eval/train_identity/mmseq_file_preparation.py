import os
import json
import pickle
from pathlib import Path


def collate_fasta(input_folder, output_file):
    if not input_folder:
        print("No input folder provided for FASTA files. Skipping collation.")
        return

    fasta_files = list(Path(input_folder).glob("*.fasta"))
    if not fasta_files:
        print("No FASTA files found in the provided folder. Skipping collation.")
        return

    # Dictionary to store mapping: numeric_id -> original_identifier
    id_mapping = {}
    # Dictionary to store sequence correspondence: index -> {sequence, fasta}
    sequence_correspondence = {}

    with open(output_file, 'w') as outfile:
        index = 0  # Start from 0 instead of 1
        for fasta_file in fasta_files:
            with open(fasta_file, 'r') as infile:
                current_sequence = ""
                current_original_id = None

                for line in infile:
                    if line.startswith('>'):
                        # If we have a previous sequence, save it
                        if current_original_id is not None and current_sequence:
                            sequence_correspondence[index - 1] = {
                                'sequence': current_sequence.strip(),
                                'fasta': fasta_file.name
                            }

                        # Extract original identifier (remove '>' and strip whitespace)
                        original_id = line[1:].strip()
                        current_original_id = original_id
                        current_sequence = ""

                        # Store mapping
                        id_mapping[index] = original_id
                        # Write new numeric identifier
                        outfile.write(f'>{index}\n')
                        index += 1
                    else:
                        # Accumulate sequence lines
                        current_sequence += line.strip()
                        outfile.write(line)

                # Don't forget the last sequence in the file
                if current_original_id is not None and current_sequence:
                    sequence_correspondence[index - 1] = {
                        'sequence': current_sequence.strip(),
                        'fasta': fasta_file.name
                    }

    # Save metadata file (now starting from 0)
    metadata_file = output_file.replace('.fasta', '_metadata.pkl')
    with open(metadata_file, 'wb') as f:
        pickle.dump(id_mapping, f)

    # Save sequence correspondence file
    correspondence_file = output_file.replace('.fasta', '_sequence_correspondence.json')
    with open(correspondence_file, 'w') as f:
        json.dump(sequence_correspondence, f, indent=2)

    print(f"Collated FASTA written to {output_file}")
    print(f"Metadata mapping saved to {metadata_file}")
    print(f"Sequence correspondence saved to {correspondence_file}")


def json_to_fasta(json_file, output_file):
    if not json_file:
        print("No JSON file provided for conversion. Skipping JSON to FASTA.")
        return

    try:
        print(f"Writing Train dataset fasta and correspondence...")
        # Dictionary to store mapping: numeric_id -> original_identifier
        id_mapping = {}
        # Dictionary to store sequence correspondence: index -> {sequence, original_id}
        sequence_correspondence = {}

        with open(json_file, 'r') as infile, open(output_file, 'w') as outfile:
            index = 0  # Start from 0 instead of 1
            prot_key = "Protein sequence"
            done = False
            for line in infile:
                try:
                    record = json.loads(line)
                    if not done and 'Reaction' in record.keys():
                        prot_key = 'Sequence'
                        done = True
                    try:
                        protein_seq = record["translation"][prot_key]
                    except:
                        prot_key = "Sequence"
                        protein_seq = record["translation"][prot_key]

                    # Try to extract original identifier from the record
                    # You may need to adjust this based on your JSON structure
                    original_id = f"json_record_{index}"  # Default fallback
                    if 'id' in record:
                        original_id = str(record['id'])
                    elif 'name' in record:
                        original_id = str(record['name'])
                    elif 'identifier' in record:
                        original_id = str(record['identifier'])

                    # Store mapping
                    id_mapping[index] = original_id
                    # Store sequence correspondence
                    sequence_correspondence[index] = {
                        'sequence': protein_seq,
                        'original_id': original_id
                    }

                    outfile.write(f'>{index}\n')
                    outfile.write(f'{protein_seq}\n')
                    index += 1
                except (KeyError, json.JSONDecodeError) as e:
                    print(f"Skipping invalid JSON record: {line.strip()}\n{e}")

        # Save metadata file
        metadata_file = output_file.replace('.fasta', '_metadata.pkl')
        with open(metadata_file, 'wb') as f:
            pickle.dump(id_mapping, f)

        # Save sequence correspondence file for training data
        correspondence_file = output_file.replace('.fasta', '_sequence_correspondence.json')
        with open(correspondence_file, 'w') as f:
            json.dump(sequence_correspondence, f, indent=2)

        print(f"Metadata mapping saved to {metadata_file}")
        print(f"Training sequence correspondence saved to {correspondence_file}")

    except FileNotFoundError:
        print(f"JSON file {json_file} not found. Skipping JSON to FASTA.")
    print(f"FASTA file from JSON written to {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Collate FASTA files and convert JSON to FASTA")
    parser.add_argument('--fasta_folder', type=str, help="Path to the folder containing FASTA files")
    parser.add_argument('--json_file', type=str, help="Path to the JSON file")
    parser.add_argument('--collated_fasta', type=str, default='collated.fasta',
                        help="Output FASTA file for collated sequences")

    args = parser.parse_args()

    if args.fasta_folder:
        subfolder_name = 'mmseqs'
        output_dir = os.path.join(args.fasta_folder, subfolder_name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_file = os.path.join(output_dir, args.collated_fasta)
        print(f'Creating {output_file} ...')
        collate_fasta(args.fasta_folder, output_file)

    if args.json_file:
        output_dir = args.json_file.replace('.json', '') + '.fasta'
        print(f'Creating {output_dir} ...')
        if not os.path.exists(output_dir):
            json_to_fasta(args.json_file, output_dir)
        else:
            print(f'Skipping, it already exists')


if __name__ == "__main__":
    main()