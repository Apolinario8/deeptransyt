import os
import gc
import time
import torch
import numpy as np
import pandas as pd
from Bio import SeqIO
import esm
from .auxiliary_functions import remove_ambiguous_aa


def load_sequences(file_path: str) -> pd.DataFrame:
    """ Load sequences from fasta or txt files and return a DataFrame with the ID's and corresponding sequences"""

    sequences = []
    file_extension = file_path.split('.')[-1].lower()
    
    if file_extension in ['fasta', 'faa']:
        for record in SeqIO.parse(file_path, "fasta"):
            sequences.append((record.id, str(record.seq)))
    elif file_extension == 'txt':
        with open(file_path, 'r') as file:
            id = None
            seq = []
            for line in file:
                line = line.strip()
                if line.startswith('>'):
                    if id is not None:
                        sequences.append((id, ''.join(seq)))
                    id = line[1:].strip()
                    seq = []
                else:
                    seq.append(line)
            if id is not None:
                sequences.append((id, ''.join(seq)))
    else:
        raise ValueError("Unsupported file format. Supported formats are: .fasta, .faa and .txt")
    
    df = pd.DataFrame(sequences, columns=['ID', 'Sequence'])
    return df


def preprocess_sequences(df: pd.DataFrame,  max_length: int = 600) -> pd.DataFrame:
    """Preprocess sequences by padding/truncating and removing ambiguous amino acids."""

    processed_sequences = []

    for _, row in df.iterrows():
        id, seq = row['ID'], row['Sequence']
        if len(seq) < 50:
            continue  # remove fragments
        if len(seq) < max_length:
            seq = seq.ljust(max_length, '-')  # Pad sequences shorter than max_length with '-'
        elif len(seq) > max_length:
            seq = seq[:max_length]  # Truncate sequences longer than max_length
        processed_sequences.append((id, seq))

    df = pd.DataFrame(processed_sequences, columns=['ID', 'Sequence'])
    df = remove_ambiguous_aa(df)            # removing ambiguous amino acids

    return df


def create_encodings(df: pd.DataFrame, input_filename: str, model_name: str = 'esm2_t33_650M_UR50D', batch_size: int = 8, output_dir: str = 'encodings_genome', preprocess: bool = True) -> tuple:

    if preprocess:
        df = preprocess_sequences(df)

    repr_layer_map = {
        "esm2_t33": 33,
        "esm2_t6": 6,
        "esm2_t12": 12,
        "esm2_t30": 30,
        "esm2_t48": 48,
        "esm2_t36": 36
    }

    repr_layer = next((layer for name, layer in repr_layer_map.items() if name in model_name), None)
    if repr_layer is None:
        raise ValueError("Invalid model name provided")

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    batch_converter = alphabet.get_batch_converter()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    labels = df["ID"].tolist()
    sequences = df["Sequence"].tolist()

    # Create batches
    data_batches = [(labels[start:start+batch_size], sequences[start:start+batch_size]) for start in range(0, len(labels), batch_size)]

    sequence_representations = []
    start_time = time.time()

    for batch_labels, batch_sequences in data_batches:
        batch_data = [(label, sequence) for label, sequence in zip(batch_labels, batch_sequences)]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=True)

        token_representations = results["representations"][repr_layer].cpu()

        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1:tokens_len - 1].mean(0).cpu())     # Generate per-sequence representations
    total_time = round(time.time() - start_time)
    print("Time taken for encodings generation:", total_time, "seconds")

    os.makedirs(output_dir, exist_ok=True)
    input_file_base = os.path.splitext(os.path.basename(input_filename))[0]
    encodings_filename = f"{output_dir}/{input_file_base}_encodings.npy"

    np.save(encodings_filename, np.array(sequence_representations))  

    # Clean up to free memory
    del model, alphabet, batch_converter, token_representations, results
    gc.collect()
    torch.cuda.empty_cache()

    return np.array(sequence_representations), labels