import time
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from Bio import SeqIO
import logging
from transformers import EsmModel, AutoTokenizer
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
            continue # remove fragments
        #     seq = seq.ljust(max_length, '-')  # Pad sequences shorter than max_length with '-'
        # elif len(seq) > max_length:
        #     seq = seq[:max_length]  # Truncate sequences longer than max_length
        processed_sequences.append((id, seq))

    df = pd.DataFrame(processed_sequences, columns=['ID', 'Sequence'])
    df = remove_ambiguous_aa(df)            # processing ambiguous amino acids

    return df


def create_embeddings(df: pd.DataFrame, model_name: str = 'facebook/esm2_t33_650M_UR50D', gpu: int=2) -> tuple:
    
    df = preprocess_sequences(df)

    ESMs = ["facebook/esm2_t6_8M_UR50D" ,
         "facebook/esm2_t12_35M_UR50D" ,
         "facebook/esm2_t30_150M_UR50D" ,
         "facebook/esm2_t33_650M_UR50D" ,
         "facebook/esm2_t36_3B_UR50D",
         "facebook/esm1b_t33_650M_UR50S"]

    if model_name not in ESMs:
        raise ValueError("Invalid model name provided") 
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float32 if model_name == "facebook/esm1b_t33_650M_UR50S" else torch.float16
    model = EsmModel.from_pretrained(model_name, torch_dtype=dtype)
    
#     ds_config = {
#     "fp16": {
#         "enabled": True
#     },
#     "zero_optimization": {
#         "stage": 3,
#         "offload_param": {
#             "device": "cpu",
#             "pin_memory": True
#         }
#     },
#     "train_micro_batch_size_per_gpu": 1,
#     "wall_clock_breakdown": True,
#     "activation_checkpointing": {
#         "partition_activations": True,
#         "contiguous_memory_optimization": True,
#         "cpu_checkpointing": False,
#         "synchronize_checkpoint_boundary": False,
#         "profile": False
#     }
# }
#     model, _, _, _ = deepspeed.initialize(model=model, config=ds_config)

    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if "esm2" in model_name:
        model = model.half()

    emb = []

    start_time = time.time()

#Although esm2 has no max_leng we truncate sequences bigger than 1024 (same as in the original esm2 training) because of memort constraints 
#memory requirements scale quadratically with sequence length
    for i in tqdm(range(0,len(df))):    #implementar batches
        inputs = tokenizer(df["Sequence"].loc[i], return_tensors="pt", max_length = 1024, truncation=True, padding=False).to(device)  

        with torch.no_grad():
            emb.append( np.array( torch.mean( model(**inputs).last_hidden_state.cpu(), dim = 1)))

    total_time = round(time.time() - start_time)
    logging.info(f"Time taken for encodings generation: {total_time} seconds")

    #create embedding df
    df_emb = pd.DataFrame(np.concatenate(emb))
    df_emb.reset_index( drop = True, inplace = True)
    df_emb["Sequence"] = df["Sequence"]
    df_emb["ID"] = df["ID"]  

    del model
    del tokenizer
    del df
    del inputs
    torch.cuda.empty_cache()
     
    return df_emb
