import pandas as pd
import numpy as np
#from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

def remove_ambiguous_aa(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing to replace ambiguous amino acids."""

    aa_replacement_map = {
        'B': 'X',  
        'Z': 'X',  
        'U': 'X',  
        'O': 'X',  
        'J': 'X'   
    }

    processed_sequences = []
    for _, row in df.iterrows():
        sequence = row['Sequence']
        for aa, replacement in aa_replacement_map.items():
            sequence = sequence.replace(aa, replacement)
        processed_sequences.append((row['ID'], sequence))

    processed_df = pd.DataFrame(processed_sequences, columns=['ID', 'Sequence'])
    return processed_df


def getMeanRepr(smiles_data, tokenizer, model):
    mean_repr = np.zeros((smiles_data.shape[0], 767))
    for i, sequence in enumerate(tqdm(smiles_data)):
        inputs = tokenizer.encode(sequence, return_tensors="pt")
        output_repr = model(inputs)
        mean_repr[i] = output_repr.logits[0].mean(dim=0).detach().numpy()
    return mean_repr


def prepare_prediction_data(transporter_encoding, substrate_chemb):
    X = []
    for ecfp in substrate_chemb:
        X.append(np.concatenate([ecfp, transporter_encoding]))
    return np.array(X)


def get_chebi_id(family, subfamily, family_to_chebi, subfamily_to_chebi):
    if not isinstance(subfamily, str):
        return "No CHEBI ID"
    if not isinstance(family, str):
        return "No CHEBI ID"
    
    if subfamily.startswith(family):
        chebis = subfamily_to_chebi.get(subfamily, [])
    else:
        chebis = family_to_chebi.get(family, [])
        
    if isinstance(chebis, list):
        return ", ".join(chebis)
    else:
        return "No CHEBI ID"


# def get_subfamily_and_chebi(family, subfamily, family_to_chebi, subfamily_to_chebi):
#     if not isinstance(family, str) or pd.isna(family):
#         family = ""
#     if not isinstance(subfamily, str) or pd.isna(subfamily):
#         subfamily = ""

#     if subfamily.startswith(family):
#         new_subfamily = subfamily
#         chebis = subfamily_to_chebi.get(subfamily, [])
#     else:
#         new_subfamily = "-"
#         chebis = family_to_chebi.get(family, [])
#     if isinstance(chebis, list):
#         associated_chebis = ", ".join(chebis)
#     else:
#         associated_chebis = "No CHEBI ID"

#     return new_subfamily, associated_chebis
