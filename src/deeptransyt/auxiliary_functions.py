import pandas as pd
import numpy as np
#from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
import cobra

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


def get_final_label(row, threshold):
    if row["SubFamily_confidence"] >= threshold:
        return row["Predicted_SubFamily"]
    if row["Family_confidence"] >= threshold:
        return row["Predicted_Family"]
    if row["Subclass_confidence"] >= threshold:
        return row["Predicted_Subclass"]
    if row["Class_confidence"] >= threshold:
        return row["Predicted_Class"]
    return "No annotation" 


def get_substrates(final_label, family_to_chebi, subfamily_to_chebi):
    if final_label in subfamily_to_chebi:  
        return subfamily_to_chebi[final_label]
    elif final_label in family_to_chebi: 
        return family_to_chebi[final_label]
    else:
        return [] 
    

def filter_chebi_substrates(df, model, chebi_column="Substrates"):
    if model is None:
        raise ValueError("Metabolic model must be provided.")

    model_chebis = set()
    for met in model.metabolites:
        if hasattr(met, "_annotation") and isinstance(met._annotation, dict):
            chebi_ids = met._annotation.get("chebi", [])
            if isinstance(chebi_ids, list):
                model_chebis.update(chebi_ids)
            elif isinstance(chebi_ids, str):
                model_chebis.add(chebi_ids)

    def filter_valid_substrates(substrates):
        if isinstance(substrates, str):
            substrates = substrates.split(", ") 
        return [s for s in substrates if s in model_chebis]

    df["Valid_Substrates"] = df[chebi_column].apply(filter_valid_substrates)

    return df
