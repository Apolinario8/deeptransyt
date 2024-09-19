import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '.')
MAPPING_DIR = os.path.join(BASE_DIR, 'mappings')

def remove_ambiguous_aa(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing to replace ambiguous amino acids."""

    aa_replacement_map = {
        'B': 'N',  
        'Z': 'Q',  
        'U': 'C',  
        'O': 'K',  
        'J': 'I',  
        'X': ''   
    }

    processed_sequences = []
    for _, row in df.iterrows():
        sequence = row['Sequence']
        for aa, replacement in aa_replacement_map.items():
            sequence = sequence.replace(aa, replacement)
        processed_sequences.append((row['ID'], sequence))

    processed_df = pd.DataFrame(processed_sequences, columns=['ID', 'Sequence'])
    return processed_df