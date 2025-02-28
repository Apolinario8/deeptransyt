import os
import argparse
import json
import pandas as pd
import logging 
import numpy as np
import requests
from .auxiliary_functions import get_chebi_id
from .sequence_processing import load_sequences, preprocess_sequences, create_embeddings
from .make_predictions import (
    predict_binary,
    predict_family,
    predict_subfamily,
    predict_substrate_classes
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models_mappings')
BASE_URL = 'https://github.com/Apolinario8/deeptransyt/releases/download/v0.0.1/'

FILE_URLS = {
    'tcdb_susbtrate_mappings.json': BASE_URL + 'tcdb_susbtrate_mappings.json',
    'family_deploy_mappings.json': BASE_URL + 'family_deploy_mappings.json',
    'binary_esm650M_ratio_1_3.ckpt': BASE_URL + 'binary_esm650M_ratio_1_3.ckpt',
    'family_650M_deploy.ckpt': BASE_URL + 'family_650M_deploy.ckpt',
    'family_descriptions.json': BASE_URL + 'family_descriptions.json',
    'mapping_susbtrate_classes.json': BASE_URL + 'mapping_susbtrate_classes.json',
    'substrate_multiclass.ckpt': BASE_URL + 'substrate_multiclass.ckpt',
    'subfamily_650M.ckpt': BASE_URL + 'subfamily_650M.ckpt',
    'subfamily_mappings.json': BASE_URL + 'subfamily_mappings.json'
}


def download_file(file_name, url):
    file_path = os.path.join(MODEL_DIR, file_name)
    
    if not os.path.exists(file_path):
        #print(f"Downloading {file_name} from {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            #print(f"{file_name} downloaded successfully!")
        else:
            raise RuntimeError(f"Failed to download {file_name}. Status code: {response.status_code}")
    #else:
        #print(f"{file_name} already exists. Skipping download.")

def download_all_files():
    logging.info("Downloading trained models...")

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    for file_name, url in FILE_URLS.items():
        download_file(file_name, url)
 
download_all_files()
 
def main(input_file: str=None, output_dir: str = "results", gpu: int = 2, embeddings_file: str = None, organism_id: str = None, substrates_inchis: list = None, binary_threshold=0.5, annotation_threshold=0.5):
    
    if embeddings_file:
        logging.info("Loading existing embeddings...")
        df_embeddings = np.load(embeddings_file, allow_pickle=True)
        df_embeddings = pd.DataFrame(df_embeddings)
        embeddings = df_embeddings.iloc[:, :-2].astype(float).values
        accessions = df_embeddings.iloc[:, -1].tolist()
    else:
        df_sequences = load_sequences(input_file)
        df_sequences = preprocess_sequences(df_sequences)
        #embeddings, accessions = create_embeddings(df_sequences, gpu=gpu)
        df_embeddings = create_embeddings(df_sequences, gpu=gpu)
        embeddings = df_embeddings.drop(columns=["Sequence", "ID"]).values  
        accessions = df_embeddings["ID"].tolist()

        # saving the embeddings file
        output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}_embeddings.npy")
        np.save(output_file, df_embeddings.values)

    df_binary_predictions, binary_labels = predict_binary(embeddings, accessions, threshold=binary_threshold)

    transporter_indices = np.where(binary_labels == 1)[0]
    transporter_embeddings = np.array(embeddings)[transporter_indices]
    transporter_accessions = np.array(accessions)[transporter_indices]

    df_family_predictions = predict_family(transporter_embeddings, transporter_accessions, threshold=annotation_threshold)
    df_subfamily_predictions = predict_subfamily(transporter_embeddings, transporter_accessions, threshold=annotation_threshold)
    #df_susbtrate_classes_predictions = predict_substrate_classes(transporter_embeddings, transporter_accessions)

    df_merged = df_binary_predictions.merge(df_family_predictions, on='Accession', how='left')
    df_merged = df_merged.merge(df_subfamily_predictions, on='Accession', how='left')
    #df_merged = df_merged.merge(df_susbtrate_classes_predictions, on='Accession', how='left')

    #with open(os.path.join(MODEL_DIR, 'family_descriptions.json'), 'r') as f:
        #family_descriptions = json.load(f)

    #df_merged['Family_Description'] = df_merged['Predicted_Family'].map(family_descriptions)

    #getting the substrates associated either with family or subfamily (if they match the family)
    #with open(os.path.join(MODEL_DIR, 'tcdb_susbtrate_mappings.json'), 'r') as file:
    #    data = json.load(file)

    #family_to_chebi = data["family"]
    #subfamily_to_chebi = data["subfamily"]

    #df_merged["Associated ChEBIs"] = df_merged.apply(
    #lambda row: get_chebi_id(row["Predicted_Family"], row["Predicted_SubFamily"], family_to_chebi, subfamily_to_chebi),
    #axis=1
#)

    # df_merged["Corrected_SubFamily"] = df_merged.apply(
    #     lambda row: row["Predicted_SubFamily"]
    #     if isinstance(row["Predicted_SubFamily"], str)
    #     and row["Predicted_SubFamily"].startswith(row["Predicted_Family"])
    #     else "-",
    #     axis=1
    # )

    # Define Associated ChEBIs de forma simples
    # df_merged["Associated ChEBIs"] = df_merged.apply(
    #     lambda row: get_subfamily_and_chebi(
    #         row["Predicted_Family"],
    #         row["Corrected_SubFamily"],   
    #         family_to_chebi,
    #         subfamily_to_chebi
    #     ),
    #     axis=1
    # )

    df_final = df_merged[df_merged['Accession'].isin(transporter_accessions)]

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "final_predictions.csv")
    df_final.to_csv(output_file, index=False)
    logging.info(f"All predictions saved to {output_file}")
    
    return df_final

def cli_main():
    parser = argparse.ArgumentParser(description="Run the prediction pipeline")
    parser.add_argument('--input_file', type=str, required=False, help='Path to fasta containing sequences (genome)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory path')
    parser.add_argument('--gpu', type=int, default=2, help='GPU index to use')
    parser.add_argument('--embeddings_file', type=str, help='Path to existing embeddings file (optional)')
    parser.add_argument('--binary_threshold', type=float, default=0.5, help='Threshold for binary predictions')

    args = parser.parse_args()
    main(args.input_file, 
         args.output_dir, 
         args.gpu, 
         args.embeddings_file, 
         args.binary_threshold) 

if __name__ == "__main__":
    cli_main()