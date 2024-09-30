import os
import argparse
import json
import logging
import numpy as np
import requests
from .sequence_processing import load_sequences, preprocess_sequences, create_encodings
from .make_predictions import (
    predict_binary,
    predict_family,
   # predict_subfamily,
   # predict_metabolic_important,
    predict_substrate_classes
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models_mappings')
BASE_URL = 'https://github.com/Apolinario8/deeptransyt/releases/download/v0.0.1/'

FILE_URLS = {
    'mapping_family12.json': BASE_URL + 'mapping_family12.json',
    'DNN_allclasses.ckpt': BASE_URL + 'DNN_allclasses.ckpt',
    'family_DNN_no9_12.ckpt': BASE_URL + 'family_DNN_no9_12.ckpt',
    'family_descriptions.json': BASE_URL + 'family_descriptions.json',
    'mapping_susbtrate_classes.json': BASE_URL + 'mapping_susbtrate_classes.json',
    'substrate_classes.ckpt': BASE_URL + 'substrate_classes.ckpt'
}

def download_file(file_name, url):
    file_path = os.path.join(MODEL_DIR, file_name)
    
    if not os.path.exists(file_path):
        print(f"Downloading {file_name} from {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"{file_name} downloaded successfully!")
        else:
            raise RuntimeError(f"Failed to download {file_name}. Status code: {response.status_code}")
    else:
        print(f"{file_name} already exists. Skipping download.")

def download_all_files():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    for file_name, url in FILE_URLS.items():
        download_file(file_name, url)

download_all_files()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(input_file: str, output_dir: str, preprocess: bool = True, gpu: int = 2) :
    df_sequences = load_sequences(input_file)
    
    if preprocess:
        logging.info("Preprocessing sequences and creating embeddings")
        df_sequences = preprocess_sequences(df_sequences)

    encodings, accessions = create_encodings(df_sequences, input_file)

    df_binary_predictions, binary_labels = predict_binary(encodings, accessions)

    transporter_indices = np.where(binary_labels == 1)[0]
    transporter_encodings = np.array(encodings)[transporter_indices]
    transporter_accessions = np.array(accessions)[transporter_indices]

    df_family_predictions = predict_family(transporter_encodings, transporter_accessions)
    #df_subfamily_predictions = predict_subfamily(transporter_encodings, transporter_accessions)
    #df_metabolic_predictions = predict_metabolic_important(transporter_encodings, transporter_accessions)
    df_susbtrate_classes_predictions = predict_substrate_classes(transporter_encodings, transporter_accessions)

    df_merged = df_binary_predictions.merge(df_family_predictions, on='Accession', how='left')
    #df_merged = df_merged.merge(df_subfamily_predictions, on='Accession', how='left')
    #df_merged = df_merged.merge(df_metabolic_predictions, on='Accession', how='left')
    df_merged = df_merged.merge(df_susbtrate_classes_predictions, on='Accession', how='left')

    #adding family descriptions correspoding to collumn family>12
    with open(os.path.join(MODEL_DIR, 'family_descriptions.json'), 'r') as f:
        family_descriptions = json.load(f)

    df_merged['Family_Description'] = df_merged['PredictedFamily_>12'].map(family_descriptions)
    
    #saving only positives rows in the df
    df_final = df_merged[df_merged['Accession'].isin(transporter_accessions)]

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "final_predictions.csv")
    df_final.to_csv(output_file, index=False)
    logging.info(f"All predictions saved to {output_file}")

    return df_final

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the prediction pipeline")
    parser.add_argument('--input_dir', type=str, required=True, help='Input file path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory path')
    parser.add_argument('--gpu', type=int, default=2, help='GPU index to use')
    parser.add_argument('--nopreprocess', action='store_false', dest='preprocess', help='Disable preprocessing of sequences')
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.gpu, args.preprocess)