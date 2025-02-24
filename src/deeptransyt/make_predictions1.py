import torch
import numpy as np
import pandas as pd
from .DNN1 import DNN_binary, DNN_substrate, DNN_family, DNN_subfamily
import json
import os
import torch.nn.functional as F
from .auxiliary_functions import getMeanRepr, prepare_prediction_data
from .fetch_metabolites import get_genome_metabolites_as_smiles
import pickle
from os.path import join
from transformers import AutoTokenizer, AutoModelForMaskedLM
import xgboost as xgb
from rdkit import Chem

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models_mappings')

def predict_binary(embeddings: np.ndarray, accession: list, threshold=0.5) -> pd.DataFrame:
    model_path = os.path.join(MODEL_DIR, 'binary_esm650M_ratio_1_3.ckpt')

    model = DNN_binary.load_from_checkpoint(model_path)
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    
    tensor_embeddings = torch.tensor(embeddings, dtype=torch.float32)
    with torch.no_grad():
        predictions = torch.sigmoid(model(tensor_embeddings)).numpy().flatten()

    df_binary_predictions = pd.DataFrame({'Accession': accession, "Binary_Predictions": predictions})

    binary_labels = (predictions > threshold).astype(int)
    #binary_labels = predictions 
    #num_transporters = np.sum(binary_labels)

    return df_binary_predictions, binary_labels

def predict_family(embeddings: np.ndarray, accession: list, threshold=0.5) -> pd.DataFrame:
    model_path = os.path.join(MODEL_DIR, 'family_650M_deploy.ckpt')

    model = DNN_family.load_from_checkpoint(checkpoint_path = model_path, num_classes_level3=330)  
    #model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    device = torch.device('cpu')
    model = model.to(device)
    model.eval() 

    tensor_embeddings = torch.tensor(embeddings, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(tensor_embeddings)
        probabilities = F.softmax(outputs, dim=1)  

        max_confidences, best_indices = probabilities.max(dim=1)  
        predictions = best_indices.numpy()

    with open(os.path.join(MODEL_DIR, 'family_deploy_mappings.json'), 'r') as f:
        label_map = json.load(f)

    predicted_labels_with_threshold = []
    for idx, conf in zip(predictions, max_confidences):
        if conf.item() >= threshold:
            predicted_labels_with_threshold.append(label_map[str(idx)])
        else:
            predicted_labels_with_threshold.append("-")

    df_predictions = pd.DataFrame({
        'Accession': accession,
        'Family_confidence': max_confidences.numpy(),
        'Predicted_Family': predicted_labels_with_threshold
    })

    return df_predictions


def predict_subfamily(embeddings: np.ndarray, accession: list, threshold=0.5) -> pd.DataFrame:
    model_path = os.path.join(MODEL_DIR, 'subfamily_650M.ckpt')

    model = DNN_subfamily.load_from_checkpoint(checkpoint_path = model_path, num_classes_level4=420)  
    #model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    device = torch.device('cpu')
    model = model.to(device)
    model.eval() 

    tensor_embeddings = torch.tensor(embeddings, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(tensor_embeddings)
        probabilities = F.softmax(outputs, dim=1)  

        max_confidences, best_indices = probabilities.max(dim=1)  
        predictions = best_indices.numpy()

    with open(os.path.join(MODEL_DIR, 'subfamily_mappings.json'), 'r') as f:
            label_map = json.load(f)

    predicted_labels_with_threshold = []
    for idx, conf in zip(predictions, max_confidences):
        if conf.item() >= threshold:
            predicted_labels_with_threshold.append(label_map[str(idx)])
        else:
            predicted_labels_with_threshold.append("-")

    df_predictions = pd.DataFrame({
        'Accession': accession,
        'SubFamily_confidence': max_confidences.numpy(),
        'Predicted_SubFamily': predicted_labels_with_threshold
    })

    return df_predictions


def predict_substrate_classes(embeddings: np.ndarray, accession: list) -> pd.DataFrame:     
    
    model_path = os.path.join(MODEL_DIR, 'substrate_multiclass.ckpt')

    model = DNN_substrate.load_from_checkpoint(checkpoint_path = model_path, num_classes_level1=7)  
    #model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    device = torch.device('cpu')
    model = model.to(device)
    model.eval() 

    tensor_embeddings = torch.tensor(embeddings, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(tensor_embeddings) 
        predictions = torch.argmax(outputs, dim=1).numpy()

    with open(os.path.join(MODEL_DIR, 'mapping_susbtrate_classes.json'), 'r') as f:
            label_map = json.load(f)

    predicted_subs_labels = [label_map[str(label)] for label in predictions]

    df_predictions = pd.DataFrame({
        'Accession': accession,
        'Subs_confidence': F.softmax(outputs, dim=1).numpy().max(axis=1),
        'Predicted_substrate': predicted_subs_labels
    })

    return df_predictions
