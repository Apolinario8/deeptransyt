import torch
import numpy as np
import pandas as pd
from .models import MLP_binary, MLP_substrate, MLP_family, MLP_subfamily, MLP_class, MLP_subclass
import json
import os
import torch.nn.functional as F
#from .auxiliary_functions import getMeanRepr, prepare_prediction_data
#from .fetch_metabolites import get_genome_metabolites_as_smiles
#import pickle
from os.path import join
# from transformers import AutoTokenizer, AutoModelForMaskedLM
# import xgboost as xgb
# from rdkit import Chem

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models_mappings')

def predict_binary(embeddings: np.ndarray, accession: list, threshold=0.5) -> pd.DataFrame:
    """
    Predict binary labels (e.g., transporter vs non-transporter) for a set of protein sequences 
    based on pre-trained embeddings using a binary classification model.

    **Parameters:**
    - `embeddings` (np.ndarray): A numpy array containing the sequence embeddings to be predicted. 
      Each row corresponds to the embedding of a single protein sequence.
    - `accession` (list): A list of accession IDs corresponding to the sequences in the embeddings. 
      Each entry should match the order of the embeddings.
    - `threshold` (float, optional): The threshold for classifying the predictions. If the predicted probability is greater than this threshold, the sequence is classified as 1 (e.g., transporter), otherwise as 0 (e.g., non-transporter). Default is 0.5.

    **Returns:**
    - `pd.DataFrame`: A DataFrame with two columns:
      - `'Accession'`: The accession IDs corresponding to the sequences.
      - `'Binary_Predictions'`: The predicted probabilities for each sequence.
    - `binary_labels` (np.ndarray): A numpy array of binary labels (0 or 1) for each sequence, based on the provided threshold.

    **Raises:**
    - `FileNotFoundError`: If the model checkpoint file does not exist at the specified `model_path`.
    - `ValueError`: If the input `embeddings` array and `accession` list have mismatched lengths.
    """
    model_path = os.path.join(MODEL_DIR, 'binary_esm650M_ratio_1_3.ckpt')

    model = MLP_binary.load_from_checkpoint(model_path)
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

def predict_class(embeddings: np.ndarray, accession: list, threshold=0.5) -> pd.DataFrame:
    """
    Predict the TCDB family (e.g., transporter family) for a set of protein sequences 
    based on pre-trained embeddings using a multi-class classification model.

    **Parameters:**
    - `embeddings` (np.ndarray): A numpy array containing the sequence embeddings to be predicted. 
      Each row corresponds to the embedding of a single protein sequence.
    - `accession` (list): A list of accession IDs corresponding to the sequences in the embeddings. 
      Each entry should match the order of the embeddings.
    - `threshold` (float, optional): The threshold for classifying the predictions. If the predicted confidence is greater than this threshold, 
      the sequence is assigned a predicted family label, otherwise it is assigned a placeholder ("-"). Default is 0.5.

    **Returns:**
    - `pd.DataFrame`: A DataFrame with three columns:
      - `'Accession'`: The accession IDs corresponding to the sequences.
      - `'Family_confidence'`: The confidence score of the predicted family label.
      - `'Predicted_Family'`: The predicted TCDB family label for each sequence, or `'-'` if the confidence is below the threshold.

    **Raises:**
    - `FileNotFoundError`: If the model checkpoint file (`family_650M_deploy.ckpt`) or the mapping file (`family_deploy_mappings.json`) cannot be found.
    - `ValueError`: If the input `embeddings` array and `accession` list have mismatched lengths.

"""
    model_path = os.path.join(MODEL_DIR, 'class_650M.ckpt')

    model = MLP_class.load_from_checkpoint(checkpoint_path = model_path, num_classes_level1=5)  
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

    with open(os.path.join(MODEL_DIR, 'class_mappings.json'), 'r') as f:
        label_map = json.load(f)

    predicted_labels_with_threshold = []
    for idx, conf in zip(predictions, max_confidences):
        if conf.item() >= threshold:
            predicted_labels_with_threshold.append(label_map[str(idx)])
        else:
            predicted_labels_with_threshold.append("-")

    df_predictions = pd.DataFrame({
        'Accession': accession,
        'Class_confidence': max_confidences.numpy(),
        'Predicted_Class': predicted_labels_with_threshold
    })

    return df_predictions

def predict_subclass(embeddings: np.ndarray, accession: list, threshold=0.5) -> pd.DataFrame:
    """
    Predict the TCDB family (e.g., transporter family) for a set of protein sequences 
    based on pre-trained embeddings using a multi-class classification model.

    **Parameters:**
    - `embeddings` (np.ndarray): A numpy array containing the sequence embeddings to be predicted. 
      Each row corresponds to the embedding of a single protein sequence.
    - `accession` (list): A list of accession IDs corresponding to the sequences in the embeddings. 
      Each entry should match the order of the embeddings.
    - `threshold` (float, optional): The threshold for classifying the predictions. If the predicted confidence is greater than this threshold, 
      the sequence is assigned a predicted family label, otherwise it is assigned a placeholder ("-"). Default is 0.5.

    **Returns:**
    - `pd.DataFrame`: A DataFrame with three columns:
      - `'Accession'`: The accession IDs corresponding to the sequences.
      - `'Family_confidence'`: The confidence score of the predicted family label.
      - `'Predicted_Family'`: The predicted TCDB family label for each sequence, or `'-'` if the confidence is below the threshold.

    **Raises:**
    - `FileNotFoundError`: If the model checkpoint file (`family_650M_deploy.ckpt`) or the mapping file (`family_deploy_mappings.json`) cannot be found.
    - `ValueError`: If the input `embeddings` array and `accession` list have mismatched lengths.

"""
    model_path = os.path.join(MODEL_DIR, 'subclass_650M.ckpt')

    model = MLP_subclass.load_from_checkpoint(checkpoint_path = model_path, num_classes_level2=31)  
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

    with open(os.path.join(MODEL_DIR, 'subclass_mappings.json'), 'r') as f:
        label_map = json.load(f)

    predicted_labels_with_threshold = []
    for idx, conf in zip(predictions, max_confidences):
        if conf.item() >= threshold:
            predicted_labels_with_threshold.append(label_map[str(idx)])
        else:
            predicted_labels_with_threshold.append("-")

    df_predictions = pd.DataFrame({
        'Accession': accession,
        'Subclass_confidence': max_confidences.numpy(),
        'Predicted_Subclass': predicted_labels_with_threshold
    })

    return df_predictions


def predict_family(embeddings: np.ndarray, accession: list, threshold=0.5) -> pd.DataFrame:
    """
    Predict the TCDB family (e.g., transporter family) for a set of protein sequences 
    based on pre-trained embeddings using a multi-class classification model.

    **Parameters:**
    - `embeddings` (np.ndarray): A numpy array containing the sequence embeddings to be predicted. 
      Each row corresponds to the embedding of a single protein sequence.
    - `accession` (list): A list of accession IDs corresponding to the sequences in the embeddings. 
      Each entry should match the order of the embeddings.
    - `threshold` (float, optional): The threshold for classifying the predictions. If the predicted confidence is greater than this threshold, 
      the sequence is assigned a predicted family label, otherwise it is assigned a placeholder ("-"). Default is 0.5.

    **Returns:**
    - `pd.DataFrame`: A DataFrame with three columns:
      - `'Accession'`: The accession IDs corresponding to the sequences.
      - `'Family_confidence'`: The confidence score of the predicted family label.
      - `'Predicted_Family'`: The predicted TCDB family label for each sequence, or `'-'` if the confidence is below the threshold.

    **Raises:**
    - `FileNotFoundError`: If the model checkpoint file (`family_650M_deploy.ckpt`) or the mapping file (`family_deploy_mappings.json`) cannot be found.
    - `ValueError`: If the input `embeddings` array and `accession` list have mismatched lengths.

"""
    model_path = os.path.join(MODEL_DIR, 'family_650M_deploy.ckpt')

    model = MLP_family.load_from_checkpoint(checkpoint_path = model_path, num_classes_level3=330)  
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
    """
    Predict the TCDB subfamily (e.g., transporter subfamily) for a set of protein sequences 
    based on pre-trained embeddings using a multi-class classification model.

    **Parameters:**
    - `embeddings` (np.ndarray): A numpy array containing the sequence embeddings to be predicted. 
      Each row corresponds to the embedding of a single protein sequence.
    - `accession` (list): A list of accession IDs corresponding to the sequences in the embeddings. 
      Each entry should match the order of the embeddings.
    - `threshold` (float, optional): The threshold for classifying the predictions. If the predicted confidence is greater than this threshold, 
      the sequence is assigned a predicted subfamily label, otherwise it is assigned a placeholder ("-"). Default is 0.5.

    **Returns:**
    - `pd.DataFrame`: A DataFrame with three columns:
      - `'Accession'`: The accession IDs corresponding to the sequences.
      - `'SubFamily_confidence'`: The confidence score of the predicted subfamily label.
      - `'Predicted_SubFamily'`: The predicted TCDB subfamily label for each sequence, or `'-'` if the confidence is below the threshold.

    **Raises:**
    - `FileNotFoundError`: If the model checkpoint file (`subfamily_650M.ckpt`) or the mapping file (`subfamily_mappings.json`) cannot be found.
    - `ValueError`: If the input `embeddings` array and `accession` list have mismatched lengths.
    """

    model_path = os.path.join(MODEL_DIR, 'subfamily_650M.ckpt')

    model = MLP_subfamily.load_from_checkpoint(checkpoint_path = model_path, num_classes_level4=420)  
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
    """
    Predict substrate classes for a set of sequences based on their embeddings using a trained model.

    **Parameters:**
    - `embeddings` (np.ndarray): The sequence embeddings (e.g., from a pre-trained model) to be used for prediction.
    - `accession` (list): A list of sequence accessions (IDs) corresponding to the embeddings.

    **Returns:**
    - `pd.DataFrame`: A DataFrame containing the following columns:
        - `'Accession'`: The sequence accession (ID).
        - `'Subs_confidence'`: The confidence score (probability) for the predicted substrate class.
        - `'Predicted_substrate'`: The predicted substrate class label.
    """
    
    model_path = os.path.join(MODEL_DIR, 'substrate_multiclass.ckpt')

    model = MLP_substrate.load_from_checkpoint(checkpoint_path = model_path, num_classes_level3=7)  
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