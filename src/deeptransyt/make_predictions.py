import torch
import numpy as np
import pandas as pd
from .DNN import DNN, DNN_binary, DNN_weight
import json
from .auxiliary_functions import MODEL_DIR, MAPPING_DIR
import os
import requests

#MODEL_URL = 'https://github.com/Apolinario8/DeepTranSyT/releases/download/v0.0.1/DNN_allclasses.ckpt'

# def download_model():
#     if not os.path.exists(MODEL_DIR):
#         os.makedirs(MODEL_DIR)
    
#     model_path = os.path.join(MODEL_DIR, 'DNN_allclasses.ckpt')

#     if not os.path.exists(model_path):
#         print(f"Downloading model from {MODEL_URL}...")
#         response = requests.get(MODEL_URL)
#         if response.status_code == 200: 
#             with open(model_path, 'wb') as f:
#                 f.write(response.content)
#             print("Model downloaded successfully!")
#         else:
#             raise RuntimeError(f"Failed to download model. Status code: {response.status_code}")
        
def predict_binary(encodings: np.ndarray, accession: list) -> pd.DataFrame:
    #download_model()
    model_path = os.path.join(MODEL_DIR, 'DNN_allclasses.ckpt')

    model = DNN_binary.load_from_checkpoint(model_path)
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()

    tensor_encodings = torch.tensor(encodings, dtype=torch.float32)
    with torch.no_grad():
        predictions = torch.sigmoid(model(tensor_encodings)).numpy().flatten()

    df_binary_predictions = pd.DataFrame({'Accession': accession, "Binary_Predictions": predictions})

    binary_labels = (predictions > 0.5).astype(int)
    #num_transporters = np.sum(binary_labels)

    return df_binary_predictions, binary_labels


def predict_family(transporter_encodings: np.ndarray, transporter_accessions: list) -> pd.DataFrame:
    family_models = [
        {'path': os.path.join(MODEL_DIR, 'family_DNN_no9_10.ckpt'), 'num_classes': 402, 'column_name': 'PredictedFamily_>10', 'label_map': os.path.join(MAPPING_DIR, 'mapping_family10.json')},
        {'path': os.path.join(MODEL_DIR, 'family_DNN_no9_12.ckpt'), 'num_classes': 328, 'column_name': 'PredictedFamily_>12', 'label_map': os.path.join(MAPPING_DIR, 'mapping_family12.json')},
        {'path': os.path.join(MODEL_DIR, 'family_DNN_no9_15.ckpt'), 'num_classes': 279, 'column_name': 'PredictedFamily_>15', 'label_map': os.path.join(MAPPING_DIR, 'mapping_family15.json')},
        {'path': os.path.join(MODEL_DIR, 'family_DNN_no9_20.ckpt'), 'num_classes': 196, 'column_name': 'PredictedFamily_>20', 'label_map': os.path.join(MAPPING_DIR, 'mapping_family20.json')},
        {'path': os.path.join(MODEL_DIR, 'family_DNN_no9_30.ckpt'), 'num_classes': 109, 'column_name': 'PredictedFamily_>30', 'label_map': os.path.join(MAPPING_DIR, 'mapping_family30.json')},
        {'path': os.path.join(MODEL_DIR, 'family_DNN_no9_40.ckpt'), 'num_classes': 75, 'column_name': 'PredictedFamily_>40', 'label_map': os.path.join(MAPPING_DIR, 'mapping_family40.json')},
        {'path': os.path.join(MODEL_DIR, 'family_DNN_no9_50.ckpt'), 'num_classes': 51, 'column_name': 'PredictedFamily_>50', 'label_map': os.path.join(MAPPING_DIR, 'mapping_family50.json')},
    ]

    df_family_predictions = pd.DataFrame({'Accession': transporter_accessions})

    for model_info in family_models:
        num_classes = model_info['num_classes']
        
        family_model = DNN.load_from_checkpoint(checkpoint_path=model_info['path'], num_classes=num_classes)
        device = torch.device('cpu')
        family_model = family_model.to(device)
        family_model.eval()

        transporter_tensor = torch.tensor(transporter_encodings, dtype=torch.float32)
        with torch.no_grad():
            transporter_predictions = family_model(transporter_tensor)
        predicted_families = transporter_predictions.argmax(dim=1).numpy()

        with open(model_info['label_map'], 'r') as f:
            label_map = json.load(f)

        predicted_family_labels = [label_map[str(label)] for label in predicted_families]

        df_family_predictions[model_info['column_name']] = predicted_family_labels

    return df_family_predictions


def predict_subfamily(transporter_encodings: np.ndarray, transporter_accessions: list) -> pd.DataFrame:
    subfamily_models = [
        {'path': os.path.join(MODEL_DIR, 'subfamily_DNN_no9_15.ckpt'), 'num_classes': 267, 'column_name': 'PredictedsubFamily_>15', 'label_map': os.path.join(MAPPING_DIR, 'mapping_subfamily15.json')},
        {'path': os.path.join(MODEL_DIR, 'subfamily_DNN_no9_20.ckpt'), 'num_classes': 159, 'column_name': 'PredictedsubFamily_>20', 'label_map': os.path.join(MAPPING_DIR, 'mapping_subfamily20.json')},
        {'path': os.path.join(MODEL_DIR, 'subfamily_DNN_no9_30.ckpt'), 'num_classes': 75, 'column_name': 'PredictedsubFamily_>30', 'label_map': os.path.join(MAPPING_DIR, 'mapping_subfamily30.json')},
        {'path': os.path.join(MODEL_DIR, 'subfamily_DNN_no9_50.ckpt'), 'num_classes': 30, 'column_name': 'PredictedSubFamily_>50', 'label_map': os.path.join(MAPPING_DIR, 'mapping_subfamily50.json')}
    ]

    df_subfamily_predictions = pd.DataFrame({'Accession': transporter_accessions})

    for model_info in subfamily_models:
        num_classes = model_info['num_classes']
        
        subfamily_model = DNN.load_from_checkpoint(checkpoint_path=model_info['path'], num_classes=num_classes)
        device = torch.device('cpu')
        subfamily_model = subfamily_model.to(device)
        subfamily_model.eval()

        transporter_tensor = torch.tensor(transporter_encodings, dtype=torch.float32)
        with torch.no_grad():
            transporter_predictions = subfamily_model(transporter_tensor)
        predicted_subfamilies = transporter_predictions.argmax(dim=1).numpy()

        with open(model_info['label_map'], 'r') as f:
            label_map = json.load(f)

        predicted_subfamily_labels = [label_map[str(label)] for label in predicted_subfamilies]

        df_subfamily_predictions[model_info['column_name']] = predicted_subfamily_labels

    return df_subfamily_predictions


def predict_metabolic_important(transporter_encodings: np.ndarray, transporter_accessions: list) -> pd.DataFrame:    
    model_path = os.path.join(MODEL_DIR, 'newdataset_test_no9.ckpt')
    label_map_path = os.path.join(MAPPING_DIR, 'mapping_newdataset.json')
    
    new_model = DNN.load_from_checkpoint(checkpoint_path=model_path, num_classes=4)
    device = torch.device('cpu')
    new_model = new_model.to(device)
    new_model.eval()

    transporter_tensor = torch.tensor(transporter_encodings, dtype=torch.float32)
    with torch.no_grad():
        transporter_predictions = new_model(transporter_tensor)
    predicted_labels = transporter_predictions.argmax(dim=1).numpy()

    with open(label_map_path, 'r') as f:
        label_map = json.load(f)

    predicted_labels_names = [label_map[str(label)] for label in predicted_labels]

    df_new_predictions = pd.DataFrame({'Accession': transporter_accessions, 'Predicted_newdataset': predicted_labels_names})

    return df_new_predictions


def predict_substrate_classes(transporter_encodings: np.ndarray, transporter_accessions: list) -> pd.DataFrame:     
    
    model_path = 'models/substrate_classes.ckpt'
    label_map_path = 'mappings/mapping_susbtrate_classes.json'

    pos_weight = torch.tensor([0.2224, 0.0389, 0.0757, 0.3443, 0.0426, 0.0761, 0.2000], dtype=torch.float32)
    
    substrate_classes_model = DNN_weight.load_from_checkpoint(checkpoint_path=model_path, num_classes=7, weight=pos_weight)
    device = torch.device('cpu')
    substrate_classes_model = substrate_classes_model.to(device)
    substrate_classes_model.eval()

    transporter_tensor = torch.tensor(transporter_encodings, dtype=torch.float32)
    with torch.no_grad():
        transporter_predictions = substrate_classes_model(transporter_tensor)
    predicted_labels = transporter_predictions.argmax(dim=1).numpy()

    with open(label_map_path, 'r') as f:
        label_map = json.load(f)

    predicted_labels_names = [label_map[str(label)] for label in predicted_labels]

    df_substrate_classes = pd.DataFrame({'Accession': transporter_accessions, 'Class_substrate': predicted_labels_names})

    return df_substrate_classes