import torch
import json
import wandb
import os
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from tqdm import tqdm
from torch.utils.data import DataLoader
from models.model import PretrainedModel
from datasets.dataloader import get_dataset
from utils.statistics import _z_norm, _log_norm, _log_denorm, global_stats, mse_loss

def _model_foward(device, model, batch):
    # Récupération des tenseurs
    X_tilde = batch['X_tilde'].unsqueeze(1).to(device, non_blocking=True)
    X_true = batch['X_true'].unsqueeze(1).to(device, non_blocking=True)

    # Récupération des métadonnées
    metadata = pd.DataFrame(batch['metadata'])
    metadata['label'] = batch['label']

    # Normalisation des signaux
    X_tilde_norm = _log_norm(x=X_tilde)

    # Prédiction du modèle
    X_pred_norm  = model(X_tilde_norm)
    X_pred_norm  = X_pred_norm.unsqueeze(1)

    # Dénormalisation des signaux
    X_pred = _log_denorm(x=X_true, x_norm=X_pred_norm)

    # Valeur purement positive
    X_pred = torch.clamp(X_pred, min=0)

    return X_pred, X_true, X_tilde, metadata

# Définir le device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger configuration datatset
dataset_config = {
      "name":"LASPI",
      "fault_filter": None,
      "speed_filter": None,
      "transform_type":"psd",
      "window_size": 2048,
      "stride": 256,
      "pretext_task" : "flip",
      "downsample_factor": 2}

# Instancier le dataset via la factory
dataset_name = dataset_config['name']
dataset_params = {k: v for k, v in dataset_config.items() if k != 'name'}
dataset = get_dataset(dataset_name, **dataset_params)
input_size = dataset[0]['X_true'].shape[-1]

# Calculer la variance globale de la base de données
X = list()
for data in dataset:
    X.append(data['X_true'].unsqueeze(0))
X = torch.cat(X, dim=0)
X_var = torch.var(X)
print(f'Global variance of the dataset: {X_var.item():.12f}')

# Charger la config
pretrained_model_name = 'model_v5.1'

# Charger la configuration du modèle pré-entraîné
with open(os.path.join('results', 'pretrain', pretrained_model_name, 'used_config.json'), 'r') as f:
    model_config = json.load(f)
    model_config = model_config['model']

# Split DataLoaders
batch_size = 256
collate_fn = getattr(dataset, '_collate_fn', None)
if collate_fn is None:
    # fallback: use a default collate_fn if not present
    from torch.utils.data.dataloader import default_collate
    collate_fn = default_collate

# Création du DataLoader
dataloader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        collate_fn=collate_fn)

# Instancier le modèle
model = PretrainedModel(input_size=input_size,**model_config)
model.to(device)

# Charger les poids du modèle
model.load_state_dict(torch.load(f'checkpoint/{pretrained_model_name}/model.pth'))

# Mettre le modèle en mode évaluation
model.eval()
criterion = torch.nn.MSELoss(reduction='none')
inference_result = list()
labels_plot = list()

for batch in tqdm(dataloader, desc="Inference"):
    X_pred, X_true, X_tilde, metadata = _model_foward(device, model, batch)

    # Calcul de la perte
    batch_loss = criterion(X_pred, X_true).squeeze(1).mean(-1)

    metadata['loss']= batch_loss.detach().cpu().numpy()
    inference_result.append(metadata)

    for label in metadata['label'].unique():
        if label not in labels_plot:
            labels_plot.append(label)
            frame = metadata[metadata['label'] == label]
            idx = np.random.choice(frame.index)

            x_true = X_true[idx, 0, :].detach().cpu().numpy()
            x_tilde = X_tilde[idx, 0, :].detach().cpu().numpy()
            x_pred = X_pred[idx, 0, :].detach().cpu().numpy()

            N = len(x_true)
            x_tilde = x_tilde[:N//2]

            freq_axis = np.arange(len(x_true))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=freq_axis, y=x_true, mode='lines', name='True Signal'))
            fig.add_trace(go.Scatter(x=freq_axis[:N//2], y=x_tilde, mode='lines', name='Tilde Signal'))
            fig.add_trace(go.Scatter(x=freq_axis, y=x_pred, mode='lines', name='Predicted Signal'))
            fig.update_layout(title=f'Validation batch - Signal Comparison - Label: {label}',
                                xaxis_title='Frequency [Hz]',
                                yaxis_title='Amplitude')
            fig.write_html(f'comparison_label_{label}.html')

# Concaténer les résultats
inference_result = pd.concat(inference_result, ignore_index=True)
print('Inference results:')