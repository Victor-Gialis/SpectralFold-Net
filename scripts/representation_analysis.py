import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.colors as mcolors
import colorsys

from tqdm import tqdm
from models.model import Encoder
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader
from datasets.dataloader import get_dataset
from utils.statistics import global_stats, _z_norm, _log_norm

colors_map = {
    'normal_none' : '#7f7f7f',
    'inner_7' : '#1f77b4',
    'inner_14' : '#1f77b5',
    'inner_21' : '#1f77b6',
    'outer_7' : '#ff7f0e',
    'outer_14' : '#ff7f1e',
    'outer_21' : '#ff7f2e',
    'ball_7' : '#2ca02c',
    'ball_14' : '#2ca03c',
    'ball_21' : '#2ca04c',
}

def ajust_color_brightness(hex_color, factor):
    """Adjust the brightness of a hex color.

    Args:
        hex_color (str): The hex color code (e.g., '#ff5733').
        factor (float): The factor by which to adjust brightness. 
                        Values > 1 will make the color brighter, 
                        values < 1 will make it darker.

    Returns:
        str: The adjusted hex color code.
    """
    # Convert hex to RGB
    rgb = mcolors.hex2color(hex_color)
    # Convert RGB to HLS
    hls = colorsys.rgb_to_hls(*rgb)
    # Adjust luminance
    new_luminance = max(0, min(1, hls[1] * factor))
    # Convert back to RGB
    new_rgb = colorsys.hls_to_rgb(hls[0], new_luminance, hls[2])
    # Convert RGB back to hex
    new_hex = mcolors.to_hex(new_rgb)
    return new_hex

def _model_foward(model, batch, device):
    # X_true = batch['X_true'].unsqueeze(1).to(device, non_blocking=True)
    X_true = batch['X_tilde'].unsqueeze(1).to(device, non_blocking=True)

    labels = batch['label']
    # # One-hot encode labels
    # y_true = lb.transform(labels)
    # y_true = torch.tensor(y_true, dtype=torch.float32).to(device, non_blocking=True)
    # Get batch size
    b,_,_ = X_true.shape
    # Normalisation des signaux
    X_true_norm = _z_norm(X_true, mean.expand(b, 1, -1), std.expand(b, 1, -1))
    # Prédiction du modèle
    y_pred = model(X_true_norm)
    y_pred = y_pred[:,0] # On ne garde que le token CLS
    return y_pred, labels

# Constantes globales
SCRATCH = 'scratch'
PRETRAIN = 'pretrain'
FROZEN = 'frozen'
FINETUNE = 'finetune'

# Paramètres de la base de données à analyser
dataset_config = {
    "name":"CWRU",
    "fault_filter": None,
    "speed_filter": None,
    "transform_type": 'psd',
    "window_size": 2048,
    "stride": 256,
    "flip" : True,
    "downsample_factor": 2
}

# Création du Dataset
dataset_name = dataset_config['name']
dataset_params = {k: v for k, v in dataset_config.items()}
dataset = get_dataset(**dataset_params)

# Extraire les stats globales du dataset d'entraînement pour la normalisation
mean, std = global_stats(dataset)

# Création du  DataLoaders
batch_size = 256
collate_fn = getattr(dataset, '_collate_fn', None)
if collate_fn is None:
    # fallback: use a default collate_fn if not present
    from torch.utils.data.dataloader import default_collate
    collate_fn = default_collate

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Taille des signaux d'entrées
input_size = dataset[0]['X_true'].shape[-1]
print(f"Dataset '{dataset_name}' loaded with {len(dataset)} samples. Signal size: {input_size}")

# Charger la config du pré-entraînement pour les hyperparamètres
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean = mean.to(device)
std = std.to(device)

pretrain_model_name = 'model_v3.2'

with open(f'results/pretrain/{pretrain_model_name}/used_config.json', 'r') as f:
    pretrain_config = json.load(f)
pretrain_params = pretrain_config['model']

representations = dict()

for init_type in [SCRATCH, PRETRAIN]:
    # Chargement du backbone (encodeur)
    backbone = Encoder(
        num_patch= input_size // pretrain_params.get('patch_size', 64),
        patch_size= pretrain_params.get('patch_size', 64),
        encoder_dim= pretrain_params.get('encoder_dim', 128),
        n_layers= pretrain_params.get('n_layers', 4),
        heads= pretrain_params.get('heads', 8),
        dropout= pretrain_params.get('dropout', 0.4),
    ).to(device)

    if init_type == PRETRAIN:
        backbone.load_state_dict(torch.load(f'checkpoint/{pretrain_model_name}/model.pth'))
        print(f"Loaded pre-trained model '{pretrain_model_name}'")
    
    else :
        print("Using randomly initialized model")

    backbone.eval()

    representations[init_type] = {'X': [], 'Y': []}

    for batch in tqdm(dataloader, desc=f"Extracting representations ({init_type})"):
        # Extraction des représentations
        x, y = _model_foward(model=backbone, batch=batch, device=device)
        representations[init_type]['X'].append(x.detach())
        representations[init_type]['Y'].extend(y)
    
    representations[init_type]['X'] = torch.cat(representations[init_type]['X'], dim=0).cpu().numpy()
    representations[init_type]['Y'] = np.array(representations[init_type]['Y'])
    print(f"Extracted representations for {representations[init_type]['X'].shape[0]} samples.")

    tsne_model = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=42)
    X_tsne = tsne_model.fit_transform(representations[init_type]['X'])
    print('TSNE completed.')

    df = pd.DataFrame({
        'tsne1':X_tsne[:, 0],
        'tsne2':X_tsne[:, 1],
        'classes':representations[init_type]['Y']
    })

    fig = px.scatter(x=df['tsne1'],
                    y=df['tsne2'],
                    color=df['classes'],
                    color_discrete_map=colors_map,
                    category_orders={'classes': list(colors_map.keys())},
    )
    
    fig.write_html(f'{dataset_name}_{init_type}_embedding_space.html')
    print('oui')


