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
from models.model import PretrainingModel
from datasets.dataloader import get_dataset
from utils.statistics import _z_norm, _log_norm, _log_denorm, global_stats, mse_loss

# Config device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'model_v7.2'
result_dir = f'results/pretrain/{model_name}'

# Get dataset
dataset = get_dataset(name='CWRU',
                      fault_filter = None,
                      speed_filter = None,
                      transform_type = 'psd',
                      window_size = 2048,
                      stride = 256,
                      pretext_task = 'flip',
                      downsample_factor = 2)

# Split train/valid/test
train_size = int(0.6 * len(dataset))
valid_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - valid_size

# Generate split data
generator =torch.Generator().manual_seed(42)
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, 
                                                                           [train_size, valid_size, test_size], 
                                                                           generator=generator)

# Split DataLoaders
batch_size = 64
collate_fn = getattr(dataset, '_collate_fn', None)
if collate_fn is None:
    # fallback: use a default collate_fn if not present
    from torch.utils.data.dataloader import default_collate
    collate_fn = default_collate

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

print(f"DataLoaders created: train ({len(train_loader)} batches), valid ({len(valid_loader)} batches), test ({len(test_loader)} batches)")

# Instantiate the model in self-supervised learning
# Instantiate the model in self-supervised learning
model = PretrainingModel(pretext_task='MAE',
                         patch_size=16,
                         hidden_dim=512,
                         n_layers=3,
                         heads=8,
                         dropout=0.2565).to(device)

model.load_state_dict(torch.load(f'results/pretrain/{model_name}/pretrained_model.pth'))

patch_size = model.patch_size

for batch in tqdm(test_loader) : 
    X_true = batch['X_true'].unsqueeze(1).to(device, non_blocking=True)
    X_true_norm = _log_norm(x=X_true)

    # Récupération de la taille du batch
    b,c,l = X_true.shape

    X_pred_norm, mask  = model(X_true_norm)
    X_pred_norm  = X_pred_norm.unsqueeze(1) #Ajouter la dimension du canal

    # Dénormalisation des signaux
    X_pred = _log_denorm(x=X_true, x_norm=X_pred_norm)

    # Signal masqué
    mask_expanded = mask.unsqueeze(1).unsqueeze(-1).repeat(1, 1, 1, 16)

    X_mask = X_true.reshape(b, c, l // patch_size, patch_size)
    X_mask = X_mask * mask_expanded
    X_mask = X_mask.reshape(b, c, l)
    
    print('Shapes : ', X_true.shape, X_mask.shape, X_pred.shape)

    x_true = X_true[0,0,:].cpu().numpy()
    x_mask = X_mask[0,0,:].cpu().numpy()
    x_pred = X_pred[0,0,:].detach().cpu().numpy()

    freq_axis = np.linspace(0,6000,len(x_true))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq_axis, y=x_true, mode='lines', name='True Signal'))
    fig.add_trace(go.Scatter(x=freq_axis, y=x_mask, mode='lines', name='Masked Signal'))
    fig.add_trace(go.Scatter(x=freq_axis, y=x_pred, mode='lines', name='Predicted Signal'))
    fig.update_layout(title=f'Validation batch - Signal Comparison',xaxis_title='Frequency [Hz]',yaxis_title='Amplitude')
    fig.write_html(os.path.join(result_dir, f'reconstruction_MAE_results.html'))

    break
