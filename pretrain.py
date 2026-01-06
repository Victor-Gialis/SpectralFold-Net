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


def foward_loss(device, batch):
    pretext_task = model.pretext_task

    # Récupération des tenseurs
    X_tilde = batch['X_tilde'].unsqueeze(1).to(device, non_blocking=True)
    X_true = batch['X_true'].unsqueeze(1).to(device, non_blocking=True)

    # Récupération de la taille du batch
    b,c,l = X_true.shape

    # Normalisation des signaux
    X_true_norm = _log_norm(x=X_true)
    X_tilde_norm = _log_norm(x=X_tilde)

    # Prédiction du modèle    
    if pretext_task == 'MAE':
        X_pred_norm, mask  = model(X_true_norm)

    else :
        X_pred_norm, _  = model(X_tilde_norm)

    X_pred_norm  = X_pred_norm.unsqueeze(1) #Ajouter la dimension du canal

    # Dénormalisation des signaux
    X_pred = _log_denorm(x=X_true, x_norm=X_pred_norm)

    # Valeur purement positive
    X_pred = torch.clamp(X_pred, min=0)

    # Calcul de la loss
    if pretext_task == 'MAE':
        patch_size = model.patch_size
        target = X_true.reshape(b, c, l // patch_size, patch_size)
        pred = X_pred.reshape(b, c, l // patch_size, patch_size)

        loss = (pred- target)**2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
    
    else :
        loss = loss_function(X_pred, X_true)

    return loss

# Initialisation Weights & Biases
wandb.login(key='3e0e644169a93d59382823b35ef232fdb2b25d25')

# Save name with timestamp
now = datetime.datetime.now()
model_name = 'model_v7.2'
save_name = f"{model_name}_{now.strftime('%Y%m%d_%H%M%S')}"

# Initialisation W&Bmodel_name = 'model_v7.0'
wandb.init(project="SpectralFoldNet-Pretraining", name=save_name)

# Config device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
model = PretrainingModel(pretext_task='MAE',
                         patch_size=16,
                         hidden_dim=512,
                         n_layers=3,
                         heads=8,
                         dropout=0.2565).to(device)

# Optimizer & Scheduler
epochs = 100
learning_rate = 0.0003695
weight_decay = 1.1133e-5

optimizer = torch.optim.AdamW(model.parameters(), lr= learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

# Loss function
loss_function = torch.nn.MSELoss()

### Train loop ###
training_loss =list()
validation_loss = list()

for epoch in range(epochs):
    model.train() # Training
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        optimizer.zero_grad() # Remise à zéro des gradients

        # Compute the loss
        loss = foward_loss(device, batch)

        loss.backward()
        train_loss += loss.item()

        # Gradient clipping
        optimizer.step()
    
    training_loss.append(train_loss/len(train_loader))
    
    model.eval() # Validation
    valid_loss = 0
    for batch in tqdm(valid_loader, desc=f"Validation Epoch {epoch+1}/{epochs}"):
        with torch.no_grad():

            # Compute the loss
            loss = foward_loss(device, batch)

            valid_loss += loss.item()
    
    validation_loss.append(valid_loss/len(valid_loader))
    scheduler.step()

    wandb.log({"train_loss": training_loss[-1], "valid_loss": validation_loss[-1]}, step=epoch)

### Test loop ###
model.eval()
test_loss = 0
for batch in tqdm(test_loader, desc="Testing"):
    with torch.no_grad():

        # Compute the loss
        loss = foward_loss(device, batch)

        test_loss += loss.item()

print(f"Test Loss: {test_loss/len(test_loader)}")

# Save results
results_dir = os.path.join('results/pretrain', model_name)
os.makedirs(results_dir, exist_ok=True)

plt.figure(figsize=(10,6))
plt.plot(range(1, epochs+1), training_loss, label='Training Loss')
plt.plot(range(1, epochs+1), validation_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid()

plt.savefig(os.path.join(results_dir, 'loss_curve.png'))

# Save the trained model
torch.save(model.state_dict(), os.path.join(results_dir, 'pretrained_model.pth'))

# Fin de l'expérience W&B
wandb.finish()