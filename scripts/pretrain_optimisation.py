import torch
import json
import os
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import optuna

from optuna.integration.wandb import WeightsAndBiasesCallback
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.model import PretrainedModel
from datasets.dataloader import get_dataset
from utils.statistics import _z_norm, _log_norm, _log_denorm, global_stats, mse_loss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import wandb
from tqdm import tqdm

def _model_foward(model, batch, normalization, mean, std):
    # Récupération des tenseurs
    X_tilde = batch['X_tilde'].unsqueeze(1).to(DEVICE, non_blocking=True)
    X_true = batch['X_true'].unsqueeze(1).to(DEVICE, non_blocking=True)

    # Récupération de la taille du batch
    b,_,_ = X_true.shape

    # Normalisation des signaux
    if normalization == 'z':
        X_tilde_norm = _z_norm(x=X_tilde, mean=mean.expand(b, 1, -1), std=std.expand(b, 1, -1))
    elif normalization == 'log':
        X_tilde_norm = _log_norm(x=X_tilde)
    
    # Prédiction du modèle
    X_pred_norm  = model(X_tilde_norm)
    X_pred_norm  = X_pred_norm.unsqueeze(1)

    # Dénormalisation des signaux
    if normalization == 'z':
        X_pred = X_pred_norm * std + mean
    elif normalization == 'log':
        X_pred = _log_denorm(x=X_true, x_norm=X_pred_norm)

    # Valeur purement positive
    X_pred = torch.clamp(X_pred, min=0)

    return X_pred, X_true

# --- Import ton modèle et tes fonctions de dataset ---
# from spectralfoldnet import SpectralFoldNet
# from dataloader import MyDataset, get_dataloaders

wandb.login(key='3e0e644169a93d59382823b35ef232fdb2b25d25')

# ---------------------------
# 1. Définition de l'expérience
# ---------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Exemple placeholder :
dataset = get_dataset(name='CWRU',
                    fault_filter= None,
                    speed_filter= None,
                    transform_type="psd",
                    window_size= 2048,
                    stride= 256,
                    pretext_task= "flip",
                    downsample_factor= 2)

# Tailler l'entrée du modèle en fonction des données
input_size = dataset[0]['X_true'].shape[-1]

# Split train/valid/test
train_size = int(0.7 * len(dataset))
valid_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - valid_size

generator =torch.Generator().manual_seed(42)
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size], generator=generator)

collate_fn = getattr(dataset, '_collate_fn', None)
if collate_fn is None:
    # fallback: use a default collate_fn if not present
    from torch.utils.data.dataloader import default_collate
    collate_fn = default_collate

# Statistiques globales pour la normalisation
mean, std = global_stats(train_dataset)
mean = mean.to(DEVICE)
std = std.to(DEVICE)

def objective(trial):
    """Fonction d'optimisation pour Optuna."""

    # --- Initialisation W&B (nouvel essai) ---
    wandb.init(
        project="SpectralFoldNet-Tuning",
        name=f"trial-{trial.number}",
        reinit=True,
        config={}
    )

    # --------------------
    # 2. Hyperparamètres à explorer
    # --------------------
    patch_size = trial.suggest_categorical("patch_size", [16, 32, 64])
    encoder_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    heads = trial.suggest_categorical("num_heads", [4, 8])
    n_layers = trial.suggest_int("num_layers", 2, 6)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    normalization = trial.suggest_categorical("normalization", ['z', 'log'])

    # --------------------
    # 3. Chargement des données
    # --------------------
    # train_loader, val_loader = get_dataloaders(batch_size=batch_size)

    # Exemple placeholder :
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # --------------------
    # 4. Initialisation du modèle
    # --------------------
    model = PretrainedModel(
        input_size=input_size,
        encoder_dim=encoder_dim,
        decoder_dim=encoder_dim,
        patch_size=patch_size,
        n_layers=n_layers,
        heads=heads,
        dropout=dropout
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --------------------
    # 5. Entraînement
    # --------------------
    best_valid_loss = float("inf")

    for epoch in range(20):  # nb d’epochs court pour chaque essai
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, leave=False):
            optimizer.zero_grad()   
            # Forward pass
            X_pred, X_true = _model_foward(model, batch, normalization, mean, std)
            # Calcul de la loss et backpropagation
            loss = criterion(X_pred, X_true)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                X_pred, X_true = _model_foward(model, batch, normalization, mean, std)
                loss = criterion(X_pred, X_true)
                valid_loss += loss.item()

        valid_loss /= len(valid_loader)

        # Log W&B
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        })

        # Pruning Optuna
        trial.report(valid_loss, epoch)
        if trial.should_prune():
            wandb.finish()
            raise optuna.TrialPruned()

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

    wandb.log({"best_valid_loss": best_valid_loss})
    wandb.finish()

    return best_valid_loss


# ---------------------------
# 6. Boucle principale Optuna
# ---------------------------
if __name__ == "__main__":
    wandb_callback = WeightsAndBiasesCallback(metric_name="best_valid_loss")
    sampler = optuna.samplers.TPESampler(n_startup_trials=10)
    # Les 10 permiers trails sont purement aléatoires (exploration)
    # Ensuite, Optuna commence à exploiter les résultats passés pour concentrer les essais sur les combinaisons qui minimisent la valid_loss
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    study.optimize(objective, n_trials=50, callbacks=[wandb_callback])

    print("Best trial:")
    best_trial = study.best_trial
    print(f"Loss: {best_trial.value}")
    print("Params:")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")
