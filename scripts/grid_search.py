import torch
import logging
import itertools
import json
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.model import ViTAutoencoder
from utils.statistics import signal_normalization, global_stats, mse_loss
from datasets.dataloader import get_dataset

# === 1. Chargement de la configuration ===
with open('configs/grid_search_config.json', 'r') as f:
    config = json.load(f)

# === 2. Paramètres globaux ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_name = config['dataset']['name']
dataset_params = {k: v for k, v in config['dataset'].items() if k != 'name'}
grid_params = config['grid_search']
training_params = config['training']

# === 3. Chargement du dataset ===
dataset = get_dataset(dataset_name, **dataset_params)
size_fft = dataset[0]['X_true'].shape[-1]

collate_fn = getattr(dataset, '_collate_fn', None)
if collate_fn is None:
    # fallback: use a default collate_fn if not present
    from torch.utils.data.dataloader import default_collate
    collate_fn = default_collate

# Split train/valid datasets
train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

# DataLoaders
batch_size = training_params['batch_size']
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Global statistics
mean, std = global_stats(train_dataset)
mean = mean.to(device)
std = std.to(device)
print('Global statistics: mean = {mean.shape}, std = {std.shape}')

# === 4. Définir les hyperparamètres à optimiser ===
param_combinations = list(itertools.product(*grid_params.values()))
print(f"Total combinations: {len(param_combinations)}")

# === 5. Configurer le logger ===
results_dir = "results/grid_search/"
os.makedirs(results_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(results_dir, 'training_logs.txt'), level=logging.INFO, format='%(asctime)s - %(message)s')

# === 6. Boucle sur chaque combinaison d'hyperparamètres ===
results = []
for i, params in enumerate(param_combinations):
    print(f'Combinaison : {i + 1}/{len(param_combinations)}')
    logging.info(f'Combinaison : {i + 1}/{len(param_combinations)}')

    # Extraire les hyperparamètres
    encoder_dim, decoder_dim, patch_size, n_layers, serie_len, heads, dropout, learning_rate = params

    print(f"Testing combination: encoder_dim={encoder_dim}, decoder_dim={decoder_dim}, "
          f"patch_size={patch_size}, serie_len={serie_len}, "
          f"n_layers={n_layers}, heads={heads}, dropout={dropout}, learning_rate={learning_rate}")
    logging.info(f"Testing combination: encoder_dim={encoder_dim}, decoder_dim={decoder_dim}, "
                 f"patch_size={patch_size}, serie_len={serie_len}, "
                 f"n_layers={n_layers}, heads={heads}, dropout={dropout}, learning_rate={learning_rate}")

    # Initialiser le modèle
    model = ViTAutoencoder(
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        serie_len=size_fft,
        patch_size=patch_size,
        n_layers=n_layers,
        heads=heads,
        dropout=dropout
    ).to(device)

    # Initialiser l'optimiseur et le scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Entraîner le modèle
    num_epochs = training_params['num_epochs']
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch in tqdm(train_loader):
            X_tilde = batch['X_tilde'].unsqueeze(1).to(device, non_blocking=True)
            X_true = batch['X_true'].unsqueeze(1).to(device, non_blocking=True)

            b,_,_ = X_true.shape

            # Normalisation des signaux
            X_tilde = signal_normalization(X_tilde, mean.expand(b, 1, -1), std.expand(b, 1, -1))
            X_true = signal_normalization(X_true, mean.expand(b, 1, -1), std.expand(b, 1, -1))
        
            _, _, X_pred = model(X_tilde)
            X_pred = X_pred.unsqueeze(1)

            X_true = X_true * std + mean
            X_pred = X_pred * std + mean

            loss = mse_loss(X_true, X_pred)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}: train loss = {epoch_loss / len(train_loader):.4f}")
        logging.info(f"Epoch {epoch + 1}: train loss = {epoch_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                X_tilde = batch['X_tilde'].unsqueeze(1).to(device, non_blocking=True)
                X_true = batch['X_true'].unsqueeze(1).to(device, non_blocking=True)

                b,_,_ = X_true.shape

                # Normalisation des signaux
                X_tilde = signal_normalization(X_tilde, mean.expand(b, 1, -1), std.expand(b, 1, -1))
                X_true = signal_normalization(X_true, mean.expand(b, 1, -1), std.expand(b, 1, -1))
            
                _, _, X_pred = model(X_tilde)
                X_pred = X_pred.unsqueeze(1)

                X_true = X_true * std + mean
                X_pred = X_pred * std + mean

                loss = mse_loss(X_true, X_pred)
                val_loss += loss.item()
                
        val_loss /= len(valid_loader)
        print(f"Epoch {epoch}: val loss = {val_loss:.4f}")
        logging.info(f"Validation loss = {val_loss:.4f}")

        results.append({
            'params': params,
            'val_loss': val_loss
        })

# === 7. Trouver la meilleure combinaison d'hyperparamètres ===
best_result = min(results, key=lambda x: x['val_loss'])
print(f"Best hyperparameters: {best_result['params']}, Validation loss: {best_result['val_loss']:.4f}")
logging.info(f"Best hyperparameters: {best_result['params']}, Validation loss: {best_result['val_loss']:.4f}")

# === 8. Sauvegarder les résultats ===
results_file = os.path.join(results_dir, 'results.csv')
pd.DataFrame(results).to_csv(results_file, index=False)
print(f"Results saved to {results_file}")