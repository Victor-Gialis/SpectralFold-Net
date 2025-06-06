import torch
import logging
import itertools
import json
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.model import ViTAutoencoder
from utils.statistics import signal_normalization, global_stats
from datasets.dataloader import get_dataset

# === 1. Chargement de la configuration ===
with open('configs/grid_search_config.json', 'r') as f:
    config = json.load(f)

# === 2. Paramètres globaux ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_params = config['dataset']
grid_params = config['grid_search']
training_params = config['training']

# === 3. Chargement du dataset ===
dataset = get_dataset(dataset_params['name'], **dataset_params)
mean, std = global_stats(dataset)
mean = mean.to(device)
std = std.to(device)

# Split train/valid datasets
train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

# DataLoaders
batch_size = training_params['batch_size']
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

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
        serie_len=serie_len,
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
            batch_loss = 0

            for signal_reduce, signal_complete in zip(batch['vibration_fft_reduce'], batch['vibration_fft_complete']):
                signal_reduce = signal_reduce.unsqueeze(0).unsqueeze(0).to(device)
                signal_complete = signal_complete.unsqueeze(0).unsqueeze(0).to(device)

                # Normaliser les signaux
                signal_reduce = signal_normalization(signal_reduce, mean, std)
                signal_complete = signal_normalization(signal_complete, mean, std)

                # Entraînement
                optimizer.zero_grad()
                _, _, predicted_signal = model(signal_reduce)
                predicted_signal = predicted_signal.unsqueeze(1)
                b, c, s = predicted_signal.shape

                loss = torch.mean((predicted_signal - signal_complete[:, :, :s])**2)
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()

            epoch_loss += batch_loss / len(batch['vibration_fft_reduce'])

        print(f"Epoch {epoch + 1}: train loss = {epoch_loss / len(train_loader):.4f}")
        logging.info(f"Epoch {epoch + 1}: train loss = {epoch_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                batch_loss = 0

                for signal_reduce, signal_complete in zip(batch['vibration_fft_reduce'], batch['vibration_fft_complete']):
                    signal_reduce = signal_reduce.unsqueeze(0).unsqueeze(0).to(device)
                    signal_complete = signal_complete.unsqueeze(0).unsqueeze(0).to(device)

                    signal_reduce = signal_normalization(signal_reduce, mean, std)
                    signal_complete = signal_normalization(signal_complete, mean, std)

                    _, _, predicted_signal = model(signal_reduce)
                    predicted_signal = predicted_signal.unsqueeze(1)
                    b, c, s = predicted_signal.shape

                    loss = torch.mean((predicted_signal - signal_complete[:, :, :s])**2)
                    batch_loss += loss.item()

                val_loss += batch_loss / len(batch['vibration_fft_reduce'])

        val_loss /= len(valid_loader)
        print(f"Validation loss = {val_loss:.4f}")
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