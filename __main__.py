import torch
import logging
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import plotly.express as px
import itertools

from tqdm import tqdm
from model import ViTAutoencoder
from torch.utils.data import DataLoader

from utils import signal_normalization, global_stats
from data.data.dataloader import CWRUDataset,custom_collate_fn

# Define the device for training and testing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Size of the input signal
# The input signal is a 1D time series of length 2^14 (16384 samples).
# The signal is divided into patches of size 1024 (2^10) samples, and the model processes these patches.
size = 2**14 # 16384
recovery = 0.95 # 95% overlap between patche

# The stride is set to 5% of the window size, which means that the model will process 95% of the signal in each patch.
# The window size is set to 1024 samples (2^10), which means that the model will process 1024 samples at a time.
# The stride is set to 5% of the window size, which means that the model will process 95% of the signal in each patch.
dataset = CWRUDataset(
    fault_filter=['normal', 'inner', 'outer'],
    window_size=size,
    stride=int(size * (1 - recovery)))

# Compute global mean and std for normalization
# This is done to normalize the input signals before feeding them into the model.
mean, std = global_stats(dataset)
mean = mean.to(device)
std = std.to(device)

# Create the training and validation datasets
# The dataset is split into training and validation sets using a random split.
train_size = int(0.8 * len(dataset)) # 80% pour l'entraînement
valid_size = len(dataset) - train_size # 20% pour la validation
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size]) # 80% train, 20% valid

# Create the test dataset
# The test dataset is created with a different fault filter (only 'ball' fault).
test_dataset = CWRUDataset(
    fault_filter=['ball'],
    window_size=size,
    stride=int(size * (1 - recovery)))

# Create DataLoader for training, validation, and test datasets
# The DataLoader is used to load the data in batches during training and evaluation.
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

# Get size of fft
size_fft = dataset[0]['vibration_fft_complete'].shape[-1] # 8193

# Définir les hyperparamètres à optimiser
param_grid = {
    'encoder_dim': [2**9, 2**10],  # Exemple : 512, 1024
    'decoder_dim': [2**11, 2**12],  # Exemple : 2048, 4096
    'patch_size': [64, 128],       # Taille des patches
    'n_layers': [1,2],           # Nombre de couches
    'serie_len': [size_fft],       # Longueur de la série
    'heads': [2,4],              # Nombre de têtes d'attention
    'dropout': [0.1],        # Taux de dropout
    'learning_rate': [1e-5, 1e-4] # Taux d'apprentissage
}

# Générer toutes les combinaisons d'hyperparamètres
param_combinations = list(itertools.product(*param_grid.values()))
print(f"Total combinations: {len(param_combinations)}")

# Stocker les résultats
results = []

# Configurer le logger
logging.basicConfig(filename='training_logs.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# Boucle sur chaque combinaison d'hyperparamètres
for i,params in enumerate(param_combinations):
    print(f'Combinaison : {i}/{len(param_combinations)}')

    # Enregister les hyperparamètres dans le logger
    logging.info(f'Combinaison : {i}/{len(param_combinations)}')

    
    # Extraire les hyperparamètres
    encoder_dim, decoder_dim, patch_size, n_layers, serie_len, heads, dropout, learning_rate = params

    print(f"Testing combination: encoder_dim={encoder_dim}, decoder_dim={decoder_dim}, "
          f"patch_size={patch_size}, serie_len={serie_len}, "
          f"n_layers={n_layers}, heads={heads}, dropout={dropout}, learning_rate={learning_rate}")
    
    # Enregistrer les hyperparamètres dans le logger
    logging.info(f"Testing combination: encoder_dim={encoder_dim}, decoder_dim={decoder_dim}, "
                 f"patch_size={patch_size}, serie_len={serie_len},  "
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

    # Initialiser l'optimiseur
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Entraîner le modèle (1 ou 10 époques pour tester rapidement)
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch in tqdm(train_loader):
            batch_loss = 0

            for signal_reduce, signal_complete in zip(batch['vibration_fft_reduce'], batch['vibration_fft_complete']):
                # Ajouter une dimension pour correspondre à l'entrée du modèle
                signal_reduce = signal_reduce.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, signal_length)
                signal_complete = signal_complete.unsqueeze(0).unsqueeze(0).to(device)  # (1, signal_length)

                # Normaliser les signaux
                signal_reduce = signal_normalization(signal_reduce, mean, std)
                signal_complete = signal_normalization(signal_complete, mean, std)

                # Entraînement
                optimizer.zero_grad()
                _, _, predicted_signal = model(signal_reduce)
                predicted_signal = predicted_signal.unsqueeze(1)
                b, c, s = predicted_signal.shape

                loss = torch.mean((predicted_signal - signal_complete[:,:,:s])**2)
                loss.backward()
                optimizer.step()

                # Ajouter la perte pour ce signal
                batch_loss += loss.item()

            epoch_loss += batch_loss/len(batch['vibration_fft_reduce'])

        print(f"Epoch {epoch}: train loss = {epoch_loss / len(train_loader):.4f}")

        # Enregistrer la perte d'entraînement dans le logger
        logging.info(f"Epoch {epoch}: train loss = {epoch_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                batch_loss = 0

                for signal_reduce, signal_complete in zip(batch['vibration_fft_reduce'], batch['vibration_fft_complete']):
                    signal_reduce = signal_reduce.unsqueeze(0).unsqueeze(0).to(device)
                    signal_complete = signal_complete.unsqueeze(0).unsqueeze(0).to(device)

                    # Normaliser les signaux
                    signal_reduce = signal_normalization(signal_reduce, mean, std)
                    signal_complete = signal_normalization(signal_complete, mean, std)

                    _, _, predicted_signal = model(signal_reduce)
                    predicted_signal = predicted_signal.unsqueeze(1)
                    b, c, s = predicted_signal.shape

                    # Compute loss for each signal
                    loss = torch.mean((predicted_signal - signal_complete[:,:,:s])**2)
                    batch_loss += loss.item()
                
                val_loss += batch_loss/len(batch['vibration_fft_reduce'])

        val_loss /= len(valid_loader)
        print(f"Validation loss = {val_loss:.4f}")

        # Enregistrer la perte de validation dans le logger
        logging.info(f"Validation loss = {val_loss:.4f}")

        # Enregistrer les résultats
        results.append({
            'params': params,
            'val_loss': val_loss
        })

# Trouver la meilleure combinaison d'hyperparamètres
best_result = min(results, key=lambda x: x['val_loss'])
print(f"Best hyperparameters: {best_result['params']}, Validation loss: {best_result['val_loss']:.4f}")

# Enregistrer la meilleure combinaison d'hyperparamètres dans le logger
logging.info(f"Best hyperparameters: {best_result['params']}, Validation loss: {best_result['val_loss']:.4f}")

# Sauvegarder les resultats dans un fichier CSV
import csv
with open('results.csv', 'w') as file:
    writer = csv.writer(file)
    for row in results:
        writer.writerow([row['params'], row['val_loss']])
    
