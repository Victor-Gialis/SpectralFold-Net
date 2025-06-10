import torch
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from models.model import ViTAutoencoder
from torch.utils.data import DataLoader
from utils.statistics import global_stats, signal_normalization
from datasets.dataloader import get_dataset
import os
import matplotlib.pyplot as plt

# === 1. Chargement de la configuration ===
with open('configs/evaluate_config.json', 'r') as f:
    evaluate_config = json.load(f)

# Vérifier si une configuration spécifique pour le modèle existe dans le dossier results
model_name = evaluate_config.get('model_name', 'default_model')  # Nom du modèle à évaluer
results_config_path = f"results/{model_name}/used_config.json"

if os.path.exists(results_config_path):
    print(f"Configuration du modèle trouvée dans {results_config_path}. Utilisation de cette configuration.")
    with open(results_config_path, 'r') as f:
        model_config = json.load(f)

# === 2. Paramètres globaux ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_params = model_config['dataset']
dataloader_params = model_config['dataloader']
model_params = model_config['model']
training_params = model_config['training']

dataset_name = evaluate_config['dataset']['name']
evaluation_params = evaluate_config[f'{dataset_name}']

size = dataset_params['window_size']
recovery = dataset_params['stride'] / dataset_params['window_size']
fs = evaluation_params['sampling_frequency']
target_orders = evaluation_params['target_orders']

# Vérifier si des paramètres spécifiques pour le dataset d'évaluation existent
train_fault_filter = dataset_params.get('fault_filter', ['normal', 'inner', 'outer', 'ball'])
evaluate_fault_filter = evaluate_config['dataset'].get('fault_filter', ['normal', 'inner', 'outer', 'ball'])

train_dataset_name = dataset_params['name']
evaluate_dataset_name = evaluate_config['dataset']['name']

# Supprimer les clés inutiles de dataset_params
for key in ['name', 'fault_filter']:
    dataset_params.pop(key, None)

# === 3. Chargement des datasets ===
train_dataset = get_dataset(name=train_dataset_name, fault_filter=train_fault_filter, **dataset_params)

# Normalisation globale
mean, std = global_stats(train_dataset)
mean = mean.to(device)
std = std.to(device)

# Charger le dataset d'évaluation
dataset = get_dataset(name=evaluate_dataset_name, fault_filter=evaluate_fault_filter, **dataset_params)
dataset_loader = DataLoader(dataset, batch_size=dataloader_params['batch_size'], shuffle=False, collate_fn=dataset._collate_fn)

# === 4. Chargement du modèle ===
serie_len = dataset[0]['X_true'].shape[-1]  # Taille FFT
N = 2 * (serie_len -1)
freqs = np.fft.rfftfreq(N, d=1/fs)

model = ViTAutoencoder(
    encoder_dim=model_params['encoder_dim'],
    decoder_dim=model_params['decoder_dim'],
    serie_len=serie_len,
    patch_size=model_params['patch_size'],
    n_layers=model_params['n_layers'],
    heads=model_params['heads'],
    dropout=model_params['dropout']
).to(device)

model_path = os.path.join('checkpoint', model_name, 'model.pth')
model.load_state_dict(torch.load(model_path))
model.eval()

# === 5. Fonction pour calculer le ratio d'énergie ===
def compute_energy_ratio(signal, freqs, order_freq, running_speed, bandwidth=2, num_harmonics=10):
    running_freq = float(running_speed) / 60  # Convertir RPM en Hz
    global_energy = np.sum(signal**2)
    harmonics = order_freq * np.arange(1, num_harmonics + 1)
    ratio_freqs = freqs / running_freq
    bandwidth /= running_freq
    energy = 0.0

    for h in harmonics:
        idx = np.where((ratio_freqs >= h - bandwidth) & (ratio_freqs <= h + bandwidth))[0]
        if len(idx) > 0:
            energy += np.sum(signal[idx]**2)

    return energy / global_energy if global_energy > 0 else 0.0

# === 6. Évaluation ===
results_true = []
results_pred = []
results_corr = []
loss = []

for batch in tqdm(dataset_loader):
    metadata = pd.DataFrame(batch['metadata'])
    labels = batch['label']

    X_input = batch['X_tilde'].unsqueeze(1).to(device)  # (batch_size, 1, signal_length)
    X_true = batch['X_true'].unsqueeze(1).to(device)  # (batch_size, 1, signal_length)

    # Normalisation des signaux
    X_input = signal_normalization(X_input, mean, std)  # (batch_size, 1, signal_length)
    X_true = signal_normalization(X_true, mean, std)  # (batch_size, 1, signal_length)
    
    with torch.no_grad():
        _, _, X_pred = model(X_input)  # (batch_size, 1, signal_length)

    X_true = X_true.squeeze(1)
    X_pred = X_pred.cpu().numpy()
    X_true = X_true.cpu().numpy()

    loss.append(np.mean((X_true - X_pred)**2))

    batch_size, serie_size = X_pred.shape
    X_true = X_true[:, :serie_size]

    for signal_true, signal_pred, label, meta in zip(X_true, X_pred, labels, metadata.to_dict(orient='records')):
        energy_ratio_true = {
            name: compute_energy_ratio(signal=signal_true, freqs=freqs, order_freq=freq, running_speed=meta.get('speed', 0))
            for name, freq in target_orders.items()
        }

        energy_ratio_predict = {
            name: compute_energy_ratio(signal=signal_pred, freqs=freqs, order_freq=freq, running_speed=meta.get('speed', 0))
            for name, freq in target_orders.items()
        }

        result_true = {
            'label': label,
            **{key: meta.get(key, None) for key in meta.keys()},  # Inclure tous les métadonnées dynamiquement
            **energy_ratio_true,
            'dominant_fault': max(energy_ratio_true, key=energy_ratio_true.get)
        }

        result_pred = {
            'label': label,
            **{key: meta.get(key, None) for key in meta.keys()},  # Inclure tous les métadonnées dynamiquement
            **energy_ratio_predict,
            'dominant_fault': max(energy_ratio_predict, key=energy_ratio_predict.get)
        }

        results_true.append(result_true)
        results_pred.append(result_pred)

    # Calcul de la corrélation de Pearson
    for meta, corr in zip(metadata.to_dict(orient='records'), pearsonr(X_true, X_pred)[0]):
        results_corr.append({
            **{key: meta.get(key, None) for key in meta.keys()},  # Inclure tous les métadonnées dynamiquement
            'pearsonr': corr
        })

# === 7. Sauvegarde des résultats ===
results_dir = "results/evaluation/"
results_dir = os.path.join("results", model_name,"evaluation")
if not os.path.exists(results_dir):
    print(f"Création du dossier de résultats : {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

results_true = pd.DataFrame(results_true)
results_pred = pd.DataFrame(results_pred)
results_corr = pd.DataFrame(results_corr)

results_true.to_csv(os.path.join(results_dir, "fault_energy_ratio_true.csv"), index=False)
results_pred.to_csv(os.path.join(results_dir, "fault_energy_ratio_predict.csv"), index=False)
results_corr.to_csv(os.path.join(results_dir, "spectrum_correlation.csv"), index=False)

# Libérer la mémoire GPU
del model
torch.cuda.empty_cache()

# === 8. Visualisation des pertes ===
plt.figure()
plt.plot(loss)
plt.title("Évolution de la perte")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.savefig(os.path.join(results_dir, "loss.jpeg"))