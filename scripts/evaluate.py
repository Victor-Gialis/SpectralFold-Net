import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from models.model import ViTAutoencoder
from torch.utils.data import DataLoader
from utils.utils import global_stats, signal_normalization, mse_loss
from data.data.dataloader import CWRUDataset, custom_collate_fn

size = 2**14 # 16384
recovery = 0.95 # 95% overlap between patches
fs = 12000 # Sampling frequency
factor = 2 # Downsampling factor
target_orders = {'BPFI': 4.9469, 'BPFO': 3.0530, 'FTF': 0.3817, 'BSF': 3.9874}

N = 2*(size//factor)
freqs = np.fft.rfftfreq(N, d=1/fs)

train_dataset = CWRUDataset(
    fault_filter=['normal', 'inner', 'outer'],
    window_size=size,
    stride=int(size * (1 - recovery)),
    downsample_factor=factor)

# Extract global statistic for data normalization
mean, std = global_stats(train_dataset)

dataset = CWRUDataset(
    fault_filter=['normal', 'inner', 'outer', 'ball'],
    window_size=size,
    stride=int(size * (1 - recovery)),
    downsample_factor=factor)

dataset_loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

# Load the model for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Get size of fft
serie_len = dataset[0]['vibration_fft_complete'].shape[-1] # 8193
# Load the model for inference
model = ViTAutoencoder(encoder_dim = 2**10, 
                       decoder_dim = 2**11, 
                       serie_len = serie_len,
                       patch_size = 64,
                       n_layers = 2,
                       heads = 4,
                       dropout=0.1).to(device)

model.load_state_dict(torch.load('models/model_v2.pth'))
model.eval()

# Load global statistics for data normalization
mean = mean.to(device)
std = std.to(device)

def compute_energy_ratio(signal, freqs, order_freq, running_speed, bandwidth=1, num_harmonics=10):
    running_freq = float(running_speed) / 60 # Convert RPM in Hz
    global_energy = np.sum(signal**2)
    harmonics = order_freq * np.arange(1, num_harmonics + 1)
    ratio_freqs = freqs / running_freq
    bandwidth /= running_freq
    energy = 0.0

    for h in harmonics :
        idx = np.where((ratio_freqs >= h - bandwidth) & (ratio_freqs <= h + bandwidth))[0]
        if len(idx) > 0:
            energy += np.sum(signal[idx]**2)

    return energy / global_energy if global_energy > 0 else 0.0

results_true = []
results_pred = []
results_corr = []
loss = []

for batch in tqdm(dataset_loader):
    speeds = batch['speed']
    faults = batch['fault']
    diameters = batch['diameter']
    positions = batch['position']

    X_input = torch.stack(batch['vibration_fft_reduce']).unsqueeze(1).to(device)  # (batch_size, 1, signal_length)
    X_true = torch.stack(batch['vibration_fft_complete']).unsqueeze(1).to(device)  # (batch_size, 1, signal_length)

    # Systematic Variation Normalization (SVN)
    X_input = signal_normalization(X_input, mean, std)  # (batch_size, 1, signal_length)
    X_true = signal_normalization(X_true, mean, std)  # (batch_size, 1, signal_length)
    
    with torch.no_grad():
        _, _, X_pred = model(X_input)  # (batch_size, 1, signal_length)

    X_true = X_true.squeeze(1)

    X_pred = X_pred.cpu().numpy()
    X_true = X_true.cpu().numpy()

    loss.append(np.mean((X_true - X_pred)**2))

    batch_size, serie_size = X_pred.shape
    X_true = X_true[:,:serie_size]

    for signal_true, signal_pred, speed, fault, diameter, position in zip(X_true, X_pred, speeds, faults, diameters, positions):

        # Compute energy default frequency
        energy_ratio_true = {
            name : compute_energy_ratio(signal=signal_true, freqs=freqs, order_freq=freq, running_speed=speed)
            for name, freq in target_orders.items()
        }           

        energy_ratio_predict = {
            name : compute_energy_ratio(signal=signal_pred, freqs=freqs, order_freq=freq, running_speed=speed)
            for name, freq in target_orders.items()
        }    

        result_true = {
            'speed': float(speed),
            'fault': fault,
            'position': position,
            'diameter': diameter,
            **energy_ratio_true,
            'dominant_fault': max(energy_ratio_true, key=energy_ratio_true.get)
        }

        result_pred = {
            'speed': float(speed),
            'fault': fault,
            'position': position,
            'diameter': diameter,
            **energy_ratio_predict,
            'dominant_fault': max(energy_ratio_predict, key=energy_ratio_predict.get)
        }

        results_true.append(result_true)
        results_pred.append(result_pred)
        
    # Compute the Pearson correlation between the true and predicted signals for each batch
    for speed, fault, position, diameter, corr in zip(speeds, faults, positions, diameters, pearsonr(X_true, X_pred, axis=1)[0]):
        results_corr.append({
            'speed': float(speed),
            'fault': fault,
            'position': position,
            'diameter': diameter,
            'pearsonr': corr
        })
    
results_true = pd.DataFrame(results_true)
results_pred = pd.DataFrame(results_pred)
results_corr = pd.DataFrame(results_corr)

results_true.to_csv("results/model_v1/fault_energy_ratio_true.csv")
results_pred.to_csv("results/model_v1/fault_energy_ratio_predict.csv")
results_corr.to_csv("results/model_v1/spectrum_correlation.csv")

del model
torch.cuda.empty_cache()

import matplotlib.pyplot as plt
plt.figure()
plt.plot(loss)
plt.show()
plt.savefig('loss.jpeg')