import os
import json
import torch
import numpy as np
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from models.model import DownstreamClassifier, Encoder
from datasets.dataloader import get_dataset

# ----------------------------
# Config / modèle / dataset
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'checkpoint/SpectralFold_CWRU_1_10_Pretrain_Finetune_v1.pt'
dataset_name = model_path.split('_')[1]

patch_size = 16
batch_size = 64  # mini-batch pour forward
layer_idx = 2    # couche d'attention à inspecter

backbone = Encoder(
    num_patch=1024 // patch_size,
    patch_size=patch_size,
    encoder_dim=512,
    n_layers=3,
    heads=8,
    dropout=0.2565
)

model = DownstreamClassifier(backbone=backbone, num_classes=10).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Chargement dataset
with open('configs/data_scarcity_config.json', 'r') as f:
    data_config = json.load(f)

dataset_params = {k: v for k, v in data_config['dataset'].items() if k != 'name'}
dataset = get_dataset(name=dataset_name, **dataset_params)

# Dossier de résultats
results_dir = f'results/downstream/{dataset_name}/results_attention_plots'
os.makedirs(results_dir, exist_ok=True)

# ----------------------------
# Construction bib_sample
# ----------------------------
bib_sample = {}
for sample in dataset:
    if dataset_name == 'CWRU':
        speed = str(sample['metadata'].get('speed', None))
    elif dataset_name == 'LASPI':
        speed = str(sample['metadata'].get('freq', None))
    classe = sample['label']
    bib_sample.setdefault(speed, {})
    bib_sample[speed].setdefault(classe, [])
    bib_sample[speed][classe].append(sample['X_true'])

# Stack tensors par (speed, classe)
for speed in bib_sample:
    for classe in bib_sample[speed]:
        stacked = torch.stack(bib_sample[speed][classe], dim=0)
        if stacked.dim() == 3 and stacked.size(1) == 1:
            stacked = stacked.squeeze(1)
        bib_sample[speed][classe] = stacked

# ----------------------------
# Utils Plotly
# ----------------------------
cmap = cm.get_cmap('viridis')
def value_to_rgb_str(v):
    v = np.clip(v, 0.0, 1.0)
    rgba = cmap(float(v))
    rgb = tuple(int(255 * x) for x in rgba[:3])
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

# ----------------------------
# Boucle principale
# ----------------------------
FE = 6000 if dataset_name == 'CWRU' else 12800
SPEED = '1730' if dataset_name == 'CWRU' else '45'

if SPEED not in bib_sample:
    raise ValueError(f"SPEED {SPEED} not found. Available: {list(bib_sample.keys())}")

for classe in sorted(bib_sample[SPEED].keys()):
    X_all = bib_sample[SPEED][classe]  # (N, L)
    N, L = X_all.shape

    # ----------------------------
    # Forward en batch et collecte des attentions
    # ----------------------------
    all_attn_list = []
    for i in range(0, N, batch_size):
        batch = X_all[i:i+batch_size].to(device)
        batch_in = batch.unsqueeze(1) if batch.dim() == 2 else batch

        with torch.no_grad():
            _ = model(batch_in)

        last_attn = model.backbone.encoder_layers[layer_idx][0].fn.fn.last_attn
        attn_np = last_attn.detach().cpu().numpy()

        if attn_np.ndim == 3:
            attn_per_sample = attn_np[:,0,:]  # head 0
        else:
            attn_per_sample = attn_np

        n_tokens = attn_per_sample.shape[1]
        expected_patches = L // patch_size
        if n_tokens == expected_patches + 1:
            attn_per_sample = attn_per_sample[:,1:]
        elif n_tokens != expected_patches:
            raise RuntimeError(f"Token count {n_tokens} doesn't match expected {expected_patches} (+1 CLS).")

        all_attn_list.append(attn_per_sample)

    attn_patches = np.vstack(all_attn_list)  # (N, num_patches)

    # ----------------------------
    # Moyennes
    # ----------------------------
    mean_signal = X_all.mean(dim=0).cpu().numpy()
    mean_attn_patch = attn_patches.mean(axis=0)
    norm_attn_patch = (mean_attn_patch - mean_attn_patch.min()) / (mean_attn_patch.max() - mean_attn_patch.min() + 1e-8)

    attn_upsampled = np.repeat(norm_attn_patch, patch_size)[:L]
    colors = [value_to_rgb_str(v) for v in attn_upsampled]
    x_axis = np.arange(L) * (FE / L)

    # ----------------------------
    # Plotly : spectre moyen coloré par attention
    # ----------------------------
    fig = go.Figure()
    for i in range(L-1):
        fig.add_trace(go.Scatter(
            x=[x_axis[i], x_axis[i+1]],
            y=[mean_signal[i], mean_signal[i+1]],
            mode='lines',
            line=dict(width=3, color=colors[i]),
            showlegend=False
        ))

    # Marker invisible pour colorbar
    mean_attn_val = float(norm_attn_patch.mean())
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(
            colorscale='Viridis',
            showscale=True,
            cmin=0, cmax=1,
            color=[mean_attn_val],
            colorbar=dict(title='Mean CLS Attention')
        ),
        showlegend=False
    ))

    fig.update_layout(
        title=f"Mean Spectrum – Class {classe} – Speed {SPEED} rpm",
        xaxis_title='Frequency [Hz]',
        yaxis_title='Amplitude'
    )

    out_html = os.path.join(results_dir, f'mean_attention_spectrum_class_{classe}_speed_{SPEED}rpm.html')
    fig.write_html(out_html)
    print(f"Saved {out_html}")

    # ----------------------------
    # Matplotlib : distribution CLS attention
    # ----------------------------
    attn_mean = attn_patches.mean(axis=0)
    attn_std = attn_patches.std(axis=0)

    plt.figure(figsize=(10,6))
    plt.fill_between(np.arange(len(attn_mean)), attn_mean-3*attn_std, attn_mean+3*attn_std, alpha=0.3)
    plt.plot(attn_mean, color='blue', label='Mean CLS Attention')
    plt.title(f'CLS Attention Distribution – Class {classe} – Speed {SPEED} rpm')
    plt.xlabel('Token Index')
    plt.ylabel('Attention Weight')
    plt.yscale('log')
    plt.ylim(1e-3, 1e-1)
    plt.legend()
    out_png = os.path.join(results_dir, f'cls_attention_distribution_class_{classe}_speed_{SPEED}rpm.png')
    plt.savefig(out_png)
    plt.close()
    print(f"Saved {out_png}")


