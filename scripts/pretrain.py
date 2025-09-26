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

from tqdm import tqdm
from torch.utils.data import DataLoader
from models.model import PretrainedModel
from datasets.dataloader import get_dataset
from utils.statistics import _z_norm, _log_norm, _log_denorm, global_stats, mse_loss

def _model_foward(model, batch, device, mean, std):
    X_tilde = batch['X_tilde'].unsqueeze(1).to(device, non_blocking=True)
    X_true = batch['X_true'].unsqueeze(1).to(device, non_blocking=True)
    
    metadata = pd.DataFrame(batch['metadata'])
    metadata['label'] = batch['label']

    # Get batch size
    b,_,_ = X_true.shape

    # Normalisation des signaux
    X_tilde_norm = _z_norm(X_tilde, mean.expand(b, 1, -1), std.expand(b, 1, -1))

    # Prédiction du modèle
    X_pred_norm  = model(X_tilde_norm )
    X_pred_norm  = X_pred_norm .unsqueeze(1)

    # Dénormalisation des signaux
    X_pred = X_pred_norm * std + mean

    return X_pred, X_true, X_tilde, metadata

# Définir le device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger la config
with open('configs/pretrain_config.json', 'r') as f:
    config = json.load(f)

# Instancier le dataset via la factory
dataset_name = config['dataset']['name']
dataset_params = {k: v for k, v in config['dataset'].items() if k != 'name'}
dataset = get_dataset(dataset_name, **dataset_params)

# Taille des spectres FFT
size_fft = dataset[0]['X_true'].shape[-1]
print(f"Dataset '{dataset_name}' loaded with {len(dataset)} samples. FFT size: {size_fft}")

# Récupère le nom du sous-dossier depuis la config
save_name = config['training']['model_name'].strip('/\\')

# Dossiers pour checkpoints et résultats
checkpoint_dir = os.path.join('checkpoint', save_name)
results_dir = os.path.join('results', save_name)

# Crée les dossiers s'ils n'existent pas
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Split train/valid/test
train_size = int(0.7 * len(dataset))
valid_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - valid_size

generator =torch.Generator().manual_seed(42)
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size], generator=generator)

# Extract globale statistics from train dataset
mean, std = global_stats(train_dataset)
mean = mean.to(device)
std = std.to(device)

# Split DataLoaders
batch_size = config.get('dataloader', {}).get('batch_size', 16)
collate_fn = getattr(dataset, '_collate_fn', None)
if collate_fn is None:
    # fallback: use a default collate_fn if not present
    from torch.utils.data.dataloader import default_collate
    collate_fn = default_collate

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Instancier le modèle en autosupervision

model_params = config['model']
model = PretrainedModel(
    input_size=size_fft,
    encoder_dim=model_params.get('encoder_dim', 128),
    decoder_dim=model_params.get('decoder_dim', 256),
    patch_size=model_params.get('patch_size', 64),
    n_layers=model_params.get('n_layers', 4),
    heads=model_params.get('heads', 8),
    dropout=model_params.get('dropout', 0.4)
).to(device)

# Optimizer & Scheduler
training_params = config['training']
epochs = training_params.get('epochs', 100)
learning_rate = training_params.get('learning_rate', 1e-3) * batch_size / 256
optimizer = torch.optim.AdamW(model.parameters(), lr= learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_params.get('epochs', 100), eta_min=1e-6)

# Training loop
save_path = training_params.get('save_path', 'results/model_v0/')
os.makedirs(save_path, exist_ok=True)

train_loss = []
valid_loss = []

# Start training
start = datetime.datetime.now()
logging.basicConfig(filename=os.path.join(results_dir, 'training_logs.txt'), level=logging.INFO, format='%(asctime)s - %(message)s')

loss_function = torch.nn.MSELoss()

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    train_results = list()

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        optimizer.zero_grad() # Remise à zéro des gradients

        X_pred, X_true, X_tilde, metadata = _model_foward(model, batch, device, mean, std)

        # Calcul de la loss
        loss = loss_function(X_pred, X_true)
        loss.backward()

        # Gradient clipping
        optimizer.step()
        epoch_loss += loss.item()

        train_results.append(metadata)

    train_loss.append(epoch_loss / len(train_loader))
    print(f"Epoch {epoch}: train loss = {epoch_loss / len(train_loader):.4f}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(valid_loader):

            X_pred, X_true, X_tilde, metadata = _model_foward(model, batch, device, mean, std)

            # Calcul de la loss
            loss = loss_function(X_pred, X_true)
            val_loss += loss.item()
            
    val_loss /= len(valid_loader)
    valid_loss.append(val_loss)
    print(f"Epoch {epoch}: val loss = {val_loss:.4f}")

    scheduler.step()

# End of training
end = datetime.datetime.now()
interval = (end - start)

for label in metadata['label'].unique():
    frame = metadata[metadata['label'] == label]
    idx = np.random.choice(frame.index)

    x_true = X_true[idx, 0, :].detach().cpu().numpy()
    x_tilde = X_tilde[idx, 0, :].detach().cpu().numpy()
    x_pred = X_pred[idx, 0, :].detach().cpu().numpy()

    freq_axis = np.arange(len(x_true))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq_axis, y=x_true, mode='lines', name='True Signal'))
    fig.add_trace(go.Scatter(x=freq_axis, y=x_tilde, mode='lines', name='Tilde Signal'))
    fig.add_trace(go.Scatter(x=freq_axis, y=x_pred, mode='lines', name='Predicted Signal'))
    fig.update_layout(title=f'Validation batch - Signal Comparison - Label: {label}',
                        xaxis_title='Frequency [Hz]',
                        yaxis_title='Amplitude')
    fig.write_html(os.path.join(results_dir, f'comparison_label_{label}.html'))

# Sauvegarde du modèle et des courbes de loss
plt.figure(figsize=(10, 5))
plt.plot(train_loss, label='Train Loss')
plt.plot(valid_loss, label='Validation Loss')
plt.title('Loss Evolution')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()

test_result = list()
# Plot la loss en fonction des classes et de la vitesse
for batch in tqdm(test_loader):

    X_pred, X_true, X_tilde, metadata = _model_foward(model, batch, device, mean, std)

    # Calcul de la loss par échantillon
    loss = torch.mean((X_true - X_pred)**2,dim=-1).reshape(-1)
    metadata['loss']=loss.detach().cpu().numpy()
    test_result.append(metadata)

train_results = pd.concat(train_results, ignore_index=True)
test_result = pd.concat(test_result, ignore_index=True)

fig = px.box(test_result, x='label',y='loss',color='speed')
fig.write_html(os.path.join(results_dir, 'loss_by_label_speed.html'))

# Training delay
logging.info(f"Training time: {interval}")

# Reconstruction by class and speed
logging.info(f"mean - MSE per class : {test_result.groupby(['speed','label'])['loss'].mean()}")
logging.info(f"std - MSE per class : {test_result.groupby(['speed','label'])['loss'].std()}")
logging.info(f"train individual per class : {train_results.groupby(['speed','label'])['source'].count()}")

# Sauvegarde du modèle
torch.save(model.encoder.state_dict(), os.path.join(checkpoint_dir, 'model.pth'))

# Sauvegarde des hyperparamètres/config utilisés
with open(os.path.join(results_dir, 'used_config.json'), 'w') as f:
    json.dump(config, f, indent=2)

# Exemple pour sauvegarder une courbe de loss dans results
plt.savefig(os.path.join(results_dir, 'loss_curve.png'))
plt.close()