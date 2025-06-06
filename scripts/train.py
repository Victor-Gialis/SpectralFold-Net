import torch
import json
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.model import ViTAutoencoder
from datasets.dataloader import get_dataset
from utils.statistics import signal_normalization, global_stats, mse_loss

# === 1. Charger la config ===
with open('configs/train_config.json', 'r') as f:
    config = json.load(f)

# === 2. Instancier le dataset via la factory ===
dataset_name = config['dataset']['name']
dataset_params = {k: v for k, v in config['dataset'].items() if k != 'name'}
dataset = get_dataset(dataset_name, **dataset_params)

# === 3. Normalisation globale ===
mean, std = global_stats(dataset)

# === 4. Split train/valid
train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

# === 5. DataLoader
batch_size = config.get('dataloader', {}).get('batch_size', 16)
collate_fn = getattr(dataset, '_collate_fn', None)
if collate_fn is None:
    # fallback: use a default collate_fn if not present
    from torch.utils.data.dataloader import default_collate
    collate_fn = default_collate

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# === 6. Instancier le modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
size_fft = dataset[0]['X_true'].shape[-1]
model_params = config['model']
model = ViTAutoencoder(
    encoder_dim=model_params.get('encoder_dim', 2**10),
    decoder_dim=model_params.get('decoder_dim', 2**11),
    serie_len=size_fft,
    patch_size=model_params.get('patch_size', 128),
    n_layers=model_params.get('n_layers', 2),
    heads=model_params.get('heads', 4),
    dropout=model_params.get('dropout', 0.1)
).to(device)

mean = mean.to(device)
std = std.to(device)

# === 7. Optimizer & Scheduler
training_params = config['training']
optimizer = torch.optim.AdamW(model.parameters(), lr=training_params.get('learning_rate', 1e-4))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# === 8. Training loop
epochs = training_params.get('epochs', 100)
save_path = training_params.get('save_path', 'results/model_v1/')
os.makedirs(save_path, exist_ok=True)

train_loss = []
valid_loss = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        X_tilde = batch['X_tilde'].unsqueeze(1).to(device, non_blocking=True)
        X_true = batch['X_true'].unsqueeze(1).to(device, non_blocking=True)

        # Normalisation des signaux
        X_tilde = signal_normalization(X_tilde, mean, std)
        X_true = signal_normalization(X_true, mean, std)

        optimizer.zero_grad() # Reset gradients

        # Forward pass
        _, _, X_pred = model(X_tilde)
        X_true = X_true.squeeze(1) # Remove channel dimension

        loss = mse_loss(X_true, X_pred)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    train_loss.append(epoch_loss / len(train_loader))
    print(f"Epoch {epoch}: train loss = {epoch_loss / len(train_loader):.4f}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in valid_loader:
            X_tilde = batch['X_tilde'].unsqueeze(1).to(device)
            X_true = batch['X_true'].unsqueeze(1).to(device)

            # Normalisation des signaux
            X_tilde = signal_normalization(X_tilde, mean, std)
            X_true = signal_normalization(X_true, mean, std)

            _, _, X_pred = model(X_tilde)
            X_true = X_true.squeeze(1) # Remove channel dimension

            loss = mse_loss(X_true, X_pred)
            val_loss += loss.item()
            
    val_loss /= len(valid_loader)
    valid_loss.append(val_loss)
    print(f"Epoch {epoch}: val loss = {val_loss:.4f}")

    scheduler.step()

# === 9. Sauvegarde du modèle et des courbes de loss

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(train_loss, label='Train Loss')
plt.plot(valid_loss, label='Validation Loss')
plt.title('Loss Evolution')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()

# Récupère le nom du sous-dossier depuis la config
save_name = config['training']['model_name'].strip('/\\')

# Dossiers pour checkpoints et résultats
checkpoint_dir = os.path.join('checkpoint', save_name)
results_dir = os.path.join('results', save_name)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Sauvegarde du modèle
torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model.pth'))

# Sauvegarde des hyperparamètres/config utilisés
with open(os.path.join(results_dir, 'used_config.json'), 'w') as f:
    json.dump(config, f, indent=2)

# Exemple pour sauvegarder une courbe de loss dans results
plt.savefig(os.path.join(results_dir, 'loss_curve.png'))
plt.close()