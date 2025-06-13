import os
import json
import torch
from torch.utils.data import DataLoader
from datasets.cwru_dataset import CWRUDataset
from models.model import ViTAutoencoder
from utils.statistics import custom_loss

def train_model(model, train_loader, optimizer, loss_fn, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            X_true = batch['X_true'].to(device)
            X_tilde = batch['X_tilde'].to(device)

            optimizer.zero_grad()
            patch_emb, encoded_tokens, decoded_tokens = model(X_tilde)
            loss = loss_fn(X_true, decoded_tokens)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_loss:.4f}")

def test_model(model, test_loader, loss_fn, device, config_name):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            X_true = batch['X_true'].to(device)
            X_tilde = batch['X_tilde'].to(device)

            patch_emb, encoded_tokens, decoded_tokens = model(X_tilde)
            loss = loss_fn(X_true, decoded_tokens)
            test_loss += loss.item()
    print(f"Test Loss ({config_name}): {test_loss:.4f}")
    return test_loss

def main_pipeline():
    # === Lecture de la configuration depuis train_config.json ===
    config_path = 'train_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    # === Configuration générale ===
    root_dir = config['root_dir']
    model_name = config['model_name']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    epochs = config['epochs']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === Chargement des données d'entraînement ===
    train_dataset = CWRUDataset(
        root_dir=root_dir,
        source=config['source'],
        fault_filter=config['train_fault_filter'],
        transform_type='train'
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # === Préparation du modèle ===
    model = ViTAutoencoder(
        encoder_dim=config['encoder_dim'],
        decoder_dim=config['decoder_dim'],
        serie_len=config['serie_len'],
        patch_size=config['patch_size'],
        dropout=config['dropout'],
        n_layers=config['n_layers'],
        heads=config['heads']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = custom_loss

    # === Entraînement ===
    print("Début de l'entraînement...")
    train_model(model, train_loader, optimizer, loss_fn, device, epochs)

    # === Configurations de test ===
    test_configs = config['test_configs']

    # === Test pour chaque configuration ===
    results = {}
    for test_config in test_configs:
        print(f"Début du test pour la configuration : {test_config}")
        test_dataset = CWRUDataset(root_dir=root_dir, source=config['source'], **test_config)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_loss = test_model(model, test_loader, loss_fn, device, config_name=str(test_config))
        results[str(test_config)] = test_loss

    # === Sauvegarde des résultats ===
    results_path = os.path.join('results', model_name, 'test_results.txt')
    with open(results_path, 'w') as f:
        for test_config, loss in results.items():
            f.write(f"Configuration: {test_config}, Test Loss: {loss:.4f}\n")
    print(f"Résultats sauvegardés à : {results_path}")

    # === Sauvegarde du modèle ===
    save_path = os.path.join('results', model_name, 'model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Modèle sauvegardé à : {save_path}")

if __name__ == "__main__":
    main_pipeline()