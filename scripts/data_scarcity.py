import os
import csv
import json
import torch
import wandb
import itertools
import datetime
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from models.model import DownstreamClassifier, Encoder
from datasets.dataloader import get_dataset
from utils.statistics import _log_norm

def plot_results(csv_path):
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)

    # Regrouper les résultats par configuration et pourcentage
    df_grouped = (
        df.groupby(["pretrain", "frozen", "labeled_percentage"])
          .agg(mean_f1=("F1_score", "mean"),
               std_f1=("F1_score", "std"),
               n=("F1_score", "count"))
          .reset_index()
    )

    plt.figure(figsize=(10, 6))

    # Tracer une courbe par combinaison pretrain/frozen
    for (pretrain, finetune), sub_df in df_grouped.groupby(["pretrain", "frozen"]):
        if not pretrain and not finetune :
            color = 'orange'

        elif not pretrain and finetune :
            color = 'blue'

        elif pretrain and not finetune :
            color = 'green'

        elif pretrain and finetune :
            color = 'red'

        label = f"Pretrain={pretrain}, Finetune={finetune}"
        plt.errorbar(
            sub_df["labeled_percentage"],
            sub_df["mean_f1"],
            yerr=sub_df["std_f1"],
            label=label,
            marker='o',
            capsize=3,
            color=color
        )
        plt.fill_between(sub_df["labeled_percentage"], 
                         [y_i - e_i for y_i, e_i in zip(sub_df["mean_f1"], sub_df["std_f1"])], 
                         [y_i + e_i for y_i, e_i in zip(sub_df["mean_f1"], sub_df["std_f1"])], 
                         color=color, 
                         alpha=0.2)


    plt.title("F1 Score (moyenne sur les seeds) en fonction du pourcentage de données étiquetées")
    plt.xlabel("Pourcentage de X_train utilisé pour l'entraînement")
    plt.ylabel("Test F1 Score moyen")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Sauvegarder le graphique
    fig_path = Path(csv_path).with_suffix('.png')
    plt.savefig(fig_path)
    plt.close()

def make_strata(dataset):
    strata = []
    for sample in tqdm(dataset):
        cls = sample['label']
        speed = sample['metadata']['speed']
        strata.append(f'{cls}_{speed}')
    strata =np.array(strata)
    return strata

def classification_metrics(targets, predictions) :
    # F1-score pondéré
    f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
    # Accuracy
    acc = accuracy_score(targets, predictions,)
    # ROC AUC multi-classes
    try:
        roc_auc = roc_auc_score(targets, predictions, multi_class='ovr', average='weighted')
    except ValueError:
        # Cas où une classe est absente dans le batch
        roc_auc = float('nan')

    return f1, acc, roc_auc

def evaluation(device, model, lb, batch):
    # X_true = batch['X_true'].unsqueeze(1).to(device, non_blocking=True)
    X_true = batch['X_tilde'].unsqueeze(1).to(device, non_blocking=True)
    labels = batch['label']

    # One-hot encode labels
    y_true = lb.transform(labels)
    y_true = torch.tensor(y_true, dtype=torch.float32).to(device, non_blocking=True)

    # Normalisation des signaux
    X_true_norm = _log_norm(x=X_true)

    # Prédiction du modèle
    y_pred = model(X_true_norm)

    # Convertir les prédictions et les vraies étiquettes 
    predictions = torch.argmax(y_pred, dim=1)
    targets = torch.argmax(y_true, dim=1)

    # Calcul de la loss et du F1-score
    loss = torch.nn.functional.cross_entropy(input=y_pred, target=targets)

    # Convertir en numpy pour le calcul du F1-score
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    return loss, targets, predictions

def append_result(csv_path: Path, seed: int, pretrain: bool, frozen: bool, labeled_percentage: float, f1: float, acc: float, roc_auc: float):
    """Écrit les résultats dans le fichier CSV commun."""
    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["seed", "pretrain", "frozen", "labeled_percentage", "f1_score","accuracy","roc_auc"])
        writer.writerow([seed, pretrain, frozen, labeled_percentage, f1, acc, roc_auc])

def load_dataset(dataset_name:str):
    # Créer le dossier pour sauvegarder les résultats
    filename = os.path.basename(__file__).split('.')[0]
    folder_experiment= os.path.join('results','downstream',filename)
    os.makedirs(folder_experiment, exist_ok=True)
    
    # Configurer le device CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Charger la config
    with open(f'configs/{filename}_config.json', 'r') as f:
        downstream_config = json.load(f)
    
    training_params = downstream_config['training']
    
    # Instancier le dataset via la factory
    dataset_params = {k: v for k, v in downstream_config['dataset'].items() if k != 'name'}
    dataset = get_dataset(dataset_name, **dataset_params)
    
    # Cr DataLoaders
    collate_fn = getattr(dataset, '_collate_fn', None)
    if collate_fn is None:
        # fallback: use a default collate_fn if not present
        from torch.utils.data.dataloader import default_collate
        collate_fn = default_collate
    
    # Instancier le modèle pré-entraîné et charger les poids
    pretrain_model_name = downstream_config['pretrained_model'].get('model_name', 'default_pretrain_model')

    # Vérifier que le checkpoint existe
    if not os.path.exists(f'checkpoint/{pretrain_model_name}/model_encoder.pth'):
        raise FileNotFoundError(f"Checkpoint not found for {pretrain_model_name}")

    # Charger la config du pré-entraînement pour les hyperparamètres
    with open(f'results/pretrain/{pretrain_model_name}/used_config.json', 'r') as f:
        pretrain_config = json.load(f)

    # Charger les paramètres du modèle pré-entraîné
    pretrain_params = pretrain_config['model']
    
    # Charger les paramètres d'entraînement et ajuster le learning rate en fonction du batch size
    batch_size = downstream_config.get('dataloader', {}).get('batch_size', 16)
    training_params = downstream_config['training']
    epochs = training_params.get('epochs', 100)
    learning_rate = training_params.get('learning_rate', 1e-3) * batch_size / 256
    
    # Taille des spectres d'entré
    input_size = dataset[0]['X_true'].shape[-1]
    print(f"Dataset '{dataset_name}' loaded with {len(dataset)} samples. FFT size: {input_size}")

    return dataset, collate_fn, pretrain_params, device, pretrain_model_name, batch_size, learning_rate, epochs, training_params, input_size

def train(seed:int, labeled_percentage:float , pretrain:bool, finetune:bool, results_file: Path):
    global dataset, collate_fn, pretrain_params, device, pretrain_model_name, batch_size, learning_rate, epochs, training_params, input_size

    print("=" * 60)
    print(f"▶️  Training run:")
    print(f"    seed              : {seed}")
    print(f"    labeled_percentage: {labeled_percentage}")
    print(f"    pretrain          : {pretrain}")
    print(f"    finetune          : {finetune}")
    print("=" * 60)
    
    start = datetime.datetime.now()
    
    # Créer train, valid, test splits avec stratification
    indice = np.arange(len(dataset))
    strata = make_strata(dataset)
    
    train_idx, test_val_idx = train_test_split(
        indice,
        test_size = 0.4,
        stratify=strata,
        random_state=42
    )
    
    valid_idx, test_idx = train_test_split(
        test_val_idx,
        test_size = 0.5,
        stratify=strata[test_val_idx],
        random_state=42
    )
    
    # Boucle sur les pourcentages de données étiquetées
    # Pas de random state pour avoir une variabilité entre les seeds
    if labeled_percentage < 1.0:
        _, scarcity_train_idx = train_test_split(
        train_idx,
        test_size = labeled_percentage,
        stratify=strata[train_idx],
    )
        
    else :
        scarcity_train_idx = train_idx
    
    # Sous datasets stratifiés
    train_dataset = Subset(dataset, scarcity_train_idx)
    valid_dataset = Subset(dataset, valid_idx)
    test_dataset = Subset(dataset, test_idx)
    
    # Pondération des classes pour le sampler
    labels = [sample['label'] for sample in tqdm(train_dataset)]
    classes, class_counts = np.unique(labels, return_counts=True)
    
    class_weights = 1. / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = {cls: weight for cls, weight in zip(classes, class_weights)}
    
    sample_weights = [class_weights[label] for label in labels]
    sample_weights = torch.DoubleTensor(sample_weights)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Création des dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # One hot encoding des classes pour le calcul de la cross entropy
    lb = LabelBinarizer()
    lb.fit(classes)

    # Chargement du backbone (encodeur)
    backbone = Encoder(
        num_patch= input_size // pretrain_params.get('patch_size', 64),
        patch_size= pretrain_params.get('patch_size', 64),
        encoder_dim= pretrain_params.get('encoder_dim', 128),
        n_layers= pretrain_params.get('n_layers', 4),
        heads= pretrain_params.get('heads', 8),
        dropout= pretrain_params.get('dropout', 0.4),
    ).to(device)

    # Downstrean factory
    if not pretrain and not finetune :
        # Backbone non pré-entraîné et gelé
        model = DownstreamClassifier(
            backbone= backbone,
            num_classes= len(classes),
            freeze_backbone= True,
        ).to(device)

    elif not pretrain and finetune :
        # Backbone non pré-entraîné et entraînable
        model = DownstreamClassifier(
            backbone= backbone,
            num_classes= len(classes),
            freeze_backbone= False,
        ).to(device)

    elif pretrain and not finetune :
        # Backbone pré-entraîné et gelé
        backbone.load_state_dict(torch.load(f'checkpoint/{pretrain_model_name}/model_encoder.pth'))
        model = DownstreamClassifier(
            backbone= backbone,
            num_classes= len(classes),
            freeze_backbone= True,
        ).to(device)

    elif pretrain and finetune :
        # Backbone pré-entraîné et entraînable
        backbone.load_state_dict(torch.load(f'checkpoint/{pretrain_model_name}/model_encoder.pth'))
        model = DownstreamClassifier(
            backbone= backbone,
            num_classes= len(classes),
            freeze_backbone= False,
        ).to(device)
    
    # Optimizer et scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr= learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_params.get('epochs', 100), eta_min=1e-6)

    # Sauvegarder les métriques
    all_train_loss = []
    all_valid_loss = []
    
    # Boucle d'entraînement et de validation
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0
        train_score = 0

        # Boucle d'entraînement
        for batch in train_loader:
            optimizer.zero_grad()
            loss, targets, predictions = evaluation(device=device, model=model, batch=batch, lb=lb)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        all_train_loss.append(train_loss)

        # Boucle de validation
        model.eval()
        valid_loss = 0

        for batch in valid_loader:
            loss, targets, predictions = evaluation(device=device, model=model, batch=batch, lb=lb)

            valid_loss += loss.item()

        valid_loss /= len(valid_loader)
        all_valid_loss.append(valid_loss)

        scheduler.step() 

    stop = datetime.datetime.now()
    print(f"Training time: {stop - start}")

    # Boucle de test (évaluation finale)
    model.eval()
    test_loss = 0

    all_predictions = []
    all_targets = []

    for batch in test_loader:
        loss, targets, predictions = evaluation(device=device, model=model, batch=batch, lb=lb)

        all_predictions.extend(predictions)
        all_targets.extend(targets)

        test_loss += loss.item()
    
    test_loss /= len(test_loader)

    # Calcul des métriques de classification
    f1, acc, roc_auc = classification_metrics(all_targets, all_predictions)

    # Sauvegarder les résultats dans le fichier CSV
    append_result(results_file, seed, pretrain, finetune, labeled_percentage, f1, acc, roc_auc)

    # Sauvegarder le modèle
    # [Architecture]_[Dataset]_[Date]_[Seed]_[Ratio]_[Init]_[Downstream]_[Version].pt
    architecture = "SpectralFold"
    dataset_name = "LASPI"
    init = "Pretrain" if pretrain else "Scratch"
    downstream = "Finetune" if finetune else "Frozen"

    # Créer le dossier dans WORK si besoin
    save_dir = os.path.join(os.environ['WORK'], 'checkpoints', f'{dataset_name}')
    os.makedirs(save_dir, exist_ok=True)

    # Nom du fichier
    model_path = os.path.join(save_dir, f'{architecture}_{dataset_name}_{seed}_{int(100*labeled_percentage)}_{init}_{downstream}_v1.pt')

    # Sauvergarde du modèle Pytorch
    torch.save(model.state_dict(),model_path)
    
    del model

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="Data scarcity experiments")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--labeled_percentage", type=float, required=True)
    
    # --- MODIFICATION CLÉ : Utiliser strtobool pour la conversion ---
    # Cette fonction convertit correctement 'True', '1', 't', 'y', 'yes' en True,
    # et 'False', '0', 'f', 'n', 'no' en False (insensible à la casse).
    try:
        from distutils.util import strtobool
        bool_type = lambda x: bool(strtobool(x))
    except ImportError:
        # Fallback pour les environnements où distutils est déprécié (Python 3.12+)
        def bool_type(x):
            if x.lower() in ('true', '1', 't', 'y', 'yes'):
                return True
            elif x.lower() in ('false', '0', 'f', 'n', 'no'):
                return False
            raise argparse.ArgumentTypeError(f"Valeur booléenne attendue : {x}")

    parser.add_argument("--pretrain", type=bool_type, required=True,help="Initialisation du backbone: True pour pré-entraîné, False pour aléatoire (Scratch).")
    parser.add_argument("--finetune", type=bool_type, required=True,help="Entraînement du backbone: True pour Finetune, False pour gelé (Frozen).")
    parser.add_argument("--results_file", type=str, default="results.csv", help="Chemin du fichier CSV des résultats")
    parser.add_argument("--dataset", type=str, default="CWRU", help="Choix du dataset d'entraînement")

    args = parser.parse_args()

    # --- Charger le dataset et les configurations ---
    dataset, collate_fn, pretrain_params, device, pretrain_model_name, batch_size, learning_rate, epochs, training_params, input_size = load_dataset(dataset_name=args.dataset)

    # --- Boucle sur les différentes configurations ---
    train(
        seed=args.seed,
        labeled_percentage=args.labeled_percentage,
        pretrain=args.pretrain,
        finetune=args.finetune,
        results_file=args.results_file
    )