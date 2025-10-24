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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from models.model import PretrainedModel, DownstreamClassifier, Encoder
from datasets.dataloader import get_dataset
from utils.statistics import _z_norm, global_stats

def _make_strata(dataset):
    strata = []
    for sample in tqdm(dataset):
        cls = sample['label']
        speed = sample['metadata']['speed']
        strata.append(f'{cls}_{speed}')
    strata =np.array(strata)
    return strata

def _model_foward(model, batch, lb, device):
    X_true = batch['X_true'].unsqueeze(1).to(device, non_blocking=True)
    labels = batch['label']
    # One-hot encode labels
    y_true = lb.transform(labels)
    y_true = torch.tensor(y_true, dtype=torch.float32).to(device, non_blocking=True)
    # Get batch size
    b,_,_ = X_true.shape
    # Normalisation des signaux
    X_true_norm = _z_norm(X_true, mean.expand(b, 1, -1), std.expand(b, 1, -1))
    # Prédiction du modèle
    y_pred = model(X_true_norm)
    return y_pred, y_true

# Constantes globales
SCRATCH = 'scratch'
PRETRAIN = 'pretrain'
FROZEN = 'frozen'
FINETUNE = 'finetune'

start = datetime.datetime.now()

# Créer le dossier pour sauvegarder les résultats
filename = os.path.basename(__file__).split('.')[0]
folder_experiment= os.path.join('results','downstream',filename)
os.makedirs(folder_experiment, exist_ok=True)

# Configurer le device CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger la config
with open(f'configs/{filename}_config.json', 'r') as f:
    downstrean_config = json.load(f)

training_params = downstrean_config['training']
labeled_percentage = training_params.get('labeled_percentage', 1.0)

# Instancier le dataset via la factory
dataset_name = downstrean_config['dataset']['name']
dataset_params = {k: v for k, v in downstrean_config['dataset'].items() if k != 'name'}
dataset = get_dataset(dataset_name, **dataset_params)

# Cr DataLoaders
collate_fn = getattr(dataset, '_collate_fn', None)
if collate_fn is None:
    # fallback: use a default collate_fn if not present
    from torch.utils.data.dataloader import default_collate
    collate_fn = default_collate

# Instancier le modèle pré-entraîné et charger les poids
pretrain_model_name = downstrean_config['pretrained_model'].get('model_name', 'default_pretrain_model')

# Charger la config du pré-entraînement pour les hyperparamètres
with open(f'results/pretrain/{pretrain_model_name}/used_config.json', 'r') as f:
    pretrain_config = json.load(f)
pretrain_params = pretrain_config['model']

# Charger les paramètres d'entraînement et ajuster le learning rate en fonction du batch size
batch_size = downstrean_config.get('dataloader', {}).get('batch_size', 16)
training_params = downstrean_config['training']
epochs = training_params.get('epochs', 100)
learning_rate = training_params.get('learning_rate', 1e-3) * batch_size / 256

# Taille des spectres FFT
input_size = dataset[0]['X_true'].shape[-1]
print(f"Dataset '{dataset_name}' loaded with {len(dataset)} samples. FFT size: {input_size}")

# Créer train, valid, test splits avec stratification
indice = np.arange(len(dataset))
strata = _make_strata(dataset)

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

# Preparer le tableau de résultats
results = {"Labeled Percentage": [], 
           "Init Type": [], 
           "Downstream": [], 
           "Seeds":[],
           "Test Loss": [], 
           "Test F1 Score": []}

# Créer le dossier pour sauvegarder les résultats
folder_results = os.path.join(folder_experiment,f'{dataset_name}_dataset_{pretrain_model_name}_backbone')
os.makedirs(folder_results, exist_ok=True)

# Boucle sur les pourcentages de données étiquetées
for labeled_percentage in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0]:
    print(f"Labeled percentage: {labeled_percentage*100}%")

    folder_scarcity = os.path.join(folder_results,f'{int(100*labeled_percentage)}%_scarcity')
    os.makedirs(folder_scarcity, exist_ok=True)

    if labeled_percentage < 1.0:
        _, scarcity_train_idx = train_test_split(
        train_idx,
        test_size = labeled_percentage,
        stratify=strata[train_idx],
        random_state=42
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
    claas_weights = class_weights / class_weights.sum()
    class_weights = {cls: weight for cls, weight in zip(classes, class_weights)}

    sample_weights = [class_weights[label] for label in labels]
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Extraire les stats globales du dataset d'entraînement pour la normalisation
    mean, std = global_stats(train_dataset)
    mean = mean.to(device)
    std = std.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # One hot encoding des classes pour le calcul de la cross entropy
    lb = LabelBinarizer()
    lb.fit(classes)

    for init_type in [SCRATCH, PRETRAIN]:
    # for init_type in [PRETRAIN] :
        for downstream in [FROZEN, FINETUNE]:
        # for downstream in [FROZEN] :

            folder_model = os.path.join(folder_scarcity,f'{init_type}_init_{downstream}_downstream')
            os.makedirs(folder_model, exist_ok=True)
            
            # Répétition de plusieurs entraînement avec même config
            for seed in range(10) :
                folder_seed =os.path.join(folder_model,f'seed_{seed+1}')

                if os.path.exists(folder_seed):
                    print(f"Results for seed {seed+1} already exist. Skipping...")
                    continue
                
                else :
                    os.makedirs(folder_seed, exist_ok=True)

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
                if init_type == SCRATCH and downstream == FROZEN :
                    # Backbone non pré-entraîné et gelé
                    model = DownstreamClassifier(
                        backbone= backbone,
                        num_classes= len(classes),
                        freeze_backbone= True,
                    ).to(device)

                elif init_type == SCRATCH and downstream == FINETUNE :
                    # Backbone non pré-entraîné et entraînable
                    model = DownstreamClassifier(
                        backbone= backbone,
                        num_classes= len(classes),
                        freeze_backbone= False,
                    ).to(device)

                elif init_type == PRETRAIN and downstream == FROZEN :
                    # Backbone pré-entraîné et gelé
                    backbone.load_state_dict(torch.load(f'checkpoint/{pretrain_model_name}/model.pth'))
                    model = DownstreamClassifier(
                        backbone= backbone,
                        num_classes= len(classes),
                        freeze_backbone= True,
                    ).to(device)

                elif init_type == PRETRAIN and downstream == FINETUNE :
                    # Backbone pré-entraîné et entraînable
                    backbone.load_state_dict(torch.load(f'checkpoint/{pretrain_model_name}/model.pth'))
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

                all_train_score = []
                all_valid_score = []

                print(f"Starting training: Init type = {init_type}, Downstream = {downstream}, Seed = {seed+1}")
                # Boucle d'entraînement et de validation
                for epoch in tqdm(range(1,epochs+1), desc="Overall Training"):
                    model.train()
                    train_loss = 0
                    train_score = 0

                    for batch in train_loader:
                        optimizer.zero_grad()
                        y_pred, y_true = _model_foward(model=model, batch=batch, lb=lb, device=device)

                        predictions = torch.argmax(y_pred, dim=1).cpu().numpy()
                        targets = torch.argmax(y_true, dim=1).cpu().numpy()

                        loss = torch.nn.functional.cross_entropy(y_pred, y_true)
                        score = f1_score(targets, predictions, average='weighted', zero_division=0)

                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item()
                        train_score += score

                    train_loss /= len(train_loader)
                    train_score /= len(train_loader)

                    all_train_loss.append(train_loss)
                    all_train_score.append(train_score)

                    # print(f"Epoch {epoch}: train loss = {train_loss:.4f}")

                    model.eval()
                    valid_loss = 0
                    valid_score = 0

                    for batch in valid_loader:
                        
                        y_pred, y_true = _model_foward(model=model, batch=batch, lb=lb, device=device)

                        predictions = torch.argmax(y_pred, dim=1).cpu().numpy()
                        targets = torch.argmax(y_true, dim=1).cpu().numpy()

                        loss = torch.nn.functional.cross_entropy(y_pred, y_true)
                        score = f1_score(targets, predictions, average='weighted', zero_division=0)

                        valid_loss += loss.item()
                        valid_score += score

                        optimizer.zero_grad()

                    valid_loss /= len(valid_loader)
                    valid_score /= len(valid_loader)

                    all_valid_loss.append(valid_loss)
                    all_valid_score.append(valid_score)

                    # print(f"Epoch {epoch}: valid loss = {valid_loss:.4f}")

                    scheduler.step() # Mettre à jour le scheduler

                # Sauvegarde du modèle et des courbes de loss
                plt.figure(figsize=(10, 5))
                plt.plot(all_train_loss, label='Train Loss')
                plt.plot(all_valid_loss, label='Validation Loss')
                plt.title(f'Loss Evolution\nDownstream: {downstream}')
                plt.xlabel('Epochs')
                plt.ylabel('Cross entropy')
                plt.legend()
                plt.savefig(f'{folder_seed}/loss_curve.png')

                # Sauvegarde du modèle et des courbes de f1 score
                plt.figure(figsize=(10, 5))
                plt.plot(all_train_score, label='Train Score')
                plt.plot(all_valid_score, label='Validation Score')
                plt.title(f' Classification score Evolution\nDownstream: {downstream}')
                plt.xlabel('Epochs')
                plt.ylabel('F1 Score')
                plt.legend()
                plt.savefig(f'{folder_seed}/score_curve.png')

                plt.close('all')

                # Boucle de test (évaluation finale)
                model.eval()
                test_loss = 0
                test_score = 0

                all_predictions = []
                all_targets = []

                for batch in test_loader:
                    
                    y_pred, y_true = _model_foward(model=model, batch=batch, lb=lb, device=device)

                    predictions = torch.argmax(y_pred, dim=1).cpu().numpy()
                    targets = torch.argmax(y_true, dim=1).cpu().numpy()

                    loss = torch.nn.functional.cross_entropy(y_pred, y_true)
                    score = f1_score(targets, predictions, average='weighted', zero_division=0)

                    all_predictions.extend(predictions)
                    all_targets.extend(targets)

                    test_loss += loss.item()
                    test_score += score
                
                test_loss /= len(test_loader)
                test_score /= len(test_loader)

                score = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
                print(f"Test loss = {test_loss:.4f}, Test F1 Score = {test_score:.4f}, Overall F1 Score = {score:.4f}")

                cm = confusion_matrix(all_targets, all_predictions, normalize='true',labels=list(range(len(classes))))

                # Matrice de confusion des données de test
                plt.figure(figsize=(8, 6))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
                disp.plot(cmap=plt.cm.Blues)
                plt.title(f'Confusion Matrix - Test Set\nBackbone: {downstrean_config["pretrained_model"]["model_name"]}\nDownstream type: {downstream}')
                plt.savefig(f'{folder_seed}/confusion_matrix.png')

                results["Labeled Percentage"].append(labeled_percentage)
                results["Init Type"].append(init_type)
                results["Downstream"].append(downstream)
                results["Seeds"].append(seed)
                results["Test Loss"].append(test_loss)
                results["Test F1 Score"].append(score)

# Sauvegarde des résultats dans un fichier CSV
results_df = pd.DataFrame(results)

if os.path.exists(f'{folder_results}/results.csv'):
    existing_results_df = pd.read_csv(f'{folder_results}/results.csv')
    results_df = pd.concat([existing_results_df, results_df], ignore_index=True)

results_df.to_csv(f'{folder_results}/results.csv', index=False)

end = datetime.datetime.now()
interval = (end - start)

print(f"Total execution time: {interval}")

# Visualisation des résultats
labeleds_percentage = list(results_df['Labeled Percentage'].unique())
labeleds_percentage.sort()

plt.figure(figsize=(10, 6))

for init_type in [SCRATCH, PRETRAIN]:
    for downstream in [FROZEN, FINETUNE]:
        frame = results_df.loc[(results_df['Init Type'] == init_type) & (results_df['Downstream'] == downstream)]
        
        x = list()
        y = list()
        e = list()
        
        for labeled_percentage in labeleds_percentage :
            subset = frame[frame['Labeled Percentage'] == labeled_percentage]
            mean_f1 = subset['Test F1 Score'].mean()
            std_f1 = subset['Test F1 Score'].std()

            x.append(labeled_percentage)
            y.append(mean_f1)
            e.append(std_f1)

        if init_type == SCRATCH :
            color = 'blue'
        
        elif init_type == PRETRAIN :
            color = 'red'

        if downstream == FROZEN :
            linestyle = '-'
        
        elif downstream == FINETUNE :
            linestyle = '--'

        plt.plot(x, y, marker='o', color= color, linestyle=linestyle, label=f'{init_type} + {downstream}')
        plt.errorbar(x, y, yerr=e, fmt='o', color=color, linestyle=linestyle, capsize=5)

plt.xlabel('Labeled Percentage')
plt.ylabel('Test F1 Score')
plt.title('Downstream Task Performance')
plt.legend()
plt.grid(True)
plt.savefig(f'{folder_results}/performance.png')