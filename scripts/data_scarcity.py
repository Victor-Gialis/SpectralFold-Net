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

start = datetime.datetime.now()

folder_experiment= os.path.join('results','downstream',os.path.basename(__file__))
os.makedirs(folder_experiment, exist_ok=True)

SCRATCH = 'scratch'
PRETRAIN = 'pretrain'
FROZEN = 'frozen'
FINETUNE = 'finetune'

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
    diameters = batch['metadata']['diameter']
    classes = labels + '-' + diameters
    # One-hot encode labels
    y_true = lb.transform(classes)
    y_true = torch.tensor(y_true, dtype=torch.float32).to(device, non_blocking=True)
    # Get batch size
    b,_,_ = X_true.shape
    # Normalisation des signaux
    X_true = _z_norm(X_true, mean.expand(b, 1, -1), std.expand(b, 1, -1))
    # Prédiction du modèle
    y_pred = model(X_true)
    return y_pred, y_true

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger la config
with open('configs/downstream_config.json', 'r') as f:
    downstrean_config = json.load(f)

training_params = downstrean_config['training']
labeled_percentage = training_params.get('labeled_percentage', 1.0)

# Instancier le dataset via la factory
dataset_name = downstrean_config['dataset']['name']
dataset_params = {k: v for k, v in downstrean_config['dataset'].items() if k != 'name'}
dataset = get_dataset(dataset_name, **dataset_params)

# Create DataLoaders
batch_size = downstrean_config.get('dataloader', {}).get('batch_size', 16)
collate_fn = getattr(dataset, '_collate_fn', None)
if collate_fn is None:
    # fallback: use a default collate_fn if not present
    from torch.utils.data.dataloader import default_collate
    collate_fn = default_collate

# Instancier le modèle pré-entraîné et charger les poids
pretrain_model_name = downstrean_config['pretrained_model'].get('model_name', 'default_pretrain_model')

# Charger la config du pré-entraînement pour les hyperparamètres
with open(f'results/{pretrain_model_name}/used_config.json', 'r') as f:
    pretrain_config = json.load(f)
pretrain_params = pretrain_config['model']

# Logging downstream parameters
training_params = downstrean_config['training']
epochs = training_params.get('epochs', 100)
learning_rate = training_params.get('learning_rate', 1e-3) * batch_size / 256

# Taille des spectres FFT
input_size = dataset[0]['X_true'].shape[-1]
print(f"Dataset '{dataset_name}' loaded with {len(dataset)} samples. FFT size: {input_size}")

# Create train, valid, test splits with stratification
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

folder_results = os.path.join(folder_experiment,f'{dataset_name}_dataset_{pretrain_model_name}_backbone')
os.makedirs(folder_results)

for labeled_percentage in [1.0, 0.5, 0.25, 0.2, 0.15, 0.1, 0.05]:
    print(f"Labeled percentage: {labeled_percentage*100}%")

    if labeled_percentage < 1.0:
        _, scarcity_train_idx = train_test_split(
        train_idx,
        test_size = labeled_percentage,
        stratify=strata[train_idx],
        random_state=42
    )
        
    else :
        scarcity_train_idx = train_idx

    train_dataset = Subset(dataset, scarcity_train_idx)
    valid_dataset = Subset(dataset, valid_idx)
    test_dataset = Subset(dataset, test_idx)

    # Weighted random sampler to balance classes in the training set
    labels = [sample['label'] + '_' + sample['metadata']['diameter'] for sample in tqdm(train_dataset)]
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

    # Extract globale statistics from train dataset
    mean, std = global_stats(train_dataset)
    mean = mean.to(device)
    std = std.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # All classes on the dataset, One hot encoding
    lb = LabelBinarizer()
    lb.fit(classes)

    for init_type in [SCRATCH, PRETRAIN]:
        for downstream in [FROZEN, FINETUNE]:

            folder_model = os.path.join(folder_results,f'{labeled_percentage}_scarcity_{init_type}_init_{downstream}_downstream')
            os.makedirs(folder_model, exist_ok=True)
            
            # Répétition de plusieurs entraînement avec même config
            for seed in range(5) :

                folder_seed =os.path.join(folder_model,f'seed_{seed+1}')
                os.makedirs(folder_seed, exist_ok=True)

                # Logging the backbone
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
                    model = DownstreamClassifier(
                        backbone= backbone,
                        num_classes= len(classes),
                        freeze_backbone= True,
                    ).to(device)

                elif init_type == SCRATCH and downstream == FINETUNE :
                    model = DownstreamClassifier(
                        backbone= backbone,
                        num_classes= len(classes),
                        freeze_backbone= False,
                    ).to(device)

                elif init_type == PRETRAIN and downstream == FROZEN :
                    backbone.load_state_dict(torch.load(f'checkpoint/{pretrain_model_name}/model.pth'))
                    model = DownstreamClassifier(
                        backbone= backbone,
                        num_classes= len(classes),
                        freeze_backbone= True,
                    ).to(device)

                elif init_type == PRETRAIN and downstream == FINETUNE :
                    backbone.load_state_dict(torch.load(f'checkpoint/{pretrain_model_name}/model.pth'))
                    model = DownstreamClassifier(
                        backbone= backbone,
                        num_classes= len(classes),
                        freeze_backbone= False,
                    ).to(device)
                    
                optimizer = torch.optim.AdamW(model.parameters(), lr= learning_rate, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_params.get('epochs', 100), eta_min=1e-6)

                # Save all metrics
                all_train_loss = []
                all_valid_loss = []

                all_train_score = []
                all_valid_score = []

                # === Training loop ===
                for epoch in range(1,epochs+1):
                    model.train()
                    train_loss = 0
                    train_score = 0

                    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
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

                    print(f"Epoch {epoch+1}: train loss = {train_loss:.4f}")

                    model.eval()
                    valid_loss = 0
                    valid_score = 0

                    for batch in tqdm(valid_loader, desc=f"Epoch {epoch}/{epochs}"):
                        
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

                    print(f"Epoch {epoch+1}: valid loss = {valid_loss:.4f}")

                    scheduler.step()

                # === End training loop ===
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

                # Testing loop
                model.eval()
                test_loss = 0
                test_score = 0

                all_predictions = []
                all_targets = []

                for batch in tqdm(test_loader, desc="Testing"):
                    
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

                cm = confusion_matrix(all_targets, all_predictions, normalize='true',labels=[0,1,2,3])
                print("Confusion Matrix:\n", cm)

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

results_df = pd.DataFrame(results)
results_df.to_csv(f'{folder_model}/results.csv', index=False)

# End of training
end = datetime.datetime.now()
interval = (end - start)

print(f"Total execution time: {interval}")


plt.figure(figsize=(10, 6))

for init_type in [SCRATCH, PRETRAIN]:
    for downstream in [FROZEN, FINETUNE]:
        frame = results_df.loc[(results_df['Init Type'] == init_type) & (results_df['Downstream'] == downstream)]
        x = frame['Labeled Percentage'].values
        y = frame['Test F1 Score'].values

        if init_type == SCRATCH :
            color = 'blue'
        
        elif init_type == PRETRAIN :
            color = 'red'

        if downstream == FROZEN :
            linestyle = '-'
        
        elif downstream == FINETUNE :
            linestyle = '--'

        plt.plot(x, y, marker='o', color= color, linestyle=linestyle, label=f'{init_type} + {downstream}')

plt.xlabel('Labeled Percentage')
plt.ylabel('Test F1 Score')
plt.title('Downstream Task Performance')
plt.legend()
plt.grid(True)
plt.savefig(f'{folder_model}/performance.png')