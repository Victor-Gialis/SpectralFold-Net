import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE

from tqdm import tqdm
from models.model import PretrainedModel, DownstreamClassifier, Encoder
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from datasets.dataloader import get_dataset
from sklearn.model_selection import train_test_split

def _make_strata(dataset):
    strata = []
    for sample in tqdm(dataset):
        informations = list()
        informations.append(sample['label'])
        for meta, data in sample['metadata'].items():
            if data != None:
                metadata = f'{meta}_{data}'
            else :
                metadata = f'{meta}_None'
            informations.append(metadata)
        informations = '_'.join(informations)
        strata.append(informations)
    strata =np.array(strata)
    return strata

batch_size = 256
dataset_config = {
    "name":"CWRU",
    "fault_filter": None,
    "speed_filter": None,
    "transform_type":"psd",
    "window_size": 2048,
    "stride": 256,
    "flip" : True,
    "downsample_factor": 2
}

dataset_name = dataset_config['name']
dataset_params = {k: v for k, v in dataset_config.items()}
dataset = get_dataset(**dataset_params)

# Create DataLoaders
batch_size = 256
collate_fn = getattr(dataset, '_collate_fn', None)
if collate_fn is None:
    # fallback: use a default collate_fn if not present
    from torch.utils.data.dataloader import default_collate
    collate_fn = default_collate

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# # Pourcentage de données étiquetées dans l'ensemble d'entraînement
# labeled_percentage = 1.0

# Taille des spectres FFT
input_size = dataset[0]['X_true'].shape[-1]
print(f"Dataset '{dataset_name}' loaded with {len(dataset)} samples. FFT size: {input_size}")

# # Create train, valid, test splits with stratification
# indice = np.arange(len(dataset))
# strata = _make_strata(dataset)

# train_idx, test_val_idx = train_test_split(
#     indice,
#     test_size = 0.4,
#     stratify=strata,
#     random_state=42
# )

# valid_idx, test_idx = train_test_split(
#     test_val_idx,
#     test_size = 0.5,
#     stratify=strata[test_val_idx],
#     random_state=42
# )

# if labeled_percentage < 1.0:
#     _, train_idx = train_test_split(
#     train_idx,
#     test_size = labeled_percentage,
#     stratify=strata[train_idx],
#     random_state=42
# )

# train_dataset = Subset(dataset, train_idx)
# valid_dataset = Subset(dataset, valid_idx)
# test_dataset = Subset(dataset, test_idx)

# datasets = {
#     "train": train_dataset,
#     "valid": valid_dataset,
#     "test": test_dataset
# }

# samplers = dict()
# dataloaders = dict()

# for split, data in datasets.items():
#     print(f"{split} dataset: {len(data)} samples")
#     # Weighted random sampler to balance classes in the training set
#     labels = [sample['label'] for sample in tqdm(data)]
#     classes, class_counts = np.unique(labels, return_counts=True)

#     class_weights = 1. / class_counts
#     claas_weights = class_weights / class_weights.sum()
#     class_weights = {cls: weight for cls, weight in zip(classes, class_weights)}

#     sample_weights = [class_weights[label] for label in labels]
#     sample_weights = torch.DoubleTensor(sample_weights)

#     sampler = WeightedRandomSampler(
#         weights=sample_weights,
#         num_samples=len(sample_weights),
#         replacement=True
#     )

#     samplers[split] = sampler
#     dataloaders[split] = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)

dataloaders = {"all": dataloader}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instancier le modèle pré-entraîné et charger les poids
pretrain_model_name = 'model_v3.2'

# Charger la config du pré-entraînement pour les hyperparamètres
with open(f'results/{pretrain_model_name}/used_config.json', 'r') as f:
    pretrain_config = json.load(f)
pretrain_params = pretrain_config['model']

# Visualize embedding space
backbone = Encoder(
    num_patch= input_size // pretrain_params.get('patch_size', 64),
    patch_size= pretrain_params.get('patch_size', 64),
    encoder_dim= pretrain_params.get('encoder_dim', 128),
    n_layers= pretrain_params.get('n_layers', 4),
    heads= pretrain_params.get('heads', 8),
    dropout= pretrain_params.get('dropout', 0.4),
).to(device)
backbone.load_state_dict(torch.load(f'checkpoint/{pretrain_model_name}/model.pth'))

backbone.eval()

embeded_tokens = list()
qualitative_data = list()

for split,loader in dataloaders.items():
    print(f"{split} dataset")

    for batch in tqdm(loader):
        X_true = batch['X_true'].unsqueeze(1).to(device, non_blocking=True)
        with torch.no_grad():
            z = backbone(X_true)
        z = z[:, 0] #Take only the CLS token
        z = z.cpu().numpy()

        metadata = pd.DataFrame(batch['metadata'])
        metadata['label'] = batch['label']

        embeded_tokens.append(z)
        qualitative_data.append(metadata)

embeded_tokens = np.concatenate(embeded_tokens)
qualitative_data = pd.concat(qualitative_data)

tsne = TSNE(n_components=2, random_state=42)
z_2d = tsne.fit_transform(embeded_tokens)

classes = qualitative_data['label'] +'_'+ qualitative_data['diameter']

df = pd.DataFrame({
    'tsne1':z_2d[:,0],
    'tsne2':z_2d[:,1],
    'speed':qualitative_data['speed'],
    'position':qualitative_data['position'],
    'diameter':qualitative_data['diameter'],
    'source':qualitative_data['source'],
    'label':qualitative_data['label'],
    'classes':classes
})

df.to_parquet(f'CWRU_embedding_space.parquet',engine='pyarrow')

fig = px.scatter(x=df['tsne1'],
                 y=df['tsne2'],
                 color=df['classes'])
fig.write_html('CWRU_embedding_space.html')
print('oui')