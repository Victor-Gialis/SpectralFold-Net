import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
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

# Pourcentage de données étiquetées dans l'ensemble d'entraînement
labeled_percentage = 1.0

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

if labeled_percentage < 1.0:
    _, train_idx = train_test_split(
    train_idx,
    test_size = labeled_percentage,
    stratify=strata[train_idx],
    random_state=42
)

train_dataset = Subset(dataset, train_idx)
valid_dataset = Subset(dataset, valid_idx)
test_dataset = Subset(dataset, test_idx)

datasets = {
    "train": train_dataset,
    "valid": valid_dataset,
    "test": test_dataset
}

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

# for split,loader in dataloaders.items():
#     result = {"batch":[],
#                 "normal":[],
#                 "inner":[],
#                 "outer":[],
#                 "ball":[]
#                 }

#     for i,batch in tqdm(enumerate(loader)): 
#         if i>20:
#             break
#         df = pd.DataFrame(batch['metadata'])
#         df['label'] = batch['label']
#         count = df.groupby(['label'])['source'].count()

#         result['batch'].append(i)
#         for cls in classes :
#             result[cls].append(count.get(cls,0))

#     result = pd.DataFrame(result)
#     print(result)

#     result.plot(x='batch', kind='bar')
#     plt.ylabel('Number of samples')
#     plt.savefig(f'{split}_data_distribution.png')

for split, data in datasets.items():
    print(f"{split} dataset: {len(data)} samples")
    # Weighted random sampler to balance classes in the training set
    labels = _make_strata(data)
    classes, class_counts = np.unique(labels, return_counts=True)