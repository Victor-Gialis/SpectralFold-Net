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
        cls = sample['label']
        speed = sample['metadata']['speed']
        strata.append(f'{cls}_{speed}')
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

input_size = dataset[0]['X_true'].shape[-1]
print(f"Dataset '{dataset_name}' loaded with {len(dataset)} samples. FFT size: {input_size}")

train_size = int(0.7 * len(dataset))
valid_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - valid_size

indice = np.arange(len(dataset))
strata = _make_strata(dataset)

train_val_idx, test_idx = train_test_split(
    indice,
    test_size = 0.2,
    stratify=strata,
    random_state=42
)

train_idx, valid_idx = train_test_split(
    train_val_idx,
    test_size = 0.13,
    stratify=strata[train_val_idx],
    random_state=42
)

generator =torch.Generator().manual_seed(42)
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size], generator=generator)

train_dataset = Subset(dataset, train_idx)
valid_dataset = Subset(dataset, valid_idx)
test_dataset = Subset(dataset, test_idx)

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

collate_fn = getattr(dataset, '_collate_fn', None)
if collate_fn is None:
    # fallback: use a default collate_fn if not present
    from torch.utils.data.dataloader import default_collate
    collate_fn = default_collate

# dataloader = DataLoader(
#     train_dataset, 
#     batch_size=batch_size, 
#     collate_fn=collate_fn
#     )

dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=sampler,
    collate_fn=collate_fn
    )

result = {"batch":[],
            "normal":[],
            "inner":[],
            "outer":[],
            "ball":[]
            }

for i,batch in tqdm(enumerate(dataloader)): 
    if i>20:
        break
    df = pd.DataFrame(batch['metadata'])
    df['label'] = batch['label']
    count = df.groupby(['label'])['source'].count()

    result['batch'].append(i)
    for cls in classes :
        result[cls].append(count.get(cls,0))

result = pd.DataFrame(result)
print(result)

result.plot(x='batch', kind='bar')
plt.ylabel('Number of samples')
plt.savefig('data_distribution.png')
