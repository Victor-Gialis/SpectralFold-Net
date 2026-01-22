import importlib
import inspect
import torch
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import train_test_split

DATASET_PATHS = {
    "CWRU": "data/raw/CWRU",
    "LASPI": "data/raw/LASPI",
}

def get_dataset(args:object):
    """
    Returns an instance of the specified dataset class.
    Args:
        name (str): Name of the dataset (e.g., 'CWRU', 'LASPI').
        **kwargs: Additional keyword arguments to pass to the dataset constructor.
    """
    name = args.name
    base_path = DATASET_PATHS[name.upper()]

    module = importlib.import_module(f"dataset.{name.lower()}")
    dataset_class = getattr(module, f"{name.upper()}Dataset")

    # Convert args -> dict
    args_dict = vars(args)
    
    # Inspecte la signature du constructeur
    sig = inspect.signature(dataset_class.__init__)
    valid_args = {
        k: v for k, v in args_dict.items()
        if k in sig.parameters and k != "self"
    }
    
    return dataset_class(root_dir=base_path, **valid_args)

def get_heterogeneous_split_dataloaders(name, batch_size=64, **kwargs):
    # Get dataset
    dataset = get_dataset(name, **kwargs)

    collate_fn = getattr(dataset, '_collate_fn', None)
    if collate_fn is None:
        # fallback: use a default collate_fn if not present
        from torch.utils.data.dataloader import default_collate
        collate_fn = default_collate

    # Split train/valid/test
    train_size = int(0.6 * len(dataset))
    valid_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - valid_size

    # Generate split data
    generator =torch.Generator().manual_seed(42)
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, 
                                                                            [train_size, valid_size, test_size], 
                                                                            generator=generator)
    
    # Split Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    return train_loader, valid_loader, test_loader

def make_strata(dataset):
    strata = []
    for sample in dataset:
        cls = sample['y_label']
        speed = sample['metadata']['speed']
        strata.append(f'{cls}_{speed}')
    strata =np.array(strata)
    return strata

def get_homogenous_split_dataloaders(args:object):
    name = args.name
    batch_size = args.batch_size
    data_ratio = args.data_ratio
    
    # Get dataset
    dataset = get_dataset(args)

    collate_fn = getattr(dataset, '_collate_fn', None)
    if collate_fn is None:
        # fallback: use a default collate_fn if not present
        from torch.utils.data.dataloader import default_collate
        collate_fn = default_collate

    # Create train, valid, test splits with stratification
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
    if data_ratio < 1.0:
        _, scarcity_train_idx = train_test_split(
        train_idx,
        test_size = data_ratio,
        stratify=strata[train_idx],
        shuffle=True,
        )
        
    else :
        scarcity_train_idx = train_idx
    
    # Subdatasets stratified
    train_dataset = Subset(dataset, scarcity_train_idx)
    valid_dataset = Subset(dataset, valid_idx)
    test_dataset = Subset(dataset, test_idx)
    
    # Labels ponderation
    labels = [sample['y_label'] for sample in train_dataset]
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, valid_loader, test_loader, np.unique(labels)

import torch
from tqdm import tqdm

@torch.no_grad()
def compute_min_max_from_dataloader(dataloader, device=None, verbose=True):
    """
    Calcule min et max de X_raw à partir d'un DataLoader.

    Args:
        dataloader: torch.utils.data.DataLoader
        device: torch.device ou None
        verbose: affiche une barre de progression

    Returns:
        dict {"min": float, "max": float}
    """
    min_val = float("inf")
    max_val = float("-inf")

    iterator = dataloader
    if verbose:
        iterator = tqdm(iterator, desc="Computing min/max from DataLoader")

    for batch in iterator:
        # X_raw = batch["X_raw"]
        X_raw = torch.log1p(batch["X_raw"])

        if device is not None:
            X_raw = X_raw.to(device, non_blocking=True)

        batch_min = X_raw.min().item()
        batch_max = X_raw.max().item()

        min_val = min(min_val, batch_min)
        max_val = max(max_val, batch_max)

    return {"min": min_val, "max": max_val}
