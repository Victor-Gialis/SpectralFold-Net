import os
import torch
import numpy as np
from dataclasses import dataclass
from scipy.signal import hilbert
from torch.utils.data import Dataset
from torch import stack, Tensor

@dataclass
class Sample:
    filepath: str
    label: str
    metadata: dict

@dataclass
class SampleWindow:
    filepath: str
    label: str = None
    start_idx: int = 0
    metadata: dict = None

class BaseDataset(Dataset):
    def __init__(self, root_dir, fault_filter=None, transform_type=None, window_size=None, stride=None):
        assert os.path.isdir(root_dir), f"Le répertoire {root_dir} n'existe pas ou n'est pas un répertoire."
        assert window_size is None or isinstance(window_size, int) and window_size > 0, "window_size doit être un entier positif."
        assert stride is None or isinstance(stride, int) and stride > 0, "stride doit être un entier positif."
        assert transform_type in [None, 'normalize', 'standardize','psd', 'psd_envelope'], "transform_type doit être None, 'normalize', 'standardize' ou 'envelope'."
        """
        Classe de base pour les datasets.
        Args:
            root_dir (str): Chemin vers le répertoire racine du dataset.
            fault_filter (list, optional): Liste des défauts à filtrer. Si None, tous les défauts sont inclus.
            transform_type (str, optional): Type de transformation à appliquer aux données. Options: 'normalize', 'standardize', 'envelope'.
            window_size (int, optional): Taille de la fenêtre pour les échantillons. Si None, chaque échantillon est une fenêtre unique.
            stride (int, optional): Pas de déplacement pour la collecte des fenêtres. Si None, pas de déplacement.
        """
        self.root_dir = root_dir
        self.fault_filter = fault_filter
        self.transform_type = transform_type
        self.window_size = window_size
        self.stride = stride

        self.samples = []
        self.windows = []

        self._collect_samples()
        self._collect_windows()
    
    def _read_sample(self, filepath):
        """
        Lit un sample à partir du fichier spécifié.
        Doit être implémenté par les sous-classes.
        """
        raise NotImplementedError("Chaque dataset doit implémenter _read_sample()")

    def _collect_samples(self):
        raise NotImplementedError("Chaque dataset doit implémenter _collect_samples()")

    def _extract_label_from_filename(self, filename):
        """
        Convertit le nom du défaut en étiquette.
        Doit être implémenté par les sous-classes.
        """
        raise NotImplementedError("Chaque dataset doit implémenter _label_from_filename()")

    def _collect_windows(self):
        """
        Collecte les fenêtres de samples en fonction de window_size et stride.
        Si window_size et stride ne sont pas définis, chaque sample est considéré comme une fenêtre unique.
        """
        for sample in self.samples:
            data = self._read_sample(sample.filepath)
            signal_length = data.shape[-1] # Longueur du signal, supposé être le dernier axe

            if self.window_size is None or self.stride is None:
                self.windows.append(SampleWindow(filepath=sample.filepath, label=sample.label, start_idx=0, metadata=sample.metadata))
            else:
                for start in range(0, signal_length - self.window_size + 1, self.stride):
                    self.windows.append(SampleWindow(filepath=sample.filepath, label=sample.label, start_idx=start, metadata=sample.metadata))

    def _collate_fn(self, batch):
        batch_out = {}
        keys = batch[0].keys()
        for key in keys:
            values = [item[key] for item in batch]
            # Si tous les éléments sont des tensors de même taille, on stack
            if all(isinstance(v, Tensor) for v in values):
                try:
                    batch_out[key] = stack(values)
                except Exception:
                    batch_out[key] = values  # fallback: liste si stack impossible
            else:
                batch_out[key] = values
        return batch_out

    def _transform(self, data):
        """
        Applique la transformation spécifiée aux données.
        Si transform_type est None, retourne les données sans transformation.
        """
        N = data.shape[-1]  # Longueur du signal, supposé être le dernier axe
        if self.transform_type is None:
            return data
        elif self.transform_type == 'normalize':
            return (data - torch.mean(data)) / torch.std(data)
        elif self.transform_type == 'standardize':
            return (data - torch.min(data)) / (torch.max(data) - torch.min(data))
        elif self.transform_type == 'psd':
            return torch.abs(torch.fft.rfft(data - torch.mean(data)))**2/N
        elif self.transform_type == 'psd_envelope':
            return torch.abs(torch.fft.rfft(torch.abs(torch.from_numpy(hilbert(data - torch.mean(data))))))**2/N
        else:
            raise ValueError(f"Transformation '{self.transform_type}' non supportée.")

    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        """
        Retourne un échantillon de données à l'index spécifié.
        Args:
            idx (int): Index de l'échantillon à retourner.
        Returns:
            dict: Un dictionnaire contenant les données de l'échantillon.
        """
        filepath = self.windows[idx].filepath
        label = self.windows[idx].label
        metadata = self.windows[idx].metadata

        X_true = self._read_sample(filepath) # Lecture du sample
        if X_true is None:
            raise ValueError(f"Le sample à l'index {idx} n'a pas pu être lu ou est vide.")

        # Si window_size est défini, on extrait la fenêtre correspondante
        if self.window_size is not None:
            start_idx = self.windows[idx].start_idx
            X_true= X_true[..., start_idx:start_idx + self.window_size]

        else:
            X_tilde = X_true

        # Alétation des données
        X_tilde = self._data_tilde_transform(X_true) # Altération des données
        
        # Transformation des données
        X_true = self._transform(X_true)
        X_tilde = self._transform(X_tilde)

        return {'X_tilde':X_tilde, 'X_true':X_true, 'label':label, 'metadata':metadata}

    @staticmethod
    def _data_tilde_transform(X_true, downsample_factor=2):
        """
        Transforme les données en les réduisant par un facteur de downsample.
        """
        X_tilde = X_true[...,::downsample_factor]
        return X_tilde