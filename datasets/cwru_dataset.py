import os
import torch
import numpy as np
from datasets.base_dataset import BaseDataset, Sample

class CWRUDataset(BaseDataset):
    def __init__(self, source='end', fault_filter=None, transform_type=None, window_size=None, stride=None):
        """
        Args:
            source (str): Source des données, peut être 'DE', 'FE' ou 'BA'.
            fault_filter (list, optional): Liste de défauts à filtrer. Si None, tous les défauts sont inclus.
            transform_type (str, optional): Type de transformation à appliquer aux données.
            window_size (int, optional): Taille de la fenêtre pour les échantillons.
            stride (int, optional): Pas de la fenêtre pour les échantillons.
        """
        assert source in ['DE', 'FE', 'BA'], "source must be 'DE', 'FE' or 'BA'"
        assert fault_filter is None or isinstance(fault_filter, list), "fault_filter doit être une liste ou None"
        self.source = source
        super().__init__(root_dir=os.path.dirname(os.path.abspath(__file__)), 
                         fault_filter=fault_filter, 
                         transform_type=transform_type, 
                         window_size=window_size, 
                         stride=stride)
    
    def __delattr__(self, name):
        """
        Permet de supprimer un attribut de l'instance.
        """
        return super().__delattr__(name)

    def _read_sample(self, filepath)-> torch.Tensor:
        """
        Lit un sample à partir du fichier npz spécifié.
        Args:
            filepath (str): Chemin vers le fichier npz.
        Returns:
            Sample: Un objet Sample contenant les données lues.
        """
        data = np.load(filepath)
        data = torch.tensor(data[self.source]).float().squeeze()
        return data

    def _collect_samples(self):
        for speed in os.listdir(self.root_dir):
            full_speed_path = os.path.join(self.root_dir, speed)
            if os.path.isdir(full_speed_path):
                for f in os.listdir(full_speed_path):
                    if f.endswith(".npz"):
                        npz_path = os.path.join(full_speed_path, f)
                        filename = f.split('.')[0]
                        pattern = filename.split('_')
                        # On suppose que le nom de fichier est toujours au format speed_fault_diameter_end
                        # Par exemple : 1700_1_1_DE.npz ou 1700_1_1_1_DE.npz
                        diameter = None
                        position = None

                        if len(pattern) == 4:
                            speed, default, diameter, source = pattern
                            if '@' in default:
                                default, position = default.split('@')
                        else:
                            speed, default = pattern
                            source = self.source  # Pour les fichiers en acquisition normale, on suppose que c'est la fin par défaut
                        
                        label = self._label_from_default(default)
                        if self.fault_filter is None or label in self.fault_filter and self.source in source:
                            # On ne garde que les fichiers qui correspondent au filtre de défaut et à la source
                            self.samples.append(Sample(filepath=npz_path,
                                                    label=label,
                                                    metadata={'speed': speed, 'position': position, 'diameter': diameter, 'source': source}))

    def _extract_label_from_filename(self, default):

        mapping = {'Normal':'normal',
                    'IR':'inner', 
                    'OR':'outer', 
                    'B':'ball'}

        return mapping.get(default, -1)

