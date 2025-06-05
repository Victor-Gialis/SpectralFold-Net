import os
import torch
import numpy as np
from datasets.base_dataset import BaseDataset, Sample

class LASPIDataset(BaseDataset):
    def __init__(self, fault_filter=None, transform_type=None, window_size=None, stride=None):
        """
        Args:
            transform (callable, optional): Transformation à appliquer aux données.
        """
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
        Lit un sample à partir du fichier CSV spécifié.
        Args:
            filepath (str): Chemin vers le fichier CSV.
        Returns:
            Sample: Un objet Sample contenant les données lues.
        """
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        data = np.transpose(data)  # Transpose pour avoir les colonnes comme caractéristiques
        data = data[3] # On récupère les données de la quatrième colonne (accéléromètre)
        data = torch.tensor(data, dtype=torch.float32)
        return data

    def _collect_samples(self):
        for default in os.listdir(self.root_dir):
            default_path = os.path.join(self.root_dir, default)
            if not os.path.isdir(default_path) or default == '__pycache__':
                continue
            for conditions in os.listdir(default_path):
                freq, load, speed = conditions.split('_')
                freq = int(freq.replace('hz', ''))
                load = int(load.replace('%', ''))
                speed = int(speed.replace('rpm', ''))

                cond_path = os.path.join(default_path, conditions)
                if not os.path.isdir(cond_path):
                    continue
                for file in os.listdir(cond_path):
                    if file.endswith('.csv'):
                        csv_path = os.path.join(cond_path, file)
                        label = self._label_from_default(default) # Convertit le nom du défaut en étiquette
                        if self.fault_filter is  None or label in self.fault_filter:
                            self.samples.append(Sample(filepath=csv_path,
                                                        label=label,
                                                        metadata={'freq': freq, 'load': load, 'speed': speed}))

    def _extract_label_from_filename(self, default):

        mapping = {'Bearing_inner_race_fault': 'inner', 
                   'Bearing_outer_race_fault': 'outer', 
                   'Gear_half_broken_tooth': 'gear_half', 
                   'Gear_half_broken_tooth_and_bearing_outer_race_faults': 'gear_half_&_outer',
                   'Gear_surface_and_bearing_inner_race_faults': 'gear_surface_&_inner',
                   'Gear_surface_damage': 'gear_surface',
                   'Healthy_motor': 'normal',
                   }

        return mapping.get(default, -1)