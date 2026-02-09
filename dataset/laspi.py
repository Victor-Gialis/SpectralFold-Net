import os
import torch
import numpy as np
from dataset.base import BaseDataset, Sample

class LASPIDataset(BaseDataset):
    def __init__(self, root_dir=None, fault_filter=None, speed_filter=None, window_size=2048, window_stride=256, downsampling_factor=None):
        """
        Args:
            transform (callable, optional): Transformation à appliquer aux données.
        """
        super().__init__(root_dir=root_dir, 
                         fault_filter=fault_filter, 
                         speed_filter=speed_filter,
                         window_size=window_size, 
                         window_stride=window_stride,
                         downsampling_factor=downsampling_factor,
                         )
    
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
        """
        Docstring for _collect_samples
        
        :param self: Description
        """ 
        # Parcours des répertoires pour collecter les échantillons
        for default in os.listdir(self.root_dir):
            default_path = os.path.join(self.root_dir, default)
            if not os.path.isdir(default_path) or default == '__pycache__':
                continue

            # Parcours des conditions de fonctionnement
            for conditions in os.listdir(default_path):
                speed, load, speed_rpm = conditions.split('_')
                
                # Nettoyage des valeurs
                speed = int(speed.replace('hz', ''))
                load = int(load.replace('%', ''))
                speed_rpm = int(speed_rpm.replace('rpm', ''))

                # Chemin vers le répertoire des conditions
                cond_path = os.path.join(default_path, conditions)
                if not os.path.isdir(cond_path):
                    continue

                # Parcours des fichiers CSV dans le répertoire des conditions
                for file in os.listdir(cond_path):
                    if file.endswith('.csv'):
                        csv_path = os.path.join(cond_path, file)
                        label = self._extract_label_from_filename(default) # Convertit le nom du défaut en étiquette
                        
                        # Filtrage des échantillons selon les filtres spécifiés
                        if self.fault_filter is  None or label in self.fault_filter:
                            if self.speed_filter is None or speed in self.speed_filter:

                                # Ajout de l'échantillon à la liste des échantillons
                                self.samples.append(Sample(filepath=csv_path,
                                                            label=label,
                                                            metadata={'speed': speed, 
                                                                      'load': load, 
                                                                      'speed_rpm': speed_rpm}))
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