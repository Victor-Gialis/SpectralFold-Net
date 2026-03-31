"""
Exemple simple d'utilisation du modèle downstream pour l'inférence.

Cet exemple montre comment charger et utiliser le modèle dans un notebook ou script.
"""

import torch
import json
import os
import numpy as np
from types import SimpleNamespace
from pathlib import Path

# Ajouter le répertoire courant au path Python
import sys
sys.path.insert(0, '/home/ngrotus/Desktop/CWRU_MAE')

from models.backbone.vit1d import ViT1DEncoder
from models.ssl.mae import MAEModel
from models.downstream.registry import get_downstream_model
from training.pretrain import load_model_checkpoint
from dataset.transform import normalization


class DownstreamModelInference:
    """Wrapper pour faire facilement l'inférence avec le modèle downstream"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Args:
            model_path: Chemin vers best_model.pth ou last_model.pth
            device: "cpu" ou "cuda"
        """
        self.device = torch.device(device)
        self.model_path = model_path
        self.run_dir = os.path.dirname(model_path)
        
        # Charger la configuration
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        print(f"✓ Configuration chargée")
        print(f"  Task: {self.config['task']}")
        print(f"  Head: {self.config['head_type']}")
        
        # Reconstruire le modèle
        self._build_model()
        
    def _build_model(self):
        """Reconstruire le modèle complet"""
        
        # Toujours créer un backbone random comme base
        backbone_random = ViT1DEncoder()
        
        # Déterminer le type de backbone
        backbone_type = self.config.get('backbone_init', 'mae').lower()
        
        # Charger le checkpoint pré-entraîné selon le type
        if backbone_type == 'random' or self.config.get('pretrain_dataset') == 'None':
            # Pas de pré-entraînement
            backbone = backbone_random
            print(f"✓ Backbone random (sans pré-entraînement)")
        
        elif backbone_type == 'mae':
            # Charger le checkpoint MAE
            ssl_mode = MAEModel(
                backbone=backbone_random,
                args_ssl=SimpleNamespace(mask_ratio=1.0)
            )
            
            pretrain_dataset = self.config.get('pretrain_dataset', 'CWRU')
            checkpoints_dir = f"results/pretrain/MAEModel/{pretrain_dataset}Dataset"
            
            if os.path.exists(checkpoints_dir):
                checkpoints = sorted(os.listdir(checkpoints_dir))
                pretrain_checkpoint_path = os.path.join(checkpoints_dir, checkpoints[-1])
                ssl_model = load_model_checkpoint(ssl_mode, pretrain_checkpoint_path)
                backbone = ssl_model.backbone
                print(f"✓ Backbone MAE chargé")
            else:
                print(f"⚠️  Checkpoint MAE non trouvé, utilisation d'un backbone random")
                backbone = backbone_random
        
        elif backbone_type == 'sap':
            # Charger le checkpoint SAP
            try:
                from models.ssl.sap import SAPModel
                ssl_mode = SAPModel(backbone=backbone_random)
            except ImportError:
                print("⚠️  SAPModel non trouvé, utilisation d'un backbone random")
                backbone = backbone_random
            else:
                pretrain_dataset = self.config.get('pretrain_dataset', 'CWRU')
                checkpoints_dir = f"results/pretrain/SAPModel/{pretrain_dataset}Dataset"
                
                if os.path.exists(checkpoints_dir):
                    checkpoints = sorted(os.listdir(checkpoints_dir))
                    pretrain_checkpoint_path = os.path.join(checkpoints_dir, checkpoints[-1])
                    ssl_model = load_model_checkpoint(ssl_mode, pretrain_checkpoint_path)
                    backbone = ssl_model.backbone
                    print(f"✓ Backbone SAP chargé")
                else:
                    print(f"⚠️  Checkpoint SAP non trouvé, utilisation d'un backbone random")
                    backbone = backbone_random
        
        else:
            print(f"⚠️  Type de backbone inconnu: {backbone_type}, utilisation d'un backbone random")
            backbone = backbone_random
        
        # Modèle downstream
        self.model = get_downstream_model(
            backbone=backbone,
            task=self.config['task'],
            head_type=self.config['head_type'],
            classes=np.array(self.config.get('classes', [])),
            freeze_backbone=not self.config.get('finetune', True),
            device=self.device
        )
        
        # Charger les poids
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Modèle chargé depuis {self.model_path}")
        
    def predict(self, x: np.ndarray) -> dict:
        """
        Faire une prédiction sur un ou plusieurs échantillons.
        
        Args:
            x: Array de shape (n_samples, window_size) ou (window_size,)
        
        Returns:
            dict avec les résultats
        """
        # Assurer que c'est un tensor 2D
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = x.to(self.device)
        
        with torch.no_grad():
            batch = {'X_raw': x}
            outputs = self.model(batch)
            
            if self.config['task'] == 'classification':
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                results = {
                    'predictions': preds.cpu().numpy(),
                    'probabilities': probs.cpu().numpy(),
                    'confidence': probs.max(dim=1).values.cpu().numpy(),
                    'class_names': [
                        self.model.head.lb.classes_[p] for p in preds.cpu().numpy()
                    ]
                }
            else:
                results = {
                    'predictions': outputs.cpu().numpy()
                }
        
        return results
    
    def get_features(self, x: np.ndarray) -> np.ndarray:
        """
        Extraire les features (CLS token) du backbone.
        
        Args:
            x: Array de shape (n_samples, window_size) ou (window_size,)
        
        Returns:
            Features de shape (n_samples, hidden_dim)
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = x.to(self.device)
        
        with torch.no_grad():
            x_norm = normalization.global_z_log_normalization(
                x=x,
                stats=self.model.backbone.stats
            )
            features = self.model.backbone(x_norm)
            cls_token = features[:, 0]  # CLS token
        
        return cls_token.cpu().numpy()


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    # 1. Initialiser le modèle
    print("="*70)
    print("INITIALISATION DU MODÈLE")
    print("="*70)
    
    # Trouver le dernier modèle entraîné
    downstream_dir = "results/downstream/CWRU_to_LASPI_backbone_mae_head_linear_finetune_False/data_ratio_0.2_epochs_50"
    model_path = os.path.join(downstream_dir, "best_model.pth")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Vérifier que le chemin est correct!")
        exit(1)
    
    model = DownstreamModelInference(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Créer des données de test (dummy)
    print("\n" + "="*70)
    print("CRÉATION DE DONNÉES DE TEST")
    print("="*70)
    
    # 2 échantillons de window_size 2048
    x_test = np.random.randn(2, 2048).astype(np.float32)
    print(f"✓ Données de test créées: shape {x_test.shape}")
    
    # 3. Faire les prédictions
    print("\n" + "="*70)
    print("PRÉDICTIONS")
    print("="*70)
    
    results = model.predict(x_test)
    
    for i, pred_class in enumerate(results['class_names']):
        confidence = results['confidence'][i]
        print(f"Échantillon {i}: {pred_class} ({confidence:.2%})")
    
    # 4. Extraire les features
    print("\n" + "="*70)
    print("EXTRACTION DES FEATURES")
    print("="*70)
    
    features = model.get_features(x_test)
    print(f"✓ Features extraites: shape {features.shape}")
    print(f"  Norme L2: {np.linalg.norm(features, axis=1)}")
    
    print("\n✓ Exemple terminé!")
