"""
Script d'inférence pour utiliser le modèle downstream entraîné.

Cet script montre comment:
1. Charger le modèle downstream entraîné
2. Charger les données de test
3. Faire des prédictions
4. Afficher les résultats
"""
import tqdm
import torch
import json
import os
import argparse
import numpy as np
from types import SimpleNamespace
from pathlib import Path

from dataset import split_data_factory
from models.backbone.vit1d import ViT1DEncoder
from models.ssl.mae import MAEModel
from models.downstream.registry import get_downstream_model
from training.pretrain import load_model_checkpoint
from dataset.transform import normalization


def load_downstream_model(
    model_path: str,
    pretrain_checkpoint_path: str = None,
    classes: np.ndarray = None,
    device: torch.device = "cpu"
):
    """
    Charger le modèle downstream entraîné avec son backbone pré-entraîné.
    
    Supporte les trois types de backbone:
    - MAE: Masked AutoEncoder pré-entraîné
    - SAP: Autres modèles SSL pré-entraînés
    - Random/Scratch: Backbone sans pré-entraînement
    
    Args:
        model_path: Chemin vers le fichier du modèle downstream (best_model.pth ou last_model.pth)
        pretrain_checkpoint_path: Chemin vers le checkpoint de pré-entraînement (optionnel)
        device: Device (cuda ou cpu)
    
    Returns:
        model: Modèle downstream chargé
        config: Configuration du modèle (depuis config.json)
    """
    
    # Charger la configuration depuis le répertoire du modèle
    run_dir = os.path.dirname(model_path)
    config_path = os.path.join(run_dir, "config.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"✓ Configuration chargée depuis: {config_path}")
    print(f"  - Backbone: {config.get('backbone_init', 'unknown')}")
    print(f"  - Pretrain dataset: {config.get('pretrain_dataset', 'N/A')}")
    print(f"  - Downstream dataset: {config['downstream_dataset']}")
    print(f"  - Head type: {config['head_type']}")
    print(f"  - Task: {config['task']}")
    
    # Déterminer le type de backbone
    backbone_type = config.get('backbone_init', 'mae').lower()
    
    # Toujours créer un backbone random comme base
    backbone_random = ViT1DEncoder()
    
    # Charger le checkpoint pré-entraîné selon le type
    if backbone_type == 'random' or config.get('pretrain_dataset') == 'None':
        # Pas de pré-entraînement
        backbone = backbone_random
        print(f"✓ Backbone random (sans pré-entraînement)")
    
    elif backbone_type == 'mae':
        # Charger le checkpoint MAE
        ssl_mode = MAEModel(
            backbone=backbone_random,
            args_ssl=SimpleNamespace(mask_ratio=1.0)
        )
        
        if pretrain_checkpoint_path is None:
            # Chercher le dernier checkpoint MAE
            pretrain_dataset = config.get('pretrain_dataset', 'CWRU')
            checkpoints_dir = f"results/pretrain/MAEModel/{pretrain_dataset}Dataset"
            
            if os.path.exists(checkpoints_dir):
                checkpoints = sorted(os.listdir(checkpoints_dir))
                pretrain_checkpoint_path = os.path.join(checkpoints_dir, checkpoints[-1])
            else:
                print(f"⚠️  Répertoire de checkpoints MAE non trouvé: {checkpoints_dir}")
                print("   Utilisation d'un backbone random...")
                backbone = backbone_random
                pretrain_checkpoint_path = None
        
        if pretrain_checkpoint_path and os.path.exists(pretrain_checkpoint_path):
            ssl_model = load_model_checkpoint(ssl_mode, pretrain_checkpoint_path)
            backbone = ssl_model.backbone
            print(f"✓ Backbone MAE chargé depuis: {pretrain_checkpoint_path}")
        else:
            backbone = backbone_random
            print("⚠️  Backbone random utilisé (aucun checkpoint MAE trouvé)")
    
    elif backbone_type == 'sap':
        # Charger le checkpoint SAP
        # Note: À adapter selon votre implémentation SAP
        try:
            from models.ssl.sap import SAPModel
            ssl_mode = SAPModel(backbone=backbone_random,
                                args_ssl=dict())
        except ImportError:
            print("⚠️  SAPModel non trouvé, utilisation d'un backbone random")
            backbone = backbone_random
        else:
            if pretrain_checkpoint_path is None:
                # Chercher le dernier checkpoint SAP
                pretrain_dataset = config.get('pretrain_dataset', 'CWRU')
                checkpoints_dir = f"results/pretrain/SAPModel/{pretrain_dataset}Dataset"
                
                if os.path.exists(checkpoints_dir):
                    checkpoints = sorted(os.listdir(checkpoints_dir))
                    pretrain_checkpoint_path = os.path.join(checkpoints_dir, checkpoints[-1])
                else:
                    print(f"⚠️  Répertoire de checkpoints SAP non trouvé: {checkpoints_dir}")
                    print("   Utilisation d'un backbone random...")
                    backbone = backbone_random
                    pretrain_checkpoint_path = None
            
            if pretrain_checkpoint_path and os.path.exists(pretrain_checkpoint_path):
                ssl_model = load_model_checkpoint(ssl_mode, pretrain_checkpoint_path)
                backbone = ssl_model.backbone
                print(f"✓ Backbone SAP chargé depuis: {pretrain_checkpoint_path}")
            else:
                backbone = backbone_random
                print("⚠️  Backbone random utilisé (aucun checkpoint SAP trouvé)")
    
    else:
        print(f"⚠️  Type de backbone inconnu: {backbone_type}, utilisation d'un backbone random")
        backbone = backbone_random
    
    # Reconstruire le modèle downstream
    model = get_downstream_model(
        backbone=backbone,
        task=config['task'],
        head_type=config['head_type'],
        classes=classes if classes is not None else np.array(config.get('classes', [])),
        freeze_backbone=not config.get('finetune', True),
        device=device
    )
    
    # Charger les poids du modèle downstream
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Poids du modèle downstream chargés depuis: {model_path}")
    else:
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    
    model.to(device)
    model.eval()
    
    return model, config


def predict(
    model,
    data_loader,
    device: torch.device = "cpu",
    return_features: bool = False
):
    """
    Faire des prédictions sur un batch de données.
    
    Args:
        model: Modèle entraîné
        data_loader: DataLoader pour les données à prédire
        device: Device (cuda ou cpu)
        return_features: Si True, retourner aussi les features du backbone
    
    Returns:
        predictions: Prédictions du modèle
        true_labels: Labels vrais (si disponibles)
        features: Features du backbone (si return_features=True)
    """
    
    all_predictions = []
    all_features = []
    
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="Predicting"):
            # Déplacer le batch sur le device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch)
            
            # Récupérer les prédictions
            if model.head.__class__.__name__.endswith("ClassificationHead"):
                # Pour la classification: prendre l'argmax
                # preds = torch.argmax(outputs, dim=1)
                preds = outputs
                probs = torch.softmax(outputs, dim=1)
                targets = model.head.transform_labels(batch['y_label'])
                all_predictions.append({
                    'logits': outputs.cpu().numpy(),
                    'predictions': preds.cpu().numpy(),
                    'targets': targets.cpu().numpy(),
                    'probabilities': probs.cpu().numpy(),
                })
            else:
                # Pour la régression
                targets = batch['y_label']
                all_predictions.append({
                    'predictions': outputs.cpu().numpy(),
                    'targets': targets.cpu().numpy(),
                })
            
            # Features du backbone (si demandé)
            if return_features:
                x_raw = batch['X_raw']
                x_norm = normalization.global_z_log_normalization(
                    x=x_raw, 
                    stats=model.backbone.stats
                )
                features = model.backbone(x_norm)
                cls_token = features[:, 0]  # CLS token
                all_features.append(cls_token.cpu().numpy())
    
    # Concaténer les résultats
    predictions = {
        k: np.concatenate([d[k] for d in all_predictions])
        for k in all_predictions[0].keys()
    }
    
    features = np.concatenate(all_features) if all_features else None
    
    return predictions, features


def infer_single_sample(model, sample_data: torch.Tensor, device: torch.device = "cpu"):
    """
    Faire une prédiction sur un seul échantillon.
    
    Args:
        model: Modèle entraîné
        sample_data: Tensor de shape (1, window_size) ou dict avec clé 'X_raw'
        device: Device (cuda ou cpu)
    
    Returns:
        Prédiction pour cet échantillon
    """
    
    model.eval()
    
    with torch.no_grad():
        # Si c'est un tensor, créer un batch
        if isinstance(sample_data, torch.Tensor):
            batch = {'X_raw': sample_data.unsqueeze(0).to(device) if sample_data.dim() == 1 
                    else sample_data.to(device)}
        else:
            batch = sample_data
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
        
        outputs = model(batch)
        
        if model.head.__class__.__name__.endswith("ClassificationHead"):
            pred = torch.argmax(outputs, dim=1)
            prob = torch.softmax(outputs, dim=1)
            class_name = model.head.lb.classes_[pred.item()]
            return {
                'predicted_class_idx': pred.item(),
                'predicted_class': class_name,
                'logits': outputs.cpu().numpy(),
                'probabilities': prob.cpu().numpy(),
                'confidence': prob.max().item()
            }
        else:
            return {
                'prediction': outputs.cpu().numpy().flatten()
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inférence pour modèle downstream")
    
    parser.add_argument(
        "--model_path",
        type=str,
        help="Chemin vers le fichier du modèle (best_model.pth ou last_model.pth)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="LASPI",
        help="Dataset pour l'inférence (CWRU ou LASPI)"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=2048,
        help="Taille de la fenêtre d'entrée"
    )
    parser.add_argument(
        "--window_stride",
        type=int,
        default=256,
        help="Stride de la fenêtre"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Taille du batch"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda ou cpu)"
    )
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # ========================================
    # 1. Charger le modèle
    # ========================================
    print("\n" + "="*60)
    print("CHARGEMENT DU MODÈLE")
    print("="*60)
    
    if args.model_path is None:
        # Si aucun chemin spécifié, utiliser le dernier modèle entraîné
        print("❌ --model_path est requis!")
        print("\nExemple:")
        print("  python inference_downstream.py --model_path results/downstream/.../best_model.pth --dataset LASPI")
        exit(1)
    
    model, config = load_downstream_model(args.model_path, device=device)
    
    # ========================================
    # 2. Charger les données de test
    # ========================================
    print("\n" + "="*60)
    print("CHARGEMENT DES DONNÉES")
    print("="*60)
    
    args_dataloader = SimpleNamespace(
        name=args.dataset,
        window_size=args.window_size,
        window_stride=args.window_stride,
        batch_size=args.batch_size,
        data_ratio=1.0,
    )
    
    _, _, test_loader, _ = split_data_factory.split_dataloader(
        split_type="independent",
        dataset=args.dataset,
        args_dataloader=args_dataloader,
        seed=0,
    )
    
    print(f"✓ Test loader prêt avec {len(test_loader)} batches")
    
    # ========================================
    # 3. Faire les prédictions
    # ========================================
    print("\n" + "="*60)
    print("PRÉDICTIONS")
    print("="*60)
    
    predictions, features = predict(
        model,
        test_loader,
        device=device,
        return_features=True
    )
    
    print(f"✓ {len(predictions['predictions'])} prédictions effectuées")
    
    # ========================================
    # 4. Afficher les résultats
    # ========================================
    print("\n" + "="*60)
    print("RÉSULTATS")
    print("="*60)
    
    if true_labels is not None:
        # Classification
        correct = (predictions['predictions'] == true_labels).sum()
        accuracy = correct / len(true_labels) * 100
        print(f"Accuracy: {accuracy:.2f}% ({correct}/{len(true_labels)})")
        
        # Afficher quelques exemples
        print("\nPremiers exemples:")
        for i in range(min(5, len(predictions['predictions']))):
            pred_class = model.head.lb.classes_[predictions['predictions'][i]]
            true_class = model.head.lb.classes_[true_labels[i]]
            confidence = predictions['probabilities'][i].max()
            match = "✓" if predictions['predictions'][i] == true_labels[i] else "✗"
            print(f"  {match} Prédiction: {pred_class} ({confidence:.2%}) | Vraie: {true_class}")
    
    print("\n✓ Inférence terminée!")
    
    # Sauvegarder les résultats
    results_save_path = os.path.join(os.path.dirname(args.model_path), "inference_results.npz")
    np.savez(
        results_save_path,
        predictions=predictions['predictions'],
        probabilities=predictions.get('probabilities', None),
        true_labels=true_labels,
        features=features
    )
    print(f"✓ Résultats sauvegardés: {results_save_path}")
