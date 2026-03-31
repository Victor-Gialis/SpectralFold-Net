"""
Comparaison des trois types de backbone:
- MAE (Masked AutoEncoder)
- SAP (Self-supervised Approach)
- Random/Scratch (sans pré-entraînement)
"""

import sys
sys.path.insert(0, '/home/ngrotus/Desktop/CWRU_MAE')

import numpy as np
import torch
from example_inference import DownstreamModelInference
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from types import SimpleNamespace
from dataset import split_data_factory


def load_and_evaluate_all_models():
    """
    Charger et évaluer les trois types de modèles.
    """
    
    print("\n" + "="*80)
    print("COMPARAISON: MAE vs SAP vs Random")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Les trois configurations
    models_config = {
        "MAE": "results/downstream/CWRU_to_LASPI_backbone_mae_head_linear_finetune_False/data_ratio_0.2_epochs_50/best_model.pth",
        "SAP": "results/downstream/CWRU_to_LASPI_backbone_sap_head_linear_finetune_False/data_ratio_0.2_epochs_50/best_model.pth",
        "Random": "results/downstream/None_to_LASPI_backbone_random_head_linear_finetune_True/data_ratio_1.0_epochs_50/best_model.pth",
    }
    
    # Charger les données de test
    args_dataloader = SimpleNamespace(
        name="LASPI",
        window_size=2048,
        window_stride=256,
        batch_size=64,
        data_ratio=1.0,
    )
    
    _, _, test_loader, _ = split_data_factory.split_dataloader(
        split_type="independent",
        dataset="LASPI",
        args_dataloader=args_dataloader,
        seed=0,
    )
    
    # Dictionnaire pour stocker les résultats
    results = {}
    
    for backbone_type, model_path in models_config.items():
        print(f"\n{'─'*80}")
        print(f"Chargement du modèle: {backbone_type}")
        print(f"{'─'*80}")
        
        try:
            # Charger le modèle
            model = DownstreamModelInference(model_path, device=device)
            
            # Évaluation
            all_preds = []
            all_trues = []
            all_features = []
            
            with torch.no_grad():
                for batch in test_loader:
                    x = batch['X_raw'].numpy()
                    y = batch['y_label'].numpy()
                    
                    pred_results = model.predict(x)
                    features = model.get_features(x)
                    
                    all_preds.extend(pred_results['predictions'])
                    all_trues.extend(y)
                    all_features.append(features)
            
            all_preds = np.array(all_preds)
            all_trues = np.array(all_trues)
            all_features = np.concatenate(all_features, axis=0)
            
            # Calculer la métrique
            accuracy = accuracy_score(all_trues, all_preds)
            
            results[backbone_type] = {
                'accuracy': accuracy,
                'predictions': all_preds,
                'true_labels': all_trues,
                'features': all_features,
                'model': model
            }
            
            print(f"✓ {backbone_type}: Accuracy = {accuracy:.4f} ({(all_preds == all_trues).sum()}/{len(all_trues)})")
            
        except FileNotFoundError as e:
            print(f"❌ {backbone_type}: Model not found")
            print(f"   {e}")
        except Exception as e:
            print(f"❌ {backbone_type}: Error during evaluation")
            print(f"   {e}")
    
    return results


def plot_comparison(results):
    """
    Visualiser la comparaison entre les trois backbones.
    """
    print("\n" + "="*80)
    print("VISUALISATION DES RÉSULTATS")
    print("="*80)
    
    if not results:
        print("❌ Aucun résultat à visualiser")
        return
    
    # 1. Comparaison des accuracies
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot 1: Accuracy comparison
    backbone_types = list(results.keys())
    accuracies = [results[t]['accuracy'] for t in backbone_types]
    
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']  # Green, Red, Gray
    ax = axes[0, 0]
    bars = ax.bar(backbone_types, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
    ax.set_title('Comparaison des Accuracies', fontweight='bold', fontsize=13)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Ajouter les valeurs sur les barres
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2%}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Subplot 2: Confidence distribution
    ax = axes[0, 1]
    for backbone_type, color in zip(backbone_types, colors):
        if backbone_type in results:
            model = results[backbone_type]['model']
            # Récréer les confidences
            x = results[backbone_type]['features'][:100]  # Utiliser les features
            if x.size > 0:
                ax.hist(results[backbone_type]['predictions'], bins=20, alpha=0.5, 
                       label=backbone_type, color=color, edgecolor='black')
    
    ax.set_xlabel('Classe prédite', fontweight='bold', fontsize=11)
    ax.set_ylabel('Nombre de prédictions', fontweight='bold', fontsize=11)
    ax.set_title('Distribution des prédictions', fontweight='bold', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Subplot 3: Feature statistics
    ax = axes[1, 0]
    for backbone_type, color in zip(backbone_types, colors):
        if backbone_type in results:
            features = results[backbone_type]['features']
            feature_norms = np.linalg.norm(features, axis=1)
            ax.hist(feature_norms, bins=30, alpha=0.5, label=backbone_type, 
                   color=color, edgecolor='black')
    
    ax.set_xlabel('Feature norm (L2)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Fréquence', fontweight='bold', fontsize=11)
    ax.set_title('Distribution des normes de features', fontweight='bold', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Subplot 4: Résumé textuel
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = "RÉSUMÉ DES RÉSULTATS\n" + "─"*40 + "\n"
    for backbone_type, color in zip(backbone_types, colors):
        if backbone_type in results:
            acc = results[backbone_type]['accuracy']
            n_correct = (results[backbone_type]['predictions'] == results[backbone_type]['true_labels']).sum()
            n_total = len(results[backbone_type]['true_labels'])
            summary_text += f"\n{backbone_type}:\n"
            summary_text += f"  • Accuracy: {acc:.2%}\n"
            summary_text += f"  • Correct: {n_correct}/{n_total}\n"
            summary_text += f"  • Features shape: {results[backbone_type]['features'].shape}\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('backbone_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Graphique sauvegardé: backbone_comparison.png")
    
    return fig


def print_detailed_comparison(results):
    """
    Afficher une comparaison détaillée.
    """
    print("\n" + "="*80)
    print("COMPARAISON DÉTAILLÉE")
    print("="*80)
    
    # Créer un tableau comparatif
    print("\n┌─────────────┬──────────────┬──────────┬─────────┬──────────────┐")
    print("│ Backbone    │ Accuracy     │ Correct  │ Dataset │ Feature Dim  │")
    print("├─────────────┼──────────────┼──────────┼─────────┼──────────────┤")
    
    for backbone_type in ['MAE', 'SAP', 'Random']:
        if backbone_type in results:
            acc = results[backbone_type]['accuracy']
            n_correct = (results[backbone_type]['predictions'] == results[backbone_type]['true_labels']).sum()
            n_total = len(results[backbone_type]['true_labels'])
            feature_dim = results[backbone_type]['features'].shape[1]
            dataset = 'LASPI'
            
            print(f"│ {backbone_type:<11} │ {acc:>12.2%} │ {n_correct:>8}/{n_total} │ {dataset:<7} │ {feature_dim:>12} │")
    
    print("└─────────────┴──────────────┴──────────┴─────────┴──────────────┘")
    
    # Quelle config a la meilleure performance?
    if results:
        best_backbone = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_accuracy = results[best_backbone]['accuracy']
        
        print(f"\n🏆 Meilleure performance: {best_backbone} ({best_accuracy:.2%})")


def feature_analysis(results):
    """
    Analyser les features extraites par chaque backbone.
    """
    print("\n" + "="*80)
    print("ANALYSE DES FEATURES")
    print("="*80)
    
    for backbone_type in ['MAE', 'SAP', 'Random']:
        if backbone_type in results:
            features = results[backbone_type]['features']
            
            print(f"\n{backbone_type}:")
            print(f"  - Shape: {features.shape}")
            print(f"  - Mean norm: {np.linalg.norm(features, axis=1).mean():.4f}")
            print(f"  - Std norm:  {np.linalg.norm(features, axis=1).std():.4f}")
            print(f"  - Min value: {features.min():.6f}")
            print(f"  - Max value: {features.max():.6f}")
            print(f"  - Mean:      {features.mean():.6f}")
            print(f"  - Std:       {features.std():.6f}")


if __name__ == "__main__":
    # Charger et évaluer tous les modèles
    results = load_and_evaluate_all_models()
    
    if results:
        # Afficher des comparaisons
        print_detailed_comparison(results)
        feature_analysis(results)
        
        # Visualiser
        plot_comparison(results)
        
        print("\n" + "="*80)
        print("✓ COMPARAISON TERMINÉE")
        print("="*80)
    else:
        print("\n❌ Aucun modèle n'a pu être chargé")
        print("   Vérifier que les chemins sont corrects et que les modèles existent")
