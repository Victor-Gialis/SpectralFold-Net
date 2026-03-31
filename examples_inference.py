"""
Exemples concrets d'utilisation du modèle downstream pour l'inférence.

Ces exemples montrent comment utiliser le modèle dans différents scénarios.
"""

import sys
sys.path.insert(0, '/home/ngrotus/Desktop/CWRU_MAE')

import numpy as np
import torch
from example_inference import DownstreamModelInference
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# ==============================================================================
# EXEMPLE 1: Prédiction simple sur un batch
# ==============================================================================

def example_simple_prediction():
    """
    Charger le modèle et faire une prédiction simple.
    """
    print("\n" + "="*70)
    print("EXEMPLE 1: Prédiction simple")
    print("="*70)
    
    # Charger le modèle
    model = DownstreamModelInference(
        "results/downstream/CWRU_to_LASPI_backbone_mae_head_linear_finetune_False/data_ratio_0.2_epochs_50/best_model.pth",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Données de test (2 échantillons)
    x_test = np.random.randn(2, 2048).astype(np.float32)
    
    # Prédictions
    results = model.predict(x_test)
    
    # Afficher les résultats
    print("\nRésultats:")
    for i in range(len(x_test)):
        print(f"  Échantillon {i}:")
        print(f"    - Classe prédite: {results['class_names'][i]}")
        print(f"    - Confiance: {results['confidence'][i]:.2%}")
        print(f"    - Probabilités: {results['probabilities'][i]}")
    
    return model, x_test, results


# ==============================================================================
# EXEMPLE 2: Batch processing avec dataloader
# ==============================================================================

def example_batch_processing():
    """
    Traiter un batch complet de données avec le dataloader.
    """
    print("\n" + "="*70)
    print("EXEMPLE 2: Batch processing")
    print("="*70)
    
    from types import SimpleNamespace
    from dataset import split_data_factory
    
    # Charger le modèle
    model = DownstreamModelInference(
        "results/downstream/CWRU_to_LASPI_backbone_mae_head_linear_finetune_False/data_ratio_0.2_epochs_50/best_model.pth",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Créer un dataloader
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
    
    # Traiter un batch
    batch = next(iter(test_loader))
    x = batch['X_raw'].numpy()
    
    results = model.predict(x)
    
    print(f"\nBatch de {len(x)} échantillons traité")
    print(f"Accuracy (sur le batch): {(results['predictions'] == batch['y_label'].numpy()).sum() / len(x):.2%}")
    
    return model, test_loader


# ==============================================================================
# EXEMPLE 3: Extraction des features pour du clustering
# ==============================================================================

def example_feature_extraction_and_clustering():
    """
    Extraire les features du backbone et les utiliser pour du clustering/visualisation.
    """
    print("\n" + "="*70)
    print("EXEMPLE 3: Feature extraction et clustering")
    print("="*70)
    
    from types import SimpleNamespace
    from dataset import split_data_factory
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # Charger le modèle
    model = DownstreamModelInference(
        "results/downstream/CWRU_to_LASPI_backbone_mae_head_linear_finetune_False/data_ratio_0.2_epochs_50/best_model.pth",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Charger les données
    args_dataloader = SimpleNamespace(
        name="LASPI",
        window_size=2048,
        window_stride=256,
        batch_size=64,
        data_ratio=1.0,
    )
    
    _, _, test_loader, labels = split_data_factory.split_dataloader(
        split_type="independent",
        dataset="LASPI",
        args_dataloader=args_dataloader,
        seed=0,
    )
    
    # Extraire toutes les features
    all_features = []
    all_labels = []
    
    for batch in test_loader:
        x = batch['X_raw'].numpy()
        features = model.get_features(x)
        all_features.append(features)
        all_labels.extend(batch['y_label'].numpy())
    
    features = np.concatenate(all_features, axis=0)
    labels = np.array(all_labels)
    
    print(f"\nFeatures extraites: shape {features.shape}")
    print(f"  - N samples: {features.shape[0]}")
    print(f"  - Feature dim: {features.shape[1]}")
    
    # K-means clustering
    kmeans = KMeans(n_clusters=len(np.unique(labels)), random_state=0)
    kmeans_labels = kmeans.fit_predict(features)
    
    # Accuracy du clustering
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(labels, kmeans_labels)
    print(f"\nAdjusted Rand Index (clustering): {ari:.3f}")
    
    # t-SNE visualization
    print("\nCalcul t-SNE (peut prendre du temps)...")
    tsne = TSNE(n_components=2, random_state=0, n_iter=1000)
    features_2d = tsne.fit_transform(features)
    
    # Visualiser
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Vraies labels
    scatter1 = ax1.scatter(features_2d[:, 0], features_2d[:, 1], 
                          c=labels, cmap='tab10', alpha=0.6, s=30)
    ax1.set_title('t-SNE: Vraies labels')
    plt.colorbar(scatter1, ax=ax1)
    
    # K-means clustering
    scatter2 = ax2.scatter(features_2d[:, 0], features_2d[:, 1], 
                          c=kmeans_labels, cmap='tab10', alpha=0.6, s=30)
    ax2.set_title('t-SNE: K-means clustering')
    plt.colorbar(scatter2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('clustering_visualization.png', dpi=100, bbox_inches='tight')
    print("\n✓ Visualisation sauvegardée: clustering_visualization.png")
    
    return model, features, labels


# ==============================================================================
# EXEMPLE 4: Prédictions avec confidence et seuil de décision
# ==============================================================================

def example_confidence_filtering():
    """
    Filtrer les prédictions par confiance (confidence threshold).
    """
    print("\n" + "="*70)
    print("EXEMPLE 4: Confidence filtering")
    print("="*70)
    
    # Charger le modèle
    model = DownstreamModelInference(
        "results/downstream/CWRU_to_LASPI_backbone_mae_head_linear_finetune_False/data_ratio_0.2_epochs_50/best_model.pth",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Données de test
    x_test = np.random.randn(100, 2048).astype(np.float32)
    results = model.predict(x_test)
    
    # Seuil de confiance
    confidence_threshold = 0.8
    
    confident_mask = results['confidence'] >= confidence_threshold
    uncertain_mask = ~confident_mask
    
    print(f"\nSur {len(x_test)} prédictions:")
    print(f"  - Confides (>= {confidence_threshold}): {confident_mask.sum()}")
    print(f"  - Incertaines (< {confidence_threshold}): {uncertain_mask.sum()}")
    
    print(f"\nPrédictions incertaines (exemples):")
    for i in np.where(uncertain_mask)[0][:5]:
        print(f"  - {results['class_names'][i]} ({results['confidence'][i]:.2%})")
    
    return model, results, confident_mask


# ==============================================================================
# EXEMPLE 5: Comparaison top-k predictions
# ==============================================================================

def example_top_k_predictions():
    """
    Afficher les top-k classes les plus probables pour chaque prédiction.
    """
    print("\n" + "="*70)
    print("EXEMPLE 5: Top-K predictions")
    print("="*70)
    
    # Charger le modèle
    model = DownstreamModelInference(
        "results/downstream/CWRU_to_LASPI_backbone_mae_head_linear_finetune_False/data_ratio_0.2_epochs_50/best_model.pth",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Données de test
    x_test = np.random.randn(5, 2048).astype(np.float32)
    results = model.predict(x_test)
    
    k = 3
    print(f"\nTop-{k} prédictions:")
    
    for i in range(len(x_test)):
        # Récupérer les indices des top-k classes
        top_k_indices = np.argsort(results['probabilities'][i])[-k:][::-1]
        
        print(f"\n  Échantillon {i}:")
        for rank, idx in enumerate(top_k_indices, 1):
            class_name = model.model.head.lb.classes_[idx]
            prob = results['probabilities'][i][idx]
            print(f"    {rank}. {class_name}: {prob:.2%}")
    
    return model, results


# ==============================================================================
# EXEMPLE 6: Prédictions sur des données brutes (numpy arrays)
# ==============================================================================

def example_raw_numpy_prediction():
    """
    Faire des prédictions directement à partir de numpy arrays.
    """
    print("\n" + "="*70)
    print("EXEMPLE 6: Prédiction sur numpy arrays")
    print("="*70)
    
    # Charger le modèle
    model = DownstreamModelInference(
        "results/downstream/CWRU_to_LASPI_backbone_mae_head_linear_finetune_False/data_ratio_0.2_epochs_50/best_model.pth",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Créer des données de test
    x_single = np.random.randn(2048).astype(np.float32)
    x_batch = np.random.randn(10, 2048).astype(np.float32)
    
    # Prédictions
    result_single = model.predict(x_single)
    result_batch = model.predict(x_batch)
    
    print(f"\nPrédiction sur single sample (shape {x_single.shape}):")
    print(f"  - Classe: {result_single['class_names'][0]}")
    print(f"  - Confiance: {result_single['confidence'][0]:.2%}")
    
    print(f"\nPrédiction sur batch (shape {x_batch.shape}):")
    print(f"  - Classes: {result_batch['class_names']}")
    print(f"  - Confidences: {result_batch['confidence']}")
    
    return model, x_single, x_batch, result_single, result_batch


# ==============================================================================
# EXEMPLE 7: Pipeline complet avec sauvegarde des résultats
# ==============================================================================

def example_complete_pipeline():
    """
    Pipeline complet: charger, prédire, sauvegarder.
    """
    print("\n" + "="*70)
    print("EXEMPLE 7: Pipeline complet")
    print("="*70)
    
    import os
    from datetime import datetime
    
    # Charger le modèle
    model_path = "results/downstream/CWRU_to_LASPI_backbone_mae_head_linear_finetune_False/data_ratio_0.2_epochs_50/best_model.pth"
    model = DownstreamModelInference(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Données
    x_test = np.random.randn(50, 2048).astype(np.float32)
    
    # Prédictions
    predictions = model.predict(x_test)
    features = model.get_features(x_test)
    
    # Créer un répertoire de résultats
    results_dir = f"inference_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Sauvegarder les résultats
    np.save(os.path.join(results_dir, "predictions.npy"), predictions['predictions'])
    np.save(os.path.join(results_dir, "probabilities.npy"), predictions['probabilities'])
    np.save(os.path.join(results_dir, "features.npy"), features)
    np.save(os.path.join(results_dir, "input_data.npy"), x_test)
    
    # Sauvegarder les metadata
    import json
    metadata = {
        "model_path": model_path,
        "n_samples": len(x_test),
        "n_classes": len(np.unique(predictions['predictions'])),
        "feature_dim": features.shape[1],
        "timestamp": datetime.now().isoformat()
    }
    
    with open(os.path.join(results_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Résultats sauvegardés dans: {results_dir}/")
    print(f"  - predictions.npy")
    print(f"  - probabilities.npy")
    print(f"  - features.npy")
    print(f"  - input_data.npy")
    print(f"  - metadata.json")
    
    return results_dir


# ==============================================================================
# MAIN: Exécuter tous les exemples
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  EXEMPLES D'INFÉRENCE - MODÈLE DOWNSTREAM  ".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    try:
        # Exemple 1
        model1, x_test1, results1 = example_simple_prediction()
        
        # Exemple 2
        model2, test_loader = example_batch_processing()
        
        # Exemple 3
        # model3, features, labels = example_feature_extraction_and_clustering()
        
        # Exemple 4
        model4, results4, confident_mask = example_confidence_filtering()
        
        # Exemple 5
        model5, results5 = example_top_k_predictions()
        
        # Exemple 6
        model6, x_single, x_batch, result_single, result_batch = example_raw_numpy_prediction()
        
        # Exemple 7
        results_dir = example_complete_pipeline()
        
        print("\n" + "="*70)
        print("✓ TOUS LES EXEMPLES TERMINÉS")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
