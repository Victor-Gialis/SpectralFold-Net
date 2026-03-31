#!/usr/bin/env python3
"""
DÉMARRAGE ULTRA-RAPIDE - Inférence downstream

Copier-coller ce code pour des prédictions immédiates avec MAE, SAP, ou Random!
"""

import sys
sys.path.insert(0, '/home/ngrotus/Desktop/CWRU_MAE')

import numpy as np
from example_inference import DownstreamModelInference

print("="*70)
print("INFÉRENCE RAPIDE - MAE, SAP, Random")
print("="*70)

# ============================================================================
# OPTION 1: MAE
# ============================================================================
print("\n1️⃣  MAE (Masked AutoEncoder)")
print("-" * 70)

try:
    mae_model = DownstreamModelInference(
        "results/downstream/CWRU_to_LASPI_backbone_mae_head_linear_finetune_False/data_ratio_0.2_epochs_50/best_model.pth"
    )
    
    # Données de test
    x = np.random.randn(3, 2048).astype(np.float32)
    
    # Prédictions
    mae_results = mae_model.predict(x)
    
    print("\n✅ MAE - Prédictions:")
    for i, (class_name, conf) in enumerate(zip(mae_results['class_names'], mae_results['confidence'])):
        print(f"   Sample {i}: {class_name:<15} (confidence: {conf:.2%})")
    
except Exception as e:
    print(f"⚠️  MAE: {e}")

# ============================================================================
# OPTION 2: SAP
# ============================================================================
print("\n2️⃣  SAP (Self-Supervised Approach)")
print("-" * 70)

try:
    sap_model = DownstreamModelInference(
        "results/downstream/CWRU_to_LASPI_backbone_sap_head_linear_finetune_False/data_ratio_0.2_epochs_50/best_model.pth"
    )
    
    # Données de test
    x = np.random.randn(3, 2048).astype(np.float32)
    
    # Prédictions
    sap_results = sap_model.predict(x)
    
    print("\n✅ SAP - Prédictions:")
    for i, (class_name, conf) in enumerate(zip(sap_results['class_names'], sap_results['confidence'])):
        print(f"   Sample {i}: {class_name:<15} (confidence: {conf:.2%})")
    
except Exception as e:
    print(f"⚠️  SAP: {e}")

# ============================================================================
# OPTION 3: Random/Scratch
# ============================================================================
print("\n3️⃣  Random/Scratch (sans pré-entraînement)")
print("-" * 70)

try:
    random_model = DownstreamModelInference(
        "results/downstream/None_to_LASPI_backbone_random_head_linear_finetune_True/data_ratio_1.0_epochs_50/best_model.pth"
    )
    
    # Données de test
    x = np.random.randn(3, 2048).astype(np.float32)
    
    # Prédictions
    random_results = random_model.predict(x)
    
    print("\n✅ Random - Prédictions:")
    for i, (class_name, conf) in enumerate(zip(random_results['class_names'], random_results['confidence'])):
        print(f"   Sample {i}: {class_name:<15} (confidence: {conf:.2%})")
    
except Exception as e:
    print(f"⚠️  Random: {e}")

# ============================================================================
# COMPARAISON SIMPLE
# ============================================================================
print("\n" + "="*70)
print("COMPARAISON RAPIDE")
print("="*70)

print("""
Utilisez compare_backbones.py pour une comparaison complète:

    python compare_backbones.py

Cela affichera:
  ✓ Accuracy de chaque type
  ✓ Distribution des prédictions
  ✓ Statistiques des features
  ✓ Graphiques de comparaison
""")

# ============================================================================
# EXTRACTION DE FEATURES
# ============================================================================
print("\n" + "="*70)
print("EXTRACTION DE FEATURES")
print("="*70)

try:
    x = np.random.randn(10, 2048).astype(np.float32)
    
    mae_features = mae_model.get_features(x)
    sap_features = sap_model.get_features(x)
    random_features = random_model.get_features(x)
    
    print(f"\n✅ Features extraites:")
    print(f"   MAE:    shape {mae_features.shape}")
    print(f"   SAP:    shape {sap_features.shape}")
    print(f"   Random: shape {random_features.shape}")
    
except Exception as e:
    print(f"⚠️  Features: {e}")

# ============================================================================
# RESSOURCES
# ============================================================================
print("\n" + "="*70)
print("📚 RESSOURCES")
print("="*70)

print("""
Consultez ces fichiers pour plus d'informations:

  1️⃣  README_INFERENCE.md        → TL;DR (2 min)
  2️⃣  BACKBONE_COMPARISON.md     → Guide des 3 types (10 min)
  3️⃣  examples_inference.py       → 7 exemples complets
  4️⃣  compare_backbones.py        → Comparaison automatique
  5️⃣  INFERENCE_GUIDE.md          → Documentation détaillée

Commencez par: README_INFERENCE.md
""")

print("✅ Script terminé!")
