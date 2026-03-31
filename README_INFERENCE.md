# 🎯 Inférence du Modèle Downstream

## TL;DR (Résumé rapide)

Après l'entraînement downstream, voici comment utiliser le modèle. **Automatiquement supporte MAE, SAP, et Random (Scratch):**

### Option 1: Wrapper simplifié (⭐ Recommandé)

```python
from example_inference import DownstreamModelInference
import numpy as np

# Charger
model = DownstreamModelInference(
    "results/downstream/CWRU_to_LASPI_backbone_mae_head_linear_finetune_False/data_ratio_0.2_epochs_50/best_model.pth",
    device="cuda"
)

# Prédire
x = np.random.randn(10, 2048)  # (n_samples, window_size)
results = model.predict(x)
print(results['class_names'])  # Classes prédites
print(results['confidence'])   # Confidences

# Features
features = model.get_features(x)  # (10, hidden_dim)
```

### Option 2: Script complet

```bash
python inference_downstream.py \
    --model_path "results/downstream/.../best_model.pth" \
    --dataset LASPI
```

## 📁 Fichiers créés

| Fichier | Description |
|---------|------------|
| `example_inference.py` | **⭐ Classe wrapper DownstreamModelInference** |
| `inference_downstream.py` | Script complet avec argparse |
| `examples_inference.py` | 7 exemples d'utilisation |
| `INFERENCE_GUIDE.md` | Documentation détaillée |
| `README_INFERENCE.md` | Ce fichier |

## 🚀 Démarrage rapide

### 1. Lister les modèles disponibles

```bash
# MAE models
ls results/downstream/CWRU_to_LASPI_backbone_mae*/*/best_model.pth

# SAP models
ls results/downstream/CWRU_to_LASPI_backbone_sap*/*/best_model.pth

# Random/Scratch models
ls results/downstream/None_to_*/*/best_model.pth
```

### 2. Exemple minimal

```python
import sys
sys.path.insert(0, '/home/ngrotus/Desktop/CWRU_MAE')
from example_inference import DownstreamModelInference
import numpy as np

# Fonctionne pour MAE, SAP et Random
model = DownstreamModelInference(
    "results/downstream/CWRU_to_LASPI_backbone_mae_head_linear_finetune_False/data_ratio_0.2_epochs_50/best_model.pth"
)
x = np.random.randn(5, 2048)
print(model.predict(x)['class_names'])
```

## 📊 Structure du modèle chargé

```
DownstreamModel
├── backbone (Type détecté automatiquement)
│   ├── Si backbone_init="mae"    → ViT1D + poids MAE
│   ├── Si backbone_init="sap"    → ViT1D + poids SAP
│   └── Si backbone_init="random" → ViT1D aléatoire
│   
│   Pour tous les types:
│   ├── Patch Embedding
│   ├── Transformer blocks
│   ├── Normalization stats
│   └── get_attention_scores() → (n_patches,)
│
└── head (Classification/Regression)
    ├── Linear ou MLP layers
    └── Label encoder
```

## 📦 Résultats retournés

### Pour la classification:
```python
{
    'predictions': np.array([0, 1, 2, ...]),      # Indices des classes
    'class_names': ['class1', 'class2', ...],     # Noms des classes
    'probabilities': np.array([[0.9, 0.1], ...]), # Softmax
    'confidence': np.array([0.9, 0.8, ...]),      # Max probability
}
```

### Features:
```python
features = model.get_features(x)  # (n_samples, hidden_dim)
```

## 💾 Configuration du modèle (config.json)

Chaque répertoire de modèle contient un `config.json` avec:
- `pretrain_dataset`, `downstream_dataset`
- `task`, `head_type`, `finetune`
- Paramètres d'entraînement
- Liste des classes

## 🔍 Points clés

1. **Détection automatique**: Pas besoin de spécifier le type, il est détecté depuis config.json
2. **Normalisation automatique**: Les données sont normalisées avec `global_z_log_normalization`
3. **Input shape**: (n_samples, 2048) par défaut
4. **GPU/CPU**: Spécifier `device="cuda"` ou `device="cpu"`
5. **Mode eval**: Automatiquement activé (pas de dropout, batch norm figées)

## 🎓 Exemples

### Exemple 1: Prédictions simples
```python
model = DownstreamModelInference(model_path)
results = model.predict(x)
print(f"Prédictions: {results['class_names']}")
print(f"Confidences: {results['confidence']}")
```

### Exemple 2: Extraction de features pour clustering
```python
features = model.get_features(x)
# Utiliser avec sklearn, t-SNE, UMAP, etc.
```

### Exemple 3: Filtrage par confiance
```python
results = model.predict(x)
confident = results['confidence'] > 0.8
uncertain = x[~confident]  # Données incertaines
```

### Exemple 4: Top-K predictions
```python
results = model.predict(x)
top_k = np.argsort(results['probabilities'][0])[-3:]  # Top 3 classes
```

## 📚 Documentation complète

Voir `INFERENCE_GUIDE.md` pour:
- Structure des fichiers sauvegardés
- Explication détaillée du flow d'inférence
- Tous les paramètres et options
- Dépannage et FAQ
- Ressources utiles

## 🧪 Exécuter les exemples

```bash
# Tous les exemples
python examples_inference.py

# Script complet avec argparse
python inference_downstream.py --model_path results/downstream/.../best_model.pth
```

## ❓ FAQ

**Q: Où sont sauvegardés les modèles?**  
R: `results/downstream/CWRU_to_LASPI_backbone_mae_.../data_ratio_.../best_model.pth`

**Q: Dois-je charger le backbone MAE manuellement?**  
R: Non, `DownstreamModelInference` le fait automatiquement.

**Q: Quelle est la taille d'entrée?**  
R: (n_samples, 2048) par défaut (window_size)

**Q: Puis-je utiliser le CPU?**  
R: Oui, `device="cpu"` (plus lent)

**Q: Comment extraire les features?**  
R: `features = model.get_features(x)`

---

**Besoin d'aide?** Consultez `INFERENCE_GUIDE.md` ou les exemples dans `examples_inference.py`
