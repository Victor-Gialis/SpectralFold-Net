# Guide d'Inférence - Modèle Downstream

## 📋 Résumé

Après l'entraînement downstream, le modèle est sauvegardé dans le répertoire de résultats. Ce guide explique comment charger et utiliser ce modèle pour faire des prédictions.

Les scripts supportent **trois types de backbone**:
- **MAE** (Masked AutoEncoder) - Pré-entraîné
- **SAP** - Pré-entraîné avec une autre approche SSL
- **Random/Scratch** - Backbone sans pré-entraînement

## 📁 Structure des fichiers sauvegardés

Après l'entraînement downstream, vous trouverez la structure suivante:

```
results/downstream/
├── CWRU_to_LASPI_backbone_mae_head_linear_finetune_False/
│   └── data_ratio_0.2_epochs_50/
│       ├── best_model.pth          # ← Meilleur modèle
│       ├── last_model.pth          # ← Dernier modèle
│       ├── config.json             # ← Configuration
│       └── ...
│
├── CWRU_to_LASPI_backbone_sap_head_linear_finetune_True/
│   └── data_ratio_0.2_epochs_50/
│       ├── best_model.pth
│       ├── config.json
│       └── ...
│
└── None_to_LASPI_backbone_random_head_linear_finetune_True/
    └── data_ratio_1.0_epochs_50/
        ├── best_model.pth
        ├── config.json
        └── ...
```

Les modèles pré-entraînés se trouvent dans:

```
results/pretrain/
├── MAEModel/
│   ├── CWRUDataset/
│   │   └── 20260123_162146/
│   │       └── checkpoint.pth
│   └── LASPIDataset/
│       └── ...
│
└── SAPModel/
    ├── CWRUDataset/
    │   └── 20260212_091100/
    │       └── checkpoint.pth
    └── ...
```

### Fichiers importants:

- **best_model.pth**: Poids du meilleur modèle (recommandé pour l'inférence)
- **last_model.pth**: Poids après le dernier epoch (peut être différent)
- **config.json**: Configuration (paramètres, dataset, task, etc.)

## 🚀 Méthode 1: Script d'inférence complet

### Utilisation

```bash
python inference_downstream.py \
    --model_path "results/downstream/CWRU_to_LASPI_backbone_mae_head_linear_finetune_False/data_ratio_0.2_epochs_50/best_model.pth" \
    --dataset LASPI \
    --window_size 2048 \
    --window_stride 256 \
    --batch_size 64
```

### Fonctionnalités:
- Charge le modèle downstream + backbone pré-entraîné
- Exécute l'inférence sur tout le dataset de test
- Calcule l'accuracy
- Affiche des exemples
- Sauvegarde les résultats en NPZ

### Code:

```python
from inference_downstream import load_downstream_model, predict

# Charger le modèle
model, config = load_downstream_model(
    model_path="results/downstream/.../best_model.pth",
    device="cuda"
)

# Prédictions
predictions, true_labels, features = predict(
    model, 
    test_loader,
    device="cuda",
    return_features=True
)

# Accéder aux résultats
print(predictions['predictions'])      # Indices des classes prédites
print(predictions['probabilities'])    # Probabilités
print(predictions['logits'])           # Logits bruts
```

## 🎯 Méthode 2: Wrapper simplifié (Recommandé)

### Utilisation simple

```python
from example_inference import DownstreamModelInference
import numpy as np

# Initialiser le modèle
model = DownstreamModelInference(
    model_path="results/downstream/.../best_model.pth",
    device="cuda"
)

# Créer ou charger des données (shape: n_samples, window_size)
x_test = np.random.randn(10, 2048).astype(np.float32)

# Prédictions
results = model.predict(x_test)

print("Prédictions:", results['class_names'])
print("Confidences:", results['confidence'])
print("Probabilités:", results['probabilities'])

# Extraire les features du backbone
features = model.get_features(x_test)  # shape: (10, hidden_dim)
```

### Avantages:
- Interface simple et intuitive
- Gère automatiquement le chargement du backbone
- Peut être utilisé dans un notebook
- Méthodes séparées pour prédictions et features

## 📊 Composition du modèle

```
DownstreamModel
├── backbone (Type dépend de la configuration)
│   ├── Type 1: ViT1DEncoder + poids MAE (Masked AutoEncoder)
│   ├── Type 2: ViT1DEncoder + poids SAP (autre SSL)
│   └── Type 3: ViT1DEncoder random (sans pré-entraînement)
│   
│   Pour tous les types:
│   ├── Patch Embedding
│   ├── Transformer blocks
│   ├── Normalization stats
│   └── get_attention_scores() → (n_patches,)
│
└── head (Classification or Regression)
    ├── Linear/MLP layers
    └── Label encoder
```

### Détection automatique du type de backbone

Le script détecte automatiquement le type à partir de `config.json`:
- `backbone_init: "mae"` → Charge depuis `results/pretrain/MAEModel/`
- `backbone_init: "sap"` → Charge depuis `results/pretrain/SAPModel/`
- `backbone_init: "random"` → Aucun checkpoint, backbone aléatoire
- `pretrain_dataset: "None"` → Pas de pré-entraînement

### Flow d'inférence:

```
Input: x_raw (batch, window_size)
  ↓
[Normalization] global_z_log_normalization
  ↓
[Backbone] Forward pass → (batch, n_patches, hidden_dim)
  ↓
[CLS Token Extraction] features[:, 0] → (batch, hidden_dim)
  ↓
[Head] Linear/MLP → (batch, n_classes) ou (batch, 1)
  ↓
[Output] Logits / Probabilities
```

## 💾 Configuration (config.json)

Example de contenu:

```json
{
    "backbone_init": "mae",
    "pretrain_dataset": "CWRU",
    "downstream_dataset": "LASPI",
    "task": "classification",
    "head_type": "linear",
    "finetune": false,
    "window_size": 2048,
    "window_stride": 256,
    "batch_size": 64,
    "data_ratio": 0.2,
    "learning_rate": 0.0003695,
    "epochs": 50,
    "classes": ["class1", "class2", "class3", ...]
}
```

### Clés importantes:

| Clé | Description | Valeurs possibles |
|-----|-------------|-------------------|
| `backbone_init` | Type de backbone | `"mae"`, `"sap"`, `"random"` |
| `pretrain_dataset` | Dataset de pré-entraînement | `"CWRU"`, `"LASPI"`, `"None"` |
| `downstream_dataset` | Dataset d'entraînement downstream | `"CWRU"`, `"LASPI"` |
| `task` | Type de tâche | `"classification"`, `"regression"` |
| `head_type` | Type de tête de prédiction | `"linear"`, `"non-linear"` |
| `finetune` | Fine-tuning du backbone | `true`, `false` |
| `classes` | Liste des noms de classes | `[...]` (pour classification) |

## 🔍 Accéder aux différentes sorties

### Classification:

```python
results = model.predict(x_test)

# Prédictions (indices des classes)
pred_indices = results['predictions']  # (n_samples,)

# Noms des classes
pred_classes = results['class_names']  # List[str]

# Probabilités (softmax)
probs = results['probabilities']  # (n_samples, n_classes)

# Confiance (max probability)
confidence = results['confidence']  # (n_samples,)

# Logits bruts
logits = results['predictions']  # (n_samples, n_classes)
```

### Régression:

```python
results = model.predict(x_test)

# Prédictions
predictions = results['predictions']  # (n_samples,) ou (n_samples, n_outputs)
```

## 📈 Extraire et utiliser les features

Les features extraites du CLS token du backbone peuvent être utiles pour:
- Clustering
- Visualisation (t-SNE, UMAP)
- Comme input pour un autre modèle
- Analyse d'interprétabilité

```python
# Extraire les features
features = model.get_features(x_test)  # (n_samples, hidden_dim)

# Normaliser
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Visualisation t-SNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
features_2d = tsne.fit_transform(features)
```

## ⚠️ Points importants

### 1. Normalisation des données
Les données sont automatiquement normalisées avec `global_z_log_normalization` en utilisant les statistiques du backbone:

```python
from dataset.transform import normalization

x_norm = normalization.global_z_log_normalization(
    x=x_raw,
    stats=model.backbone.stats  # Chargées automatiquement
)
```

### 2. Device (GPU/CPU)
```python
# Sur GPU
model = DownstreamModelInference(model_path, device="cuda")

# Sur CPU (plus lent)
model = DownstreamModelInference(model_path, device="cpu")
```

### 3. Taille des entrées
Les données doivent avoir la même dimension que celles utilisées à l'entraînement:
- Input: (n_samples, window_size)
- Par défaut: window_size = 2048

### 4. Mode évaluation
Le modèle est automatiquement mis en `eval()` pour:
- Désactiver les dropout
- Fixer les batch norm statistics
- Éviter la rétro-propagation

## 🔧 Dépannage

### "Config file not found"
→ Vérifier que le chemin `model_path` pointe vers un fichier `.pth` dans le bon répertoire

### "Backbone checkpoint not found"
→ Vérifier que le pré-entraînement MAE s'est bien terminé et que les résultats sont dans `results/pretrain/`

### "CUDA out of memory"
→ Réduire `batch_size` ou utiliser `device="cpu"`

### "Shape mismatch"
→ Vérifier que l'input a shape `(n_samples, 2048)` (par défaut)

## 📚 Fichiers connexes

- `inference_downstream.py` - Script complet d'inférence avec argparse
- `example_inference.py` - Wrapper simplifié avec classe `DownstreamModelInference`
- `submit_mae_downstream.py` - Script d'entraînement
- `training/downstream.py` - Code d'entraînement

## 💡 Exemple complet

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/ngrotus/Desktop/CWRU_MAE')

from example_inference import DownstreamModelInference
import numpy as np
import matplotlib.pyplot as plt

# 1. Charger le modèle
model = DownstreamModelInference(
    "results/downstream/CWRU_to_LASPI_backbone_mae_head_linear_finetune_False/data_ratio_0.2_epochs_50/best_model.pth",
    device="cuda"
)

# 2. Charger vos données
# x_test = np.load("your_data.npy")  # (n_samples, 2048)

# 3. Faire des prédictions
results = model.predict(x_test)

# 4. Afficher les résultats
for i in range(len(x_test)):
    print(f"Sample {i}: {results['class_names'][i]} ({results['confidence'][i]:.2%})")

# 5. Extraire et analyser les features
features = model.get_features(x_test)
print(f"Features shape: {features.shape}")

# 6. Visualiser
plt.figure(figsize=(10, 5))
plt.imshow(x_test[:10], aspect='auto')
plt.colorbar()
plt.title('Exemples d\'entrées')
plt.show()
```

## 🎓 Ressources utiles

- **ViT1DEncoder**: Backbone Vision Transformer pour les signaux 1D
- **MAEModel**: Modèle de pré-entraînement auto-supervisé (Masked AutoEncoder)
- **DownstreamModel**: Modèle de tâche aval (classification/régression)
- **normalization.global_z_log_normalization**: Normalisation des signaux

---

Pour plus d'aide, consultez le code source ou les notebooks d'exemple.
