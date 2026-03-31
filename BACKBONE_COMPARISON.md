# 📊 Inférence - Guide complet pour les 3 types de backbone

## 🎯 Résumé rapide

Vous avez trois types de modèles de fondation:
1. **MAE** (Masked AutoEncoder) - Pré-entraîné
2. **SAP** - Pré-entraîné
3. **Random/Scratch** - Sans pré-entraînement

Tous peuvent être chargés exactement de la même manière - **la détection est automatique**!

## 🚀 Usage rapide

```python
from example_inference import DownstreamModelInference
import numpy as np

# Fonctionne pour les 3 types
model = DownstreamModelInference(
    "results/downstream/CWRU_to_LASPI_backbone_mae_head_linear_finetune_False/data_ratio_0.2_epochs_50/best_model.pth"
)

# Prédictions
x = np.random.randn(10, 2048)
results = model.predict(x)
print(results['class_names'])
```

## 📁 Fichiers d'inférence

### Scripts principaux

| Fichier | Description | Usage |
|---------|-------------|-------|
| `example_inference.py` | **⭐ Wrapper simplifié** | `from example_inference import DownstreamModelInference` |
| `inference_downstream.py` | Script complet avec argparse | `python inference_downstream.py --model_path ...` |
| `examples_inference.py` | 7 exemples concrets | `python examples_inference.py` |
| `compare_backbones.py` | **Comparer MAE vs SAP vs Random** | `python compare_backbones.py` |

### Documentation

| Fichier | Description |
|---------|-------------|
| `README_INFERENCE.md` | TL;DR - démarrage rapide |
| `INFERENCE_GUIDE.md` | Documentation détaillée |
| `BACKBONE_COMPARISON.md` | Ce fichier - guide des 3 types |

## 🔧 Détection automatique du type

La détection se fait depuis `config.json` :

```json
{
    "backbone_init": "mae",           ← Détermine le type
    "pretrain_dataset": "CWRU",       ← Détermine le dataset
    ...
}
```

### Type 1: MAE
- `backbone_init`: `"mae"`
- Charge depuis: `results/pretrain/MAEModel/{dataset}Dataset/`
- Description: Vision Transformer + pré-entraînement Masked AutoEncoder

```python
model = DownstreamModelInference(
    "results/downstream/CWRU_to_LASPI_backbone_mae_head_linear_finetune_False/.../best_model.pth"
)
# Charge automatiquement les poids MAE
```

### Type 2: SAP
- `backbone_init`: `"sap"`
- Charge depuis: `results/pretrain/SAPModel/{dataset}Dataset/`
- Description: Vision Transformer + pré-entraînement SAP

```python
model = DownstreamModelInference(
    "results/downstream/CWRU_to_LASPI_backbone_sap_head_linear_finetune_True/.../best_model.pth"
)
# Charge automatiquement les poids SAP
```

### Type 3: Random/Scratch
- `backbone_init`: `"random"` ou `pretrain_dataset`: `"None"`
- Aucun checkpoint à charger
- Description: Vision Transformer avec initialisation aléatoire

```python
model = DownstreamModelInference(
    "results/downstream/None_to_LASPI_backbone_random_head_linear_finetune_True/.../best_model.pth"
)
# Pas de checkpoint, utilise juste le backbone aléatoire
```

## 📊 Comparaison des performances

Pour comparer les trois types:

```bash
python compare_backbones.py
```

Cela affichera:
- Accuracy de chaque backbone
- Distribution des prédictions
- Statistiques des features
- Graphiques de comparaison

## 🎓 Exemples d'utilisation

### Exemple 1: Prédictions simples

```python
from example_inference import DownstreamModelInference
import numpy as np

# MAE
mae_model = DownstreamModelInference(
    "results/downstream/CWRU_to_LASPI_backbone_mae_head_linear_finetune_False/data_ratio_0.2_epochs_50/best_model.pth"
)

# SAP
sap_model = DownstreamModelInference(
    "results/downstream/CWRU_to_LASPI_backbone_sap_head_linear_finetune_False/data_ratio_0.2_epochs_50/best_model.pth"
)

# Random
random_model = DownstreamModelInference(
    "results/downstream/None_to_LASPI_backbone_random_head_linear_finetune_True/data_ratio_1.0_epochs_50/best_model.pth"
)

# Tester sur les mêmes données
x = np.random.randn(10, 2048)

mae_results = mae_model.predict(x)
sap_results = sap_model.predict(x)
random_results = random_model.predict(x)

print("MAE:", mae_results['class_names'])
print("SAP:", sap_results['class_names'])
print("Random:", random_results['class_names'])
```

### Exemple 2: Extraire et comparer les features

```python
x = np.random.randn(100, 2048)

mae_features = mae_model.get_features(x)      # (100, hidden_dim)
sap_features = sap_model.get_features(x)      # (100, hidden_dim)
random_features = random_model.get_features(x) # (100, hidden_dim)

# Analyser
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Normaliser
scaler = StandardScaler()
mae_norm = scaler.fit_transform(mae_features)
sap_norm = scaler.fit_transform(sap_features)
random_norm = scaler.fit_transform(random_features)

# t-SNE
tsne = TSNE(n_components=2)
mae_2d = tsne.fit_transform(mae_norm)
sap_2d = tsne.fit_transform(sap_norm)
random_2d = tsne.fit_transform(random_norm)

# Visualiser les différences
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].scatter(*mae_2d.T)
axes[0].set_title("MAE features (t-SNE)")

axes[1].scatter(*sap_2d.T)
axes[1].set_title("SAP features (t-SNE)")

axes[2].scatter(*random_2d.T)
axes[2].set_title("Random features (t-SNE)")

plt.tight_layout()
plt.show()
```

### Exemple 3: Evaluer sur tout le dataset

```python
from types import SimpleNamespace
from dataset import split_data_factory
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Charger les données
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

# Évaluer chaque modèle
from sklearn.metrics import accuracy_score

for name, model in [("MAE", mae_model), ("SAP", sap_model), ("Random", random_model)]:
    all_preds = []
    all_trues = []
    
    for batch in test_loader:
        x = batch['X_raw'].numpy()
        y = batch['y_label'].numpy()
        
        preds = model.predict(x)['predictions']
        
        all_preds.extend(preds)
        all_trues.extend(y)
    
    accuracy = accuracy_score(all_trues, all_preds)
    print(f"{name}: {accuracy:.4f}")
```

## 🔍 Points importants

### 1. Chemins des modèles

**MAE:**
```
results/downstream/CWRU_to_LASPI_backbone_mae_head_linear_finetune_False/.../best_model.pth
results/downstream/CWRU_to_LASPI_backbone_mae_head_linear_finetune_True/.../best_model.pth
```

**SAP:**
```
results/downstream/CWRU_to_LASPI_backbone_sap_head_linear_finetune_False/.../best_model.pth
results/downstream/CWRU_to_LASPI_backbone_sap_head_linear_finetune_True/.../best_model.pth
```

**Random:**
```
results/downstream/None_to_CWRU_backbone_random_head_linear_finetune_True/.../best_model.pth
results/downstream/None_to_LASPI_backbone_random_head_linear_finetune_True/.../best_model.pth
```

### 2. Structure du backbone_init

- La détection se fait depuis le nom du répertoire ET depuis `config.json`
- Les trois types utilisent le même `ViT1DEncoder` (Vision Transformer 1D)
- Seuls les poids pré-entraînés diffèrent

### 3. Ordre de priorité

```
backbone_init dans config.json
    ↓
Si "mae" → Charge MAE
Si "sap" → Charge SAP
Si "random" → Backbone aléatoire
Si pretrain_dataset="None" → Backbone aléatoire
```

### 4. Gestion des erreurs

Si le checkpoint ne peut pas être chargé:
```python
# Le script affichera:
# ⚠️  Backbone random utilisé (aucun checkpoint MAE trouvé)

# Et utilisera simplement un backbone aléatoire
```

## 📈 Quand utiliser quel type?

### MAE - À utiliser si:
- Vous avez entraîné MAE sur votre dataset
- Vous voulez tirer parti du pré-entraînement auto-supervisé
- Vous avez des données limitées pour la tâche aval

### SAP - À utiliser si:
- Vous avez entraîné SAP sur votre dataset
- Vous voulez comparer avec une autre approche SSL
- Vous explorez différentes architectures de pré-entraînement

### Random - À utiliser si:
- Vous voulez une baseline (sans pré-entraînement)
- Vous mesurez le bénéfice du pré-entraînement
- Vous testez rapidement sans GPU

## ✅ Checklist de vérification

- [ ] Les fichiers `best_model.pth` existent
- [ ] Les fichiers `config.json` existent
- [ ] Les checkpoints pré-entraînés existent (MAE ou SAP)
- [ ] Vous avez les bonnes permissions d'accès
- [ ] Vous avez assez de RAM/GPU
- [ ] Les données d'entrée ont la bonne shape (n_samples, 2048)

## 🐛 Dépannage

### "Model not found"
→ Vérifier le chemin exact vers `best_model.pth`

### "Config file not found"
→ Vérifier que `config.json` existe dans le même répertoire

### "Backbone random utilisé"
→ Normal - le script a cherché un checkpoint mais ne l'a pas trouvé
→ Vérifier `results/pretrain/MAEModel/` ou `results/pretrain/SAPModel/`

### "CUDA out of memory"
→ Réduire `batch_size` dans vos scripts

### Prédictions très mauvaises
→ Vérifier que les données sont normalisées correctement (automatique)
→ Comparer avec `compare_backbones.py`

## 📚 Fichiers connexes

- `example_inference.py` - Classe wrapper principale
- `inference_downstream.py` - Script argparse complet
- `examples_inference.py` - 7 exemples concrets
- `compare_backbones.py` - **Comparaison MAE vs SAP vs Random**
- `INFERENCE_GUIDE.md` - Documentation technique
- `README_INFERENCE.md` - TL;DR rapide

---

**Besoin d'aide?** Consultez les documentations ou exécutez les scripts d'exemple!
