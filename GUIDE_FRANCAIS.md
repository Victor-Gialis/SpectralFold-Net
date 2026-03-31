# 🎯 INFÉRENCE - Mode d'emploi complet (Français)

## TL;DR - 30 secondes

```python
from example_inference import DownstreamModelInference
import numpy as np

# Fonctionne pour MAE, SAP, et Random - détection automatique!
model = DownstreamModelInference(
    "results/downstream/CWRU_to_LASPI_backbone_mae_head_linear_finetune_False/data_ratio_0.2_epochs_50/best_model.pth"
)

x = np.random.randn(10, 2048)  # Vos données
results = model.predict(x)

print(results['class_names'])   # Prédictions
print(results['confidence'])    # Confidences
```

**C'est tout!** Le script détecte automatiquement si c'est MAE, SAP, ou Random.

---

## 📦 Fichiers disponibles

### 🔴 Essentiels (3 fichiers)

| Fichier | Utilité | Ligne de code |
|---------|---------|---------------|
| **example_inference.py** | Classe `DownstreamModelInference` | `from example_inference import DownstreamModelInference` |
| **README_INFERENCE.md** | Guide rapide (2 min) | Lire le fichier |
| **quick_start.py** | Démo interactive | `python quick_start.py` |

### 🔵 Supplémentaires (4 fichiers)

| Fichier | Utilité |
|---------|---------|
| `inference_downstream.py` | Script CLI avec argparse |
| `examples_inference.py` | 7 exemples concrets |
| `compare_backbones.py` | Comparaison MAE vs SAP vs Random |
| `BACKBONE_COMPARISON.md` | Guide détaillé des 3 types |

### 📖 Documentation (5 fichiers)

| Fichier | Contenu |
|---------|---------|
| `README_INFERENCE.md` | **👈 Commencer ici** |
| `INFERENCE_GUIDE.md` | Guide complet |
| `BACKBONE_COMPARISON.md` | Guide des 3 types |
| `TECHNICAL_NOTES.md` | Architecture technique |
| `INDEX_INFERENCE.md` | Index de navigation |

---

## 🚀 Usage par cas d'utilisation

### Cas 1: Je veux juste faire une prédiction

```python
from example_inference import DownstreamModelInference
import numpy as np

model = DownstreamModelInference("results/downstream/.../best_model.pth")
x = np.random.randn(10, 2048)
results = model.predict(x)
print(results['class_names'])
```

### Cas 2: Je veux comparer MAE, SAP, et Random

```bash
python compare_backbones.py
```

Affiche automatiquement:
- Accuracy de chaque type
- Graphiques de comparaison
- Statistiques des features

### Cas 3: Je veux utiliser la ligne de commande

```bash
python inference_downstream.py \
    --model_path results/downstream/.../best_model.pth \
    --dataset LASPI \
    --device cuda
```

### Cas 4: Je veux apprendre comment ça marche

```bash
# Lire les ressources dans cet ordre:
1. README_INFERENCE.md (2 min)
2. quick_start.py (5 min)
3. examples_inference.py (10 min)
4. BACKBONE_COMPARISON.md (15 min)
```

---

## 🎯 Les trois types de backbone

### ✅ MAE (Masked AutoEncoder)
```
Pré-entraîné avec masquage auto-supervisé
Performance: ⭐⭐⭐⭐
Checkpoint: results/pretrain/MAEModel/{dataset}/
```

**Exemple:**
```python
model = DownstreamModelInference(
    "results/downstream/CWRU_to_LASPI_backbone_mae_.../best_model.pth"
)
# Charge automatiquement les poids MAE
```

### ✅ SAP
```
Pré-entraîné avec une autre approche SSL
Performance: ⭐⭐⭐⭐⭐ (potentiellement)
Checkpoint: results/pretrain/SAPModel/{dataset}/
```

**Exemple:**
```python
model = DownstreamModelInference(
    "results/downstream/CWRU_to_LASPI_backbone_sap_.../best_model.pth"
)
# Charge automatiquement les poids SAP
```

### ✅ Random/Scratch
```
Aucun pré-entraînement (baseline)
Performance: ⭐⭐ (référence pour mesurer le bénéfice)
Checkpoint: Aucun
```

**Exemple:**
```python
model = DownstreamModelInference(
    "results/downstream/None_to_LASPI_backbone_random_.../best_model.pth"
)
# Pas de checkpoint à charger
```

---

## 💡 Fonctionnalités clés

### 1. Détection automatique
```python
# Lit config.json et détecte le type automatiquement
model = DownstreamModelInference(model_path)
```

### 2. Prédictions simples
```python
results = model.predict(x)
print(results['class_names'])      # Noms des classes
print(results['predictions'])      # Indices
print(results['probabilities'])    # Probabilités softmax
print(results['confidence'])       # Confiance maximale
```

### 3. Extraction de features
```python
features = model.get_features(x)   # (n_samples, hidden_dim)
```

### 4. GPU/CPU
```python
model = DownstreamModelInference(model_path, device="cuda")   # GPU
model = DownstreamModelInference(model_path, device="cpu")    # CPU
```

---

## 📊 Résultats retournés

```python
results = model.predict(x)

# Pour la classification:
results = {
    'predictions': np.array([0, 1, 2, ...]),          # Indices
    'class_names': ['bearing', 'pump', ...],          # Noms
    'probabilities': np.array([[0.9, 0.1], ...]),     # Softmax
    'confidence': np.array([0.9, 0.8, ...])           # Max prob
}

# Pour la régression:
results = {
    'predictions': np.array([1.5, 2.3, ...])          # Valeurs
}
```

---

## 🎓 Exemples rapides

### Exemple 1: Charger et prédire
```python
from example_inference import DownstreamModelInference
import numpy as np

model = DownstreamModelInference(
    "results/downstream/CWRU_to_LASPI_backbone_mae_head_linear_finetune_False/data_ratio_0.2_epochs_50/best_model.pth"
)

x = np.random.randn(5, 2048)
results = model.predict(x)

for cls, conf in zip(results['class_names'], results['confidence']):
    print(f"{cls}: {conf:.2%}")
```

### Exemple 2: Comparer MAE et SAP
```python
mae_model = DownstreamModelInference("...mae...pth")
sap_model = DownstreamModelInference("...sap...pth")

x = np.random.randn(10, 2048)

mae_results = mae_model.predict(x)
sap_results = sap_model.predict(x)

print("MAE:", mae_results['class_names'][:3])
print("SAP:", sap_results['class_names'][:3])
```

### Exemple 3: Extraire features pour clustering
```python
model = DownstreamModelInference(model_path)
x = np.random.randn(100, 2048)

features = model.get_features(x)  # (100, 768)

# Utiliser avec t-SNE, UMAP, clustering, etc.
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
labels = kmeans.fit_predict(features)
```

### Exemple 4: Évaluer sur tout le dataset
```python
from types import SimpleNamespace
from dataset import split_data_factory
from sklearn.metrics import accuracy_score

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

# Évaluer
model = DownstreamModelInference(model_path)
all_preds = []
all_trues = []

for batch in test_loader:
    x = batch['X_raw'].numpy()
    y = batch['y_label'].numpy()
    
    preds = model.predict(x)['predictions']
    all_preds.extend(preds)
    all_trues.extend(y)

accuracy = accuracy_score(all_trues, all_preds)
print(f"Accuracy: {accuracy:.4f}")
```

---

## 🔍 Comment ça marche (derrière les coulisses)

```
1. Charger le fichier model_path
2. Lire config.json depuis le même répertoire
3. Détecter le type: "mae", "sap", ou "random"
4. Si MAE → Charger checkpoint depuis results/pretrain/MAEModel/
5. Si SAP → Charger checkpoint depuis results/pretrain/SAPModel/
6. Si Random → Utiliser backbone aléatoire
7. Charger les poids du head depuis best_model.pth
8. Mettre en mode eval() et sur le device (GPU/CPU)
9. Prêt pour l'inférence!
```

---

## ✅ Checklist rapide

- [ ] Vous avez le fichier `best_model.pth`
- [ ] Vous avez le fichier `config.json` dans le même répertoire
- [ ] Vous avez les données d'entrée en format numpy (n_samples, 2048)
- [ ] `python -c "import torch"` fonctionne
- [ ] Vous avez assez de RAM/GPU

Si tout ✅, alors vous pouvez utiliser:

```python
from example_inference import DownstreamModelInference
model = DownstreamModelInference(model_path)
results = model.predict(x)
```

---

## 🛠️ Commandes utiles

### Lancer le quick start
```bash
cd /home/ngrotus/Desktop/CWRU_MAE
python quick_start.py
```

### Comparer les trois types
```bash
python compare_backbones.py
```

### Lancer tous les exemples
```bash
python examples_inference.py
```

### Script CLI
```bash
python inference_downstream.py \
    --model_path results/downstream/.../best_model.pth \
    --dataset LASPI
```

---

## ⚠️ Dépannage

### "Model not found"
```
✓ Vérifier que model_path est correct
✓ Vérifier que best_model.pth existe
✓ Utiliser chemins absolus
```

### "Config file not found"
```
✓ Vérifier que config.json existe dans le même répertoire
✓ Ne pas renommer les fichiers
```

### "Backbone random utilisé"
```
✓ Normal si pas de pré-entraînement
✓ Vérifier results/pretrain/MAEModel/ ou /SAPModel/
✓ Peut indiquer que le pré-entraînement n'est pas terminé
```

### "CUDA out of memory"
```
✓ Réduire batch_size
✓ Utiliser device="cpu"
✓ Réduire window_size
```

---

## 📚 Ressources

### Fichiers à lire (dans cet ordre)

1. **README_INFERENCE.md** (2 min)
   - TL;DR et démarrage rapide

2. **quick_start.py** (5 min)
   - Script de démonstration interactive

3. **examples_inference.py** (15 min)
   - 7 exemples concrets avec explications

4. **BACKBONE_COMPARISON.md** (15 min)
   - Guide complet des 3 types de backbone

5. **INFERENCE_GUIDE.md** (20 min)
   - Documentation technique complète

### Fichiers de référence

- **TECHNICAL_NOTES.md** - Architecture et détails techniques
- **INDEX_INFERENCE.md** - Index et guide de navigation
- **RECAP_INFERENCE.md** - Résumé de ce qui a été livré

---

## 🎉 Résumé

**Vous avez maintenant:**

✅ `example_inference.py` - Classe wrapper simple et puissante  
✅ `inference_downstream.py` - Script CLI complet  
✅ `examples_inference.py` - 7 exemples concrets  
✅ `compare_backbones.py` - Comparaison automatique  
✅ `quick_start.py` - Démarrage rapide  
✅ Documentation complète (5 guides)  

**Support automatique pour MAE, SAP, et Random sans configuration manuelle.**

---

## 🚀 Prochaines étapes

1. **Démarrage immédiat:**
   ```bash
   python quick_start.py
   ```

2. **Utilisation dans votre code:**
   ```python
   from example_inference import DownstreamModelInference
   model = DownstreamModelInference(model_path)
   results = model.predict(x)
   ```

3. **Comparaison des performances:**
   ```bash
   python compare_backbones.py
   ```

4. **Approfondissement:**
   - Lire `BACKBONE_COMPARISON.md`
   - Consulter `INFERENCE_GUIDE.md`

---

**Besoin d'aide?**  
Tous les fichiers contiennent des exemples et explications détaillées.  
Commencez par `README_INFERENCE.md`!

*Inférence ready! 🚀*
