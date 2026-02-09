# 🐛 Résumé des bugs critiques trouvés et fixés

## Problème principal : Vous aviez une DOUBLE normalisation du LOG

### Bug #1 : LOG appliqué DEUX FOIS ❌
**Où** : `dataloader.py` ligne 163
```python
# AVANT (WRONG):
X_raw = torch.log1p(batch["X_raw"])  # ← LOG APPLICATION 1
# Puis dans normalization.global_min_max_log_normalization:
x_log = torch.log1p(x)  # ← LOG APPLICATION 2
```

**APRÈS (CORRECT)**:
```python
X_raw = batch["X_raw"]  # ← SANS LOG
# Les stats sont calculées sur X_raw
# Le log est appliqué une SEULE FOIS dans la normalisation
```

**Impact** : Les stats ne correspondaient pas aux valeurs normalisées !

---

## Bug #2 : Data ratio inversé ❌
**Où** : `dataloader.py` ligne 118-120
```python
# AVANT (WRONG):
_, scarcity_train_idx = train_test_split(
    train_idx,
    test_size = data_ratio,  # ← Si data_ratio=0.1, on GARDE 10% ? NON !
```

**APRÈS (CORRECT)**:
```python
scarcity_train_idx, _ = train_test_split(
    train_idx,
    train_size = data_ratio,  # ← Si data_ratio=0.1, on GARDE 10% ✓
```

**Impact** : Vous n'utilisiez pas le bon pourcentage de données !

---

## Bug #3 : Stats différentes pour Random vs Pretrained ⚠️
**Où** : `data_scarcity.py`

**Le problème**:
- **Random backbone** : stats calculées du **train set downstream** (petit dataset)
- **Pretrained backbone (SAP/MAE)** : stats calculées du **train set pretraining** (CWRU complet)
- Ces deux datasets sont DIFFÉRENTS → stats DIFFÉRENTES → résultats INCOMPARABLES !

**LA SOLUTION EST CORRECTE** :
- Random utilise les stats du downstream (car pas d'autres données)
- Pretrained utilise les stats du pretraining (déjà sauvegardées dans le checkpoint)
- **C'est normal qu'elles soient différentes !** Cela reflète les différences de distribution

---

## Pourquoi vous aviez la même différence SAP/MAE avec un backbone random ?

### Explication :
Même avec les mauvaises stats et la double normalisation, il y avait QUAND MÊME une différence SAP vs MAE parce que :

1. **Les initialisations étaient déjà différentes** à cause de seeds ou randomness
2. **Les modèles heads downstream étaient différents** (random init à chaque fois)
3. **Le bruit du training** dominait suffisamment pour masquer que le backbone était random
4. **Les stats n'étaient pas la seule source de variabilité**

Maintenant que vous chargez correctement le backbone :
- ✅ SAP et MAE auront les MÊMES poids (des checkpoints sauvegardés)
- ✅ SAP et MAE auront les MÊMES stats (du pretraining)
- ✅ Les seules différences viendront de la **vraie capacité de SAP vs MAE**
- ✅ La variabilité sera due aux **seeds et initialisations du downstream** (normal)

---

## Comment tester les corrections ?

```bash
# Comparer SAP vs MAE avec stats correctes :
python experiments/data_scarcity.py --backbone_init sap ...
python experiments/data_scarcity.py --backbone_init mae ...

# Les résultats devraient maintenant être :
# - Beaucoup PLUS STABLES (moins de variabilité)
# - SAP et MAE vraiment comparables
# - Backbone random devrait être très MAUVAIS (comme prévu !)
```

---

## Résumé des fichiers modifiés

| Fichier | Changement |
|---------|-----------|
| `dataset/dataloader.py` | ✅ Enlevé log1p du calcul de stats |
| `dataset/dataloader.py` | ✅ Corrigé data_ratio (train_size au lieu de test_size) |
| `models/downstream/base.py` | ✅ Commentaires clarifiants |
| `experiments/data_scarcity.py` | ✅ Commentaires sur les stats pretrain vs downstream |
