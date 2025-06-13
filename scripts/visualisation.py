import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Étape 1 : Charger les données
model_name = 'model_v3.1'  # Nom du modèle ou du projet
true_filepath = os.path.join('results', model_name, 'evaluation', 'fault_energy_ratio_true.csv')  # Chemin vers le fichier true
predict_filepath = os.path.join('results', model_name, 'evaluation', 'fault_energy_ratio_predict.csv')  # Chemin vers le fichier predict

# Charger les deux fichiers CSV
df_true = pd.read_csv(true_filepath)
df_predict = pd.read_csv(predict_filepath)

# Ajouter une colonne 'source' pour différencier les données
df_true['source'] = 'true'
df_predict['source'] = 'predict'

# Combiner les deux DataFrames
df = pd.concat([df_true, df_predict], ignore_index=True)

# Liste des indicateurs à visualiser
indicators = ['BPFI', 'BPFO', 'FTF', 'BSF']

# Normaliser les indicateurs entre 0 et 1
# max_values = df[indicators].max()  # Calculer le maximum pour chaque indicateur
# min_values = df[indicators].min()  # Calculer le minimum pour chaque indicateur
# df[indicators] = (df[indicators] - min_values) / (max_values - min_values)

# Obtenir les labels uniques
labels = df['label'].unique()

# === Barplot ===
fig, axs = plt.subplots(1, len(labels), figsize=(12, 10), sharey=True)  # Ajustez la taille de la figure selon vos besoins
for i, label in enumerate(labels):
    subset = df[df['label'] == label]  # Filtrer les données pour le label actuel
    
    # Transformer les données pour que les indicateurs soient sur l'axe des x
    melted_subset = subset.melt(id_vars=['label', 'source'], value_vars=indicators, var_name='Indicateur', value_name='Valeur')
    
    # Créer le barplot
    sns.barplot(data=melted_subset, x='Indicateur', y='Valeur', hue='source', ax=axs[i], palette="viridis")
    axs[i].set_title(f'Classe : {label}')
    axs[i].set_xlabel('Indicateurs')
    axs[i].set_ylabel('Valeurs normalisées')

plt.tight_layout()

# Sauvegarder le barplot
plt.savefig(os.path.join('results',model_name,'evaluation','barplot_true_predict.jpeg'))
plt.show()

# === Diagramme en araignée ===
fig, axs = plt.subplots(1, len(labels), subplot_kw=dict(polar=True), figsize=(15, 10))  # Ajustez la taille de la figure selon vos besoins

for i, label in enumerate(labels):
    subset = df[df['label'] == label]  # Filtrer les données pour le label actuel
    
    # Calculer la moyenne des indicateurs pour chaque source
    mean_true = subset[subset['source'] == 'true'][indicators].mean()
    mean_predict = subset[subset['source'] == 'predict'][indicators].mean()

    # Préparer les données pour le diagramme en araignée
    angles = np.linspace(0, 2 * np.pi, len(indicators), endpoint=False).tolist()
    values_true = mean_true.tolist()
    values_predict = mean_predict.tolist()
    values_true += values_true[:1]  # Boucler les valeurs pour fermer le diagramme
    values_predict += values_predict[:1]  # Boucler les valeurs pour fermer le diagramme
    angles += angles[:1]  # Boucler les angles pour fermer le diagramme

    # Créer le diagramme en araignée
    axs[i].plot(angles, values_true, label='True', linewidth=2, color='blue')
    axs[i].fill(angles, values_true, alpha=0.25, color='blue')
    axs[i].plot(angles, values_predict, label='Predict', linewidth=2, color='orange')
    axs[i].fill(angles, values_predict, alpha=0.25, color='orange')
    axs[i].set_yticks([np.amax(df[indicators].mean())])  # Ajustez les ticks selon vos besoins
    axs[i].set_xticks(angles[:-1])
    axs[i].set_xticklabels(indicators)
    axs[i].set_title(f'Classe : {label}')
    axs[i].legend(loc='upper right')

plt.tight_layout()

# Sauvegarder le diagramme en araignée
plt.savefig(os.path.join('results',model_name,'evaluation','radar_chart_true_predict.jpeg'))
