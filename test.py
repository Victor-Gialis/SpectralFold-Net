import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les données
spectrum_correlation = pd.read_csv('results/model_v1/spectrum_correlation.csv',index_col=0)

# Vérifier les premières lignes du fichier
print(spectrum_correlation.head())

# Tracer le boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=spectrum_correlation, x='class', y='correlation_coefficient', palette='viridis')

# Ajouter des labels et un titre
plt.title('Boxplot des coefficients de corrélation par classe de défaut')
plt.xlabel('Classe de défaut')
plt.ylabel('Coefficient de corrélation')

# Afficher le graphique
plt.savefig('spectral_correlation.jpeg')