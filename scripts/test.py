import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

def plot_results(csv_path):
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)

    # Regrouper les résultats par configuration et pourcentage
    df_grouped = (
        df.groupby(["pretrain", "frozen", "labeled_percentage"])
          .agg(mean_f1=("F1_score", "mean"),
               std_f1=("F1_score", "std"),
               n=("F1_score", "count"))
          .reset_index()
    )

    plt.figure(figsize=(10, 6))

    # Tracer une courbe par combinaison pretrain/frozen
    for (pretrain, finetune), sub_df in df_grouped.groupby(["pretrain", "frozen"]):
        if not pretrain and not finetune :
            color = 'orange'

        elif not pretrain and finetune :
            color = 'blue'

        elif pretrain and not finetune :
            color = 'green'

        elif pretrain and finetune :
            color = 'red'

        label = f"Pretrain={pretrain}, Finetune={finetune}"
        plt.errorbar(
            sub_df["labeled_percentage"],
            sub_df["mean_f1"],
            yerr=sub_df["std_f1"],
            label=label,
            marker='o',
            capsize=3,
            color=color
        )
        plt.fill_between(sub_df["labeled_percentage"], 
                         [y_i - e_i for y_i, e_i in zip(sub_df["mean_f1"], sub_df["std_f1"])], 
                         [y_i + e_i for y_i, e_i in zip(sub_df["mean_f1"], sub_df["std_f1"])], 
                         color=color, 
                         alpha=0.2)


    plt.title("F1 Score (moyenne sur les seeds) en fonction du pourcentage de données étiquetées")
    plt.xlabel("Pourcentage de X_train utilisé pour l'entraînement")
    plt.ylabel("Test F1 Score moyen")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Sauvegarder le graphique
    fig_path = Path(csv_path).with_suffix('.png')
    plt.savefig(fig_path)
    plt.close()

if __name__ == "__main__" :
    # Fichier pour sauvegarder les résultats
    results_file = 'file_results.csv'
    plot_results(results_file)

# # Constantes globales
# SCRATCH = 'scratch'
# PRETRAIN = 'pretrain'
# FROZEN = 'frozen'
# FINETUNE = 'finetune'

# filepath = 'results/downstream/data_scarcity/CWRU_dataset_model_v3.2_backbone/results.csv'
# df = pd.read_csv(filepath)
# labeled_percentage = list(df['Labeled Percentage'].unique())
# labeled_percentage.sort()

# plt.figure(figsize=(10, 6))

# for init_type in [SCRATCH, PRETRAIN]:
#     for downstream in [FROZEN, FINETUNE]:
#         frame = df.loc[(df['Init Type'] == init_type) & (df['Downstream'] == downstream)]
        
#         x = list()
#         y = list()
#         e = list()
        
#         for lp in labeled_percentage :
#             subset = frame[frame['Labeled Percentage'] == lp]
#             mean_f1 = subset['Test F1 Score'].mean()
#             std_f1 = subset['Test F1 Score'].std()

#             x.append(lp)
#             y.append(mean_f1)
#             e.append(std_f1)

#         if init_type == SCRATCH and downstream == FROZEN :
#             color = 'orange'
        
#         elif init_type == SCRATCH and downstream == FINETUNE :
#             color = 'blue'
        
#         elif init_type == PRETRAIN and downstream == FROZEN :
#             color = 'green'
        
#         elif init_type == PRETRAIN and downstream == FINETUNE :
#             color = 'red'

#         plt.plot(x, y, marker='o', color= color, label=f'{init_type} + {downstream}')
#         plt.fill_between(x, [y_i - e_i for y_i, e_i in zip(y, e)], [y_i + e_i for y_i, e_i in zip(y, e)], color=color, alpha=0.2)
#         plt.errorbar(x, y, yerr=e, fmt='o', color=color, capsize=5)

# plt.xlabel('Labeled Percentage')
# plt.ylabel('Test F1 Score')
# plt.title('Downstream Task Performance')
# plt.legend()
# plt.grid(True)
# plt.savefig(f'performance.png')