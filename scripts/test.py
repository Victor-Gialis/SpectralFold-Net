import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# Constantes globales
SCRATCH = 'scratch'
PRETRAIN = 'pretrain'
FROZEN = 'frozen'
FINETUNE = 'finetune'

filepath = 'results/downstream/data_scarcity/CWRU_dataset_model_v3.2_backbone/results.csv'
df = pd.read_csv(filepath)
labeled_percentage = list(df['Labeled Percentage'].unique())
labeled_percentage.sort()

plt.figure(figsize=(10, 6))

for init_type in [SCRATCH, PRETRAIN]:
    for downstream in [FROZEN, FINETUNE]:
        frame = df.loc[(df['Init Type'] == init_type) & (df['Downstream'] == downstream)]
        
        x = list()
        y = list()
        e = list()
        
        for lp in labeled_percentage :
            subset = frame[frame['Labeled Percentage'] == lp]
            mean_f1 = subset['Test F1 Score'].mean()
            std_f1 = subset['Test F1 Score'].std()

            x.append(lp)
            y.append(mean_f1)
            e.append(std_f1)

        if init_type == SCRATCH and downstream == FROZEN :
            color = 'orange'
        
        elif init_type == SCRATCH and downstream == FINETUNE :
            color = 'blue'
        
        elif init_type == PRETRAIN and downstream == FROZEN :
            color = 'green'
        
        elif init_type == PRETRAIN and downstream == FINETUNE :
            color = 'red'

        plt.plot(x, y, marker='o', color= color, label=f'{init_type} + {downstream}')
        plt.fill_between(x, [y_i - e_i for y_i, e_i in zip(y, e)], [y_i + e_i for y_i, e_i in zip(y, e)], color=color, alpha=0.2)
        plt.errorbar(x, y, yerr=e, fmt='o', color=color, capsize=5)

plt.xlabel('Labeled Percentage')
plt.ylabel('Test F1 Score')
plt.title('Downstream Task Performance')
plt.legend()
plt.grid(True)
plt.savefig(f'performance.png')