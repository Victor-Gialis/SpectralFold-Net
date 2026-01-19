import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. Load results
# =========================
csv_path = "results/downstream/results_summary.csv"   # adapte le chemin si besoin
df = pd.read_csv(csv_path)

# Sécurité sur les types
df["finetune"] = df["finetune"].astype(bool)
df["data_ratio"] = df["data_ratio"].astype(float)

# =========================
# 2. Aggregate statistics
# =========================
grouped = (
    df
    .groupby(["ssl_method", "finetune", "data_ratio"])
    .agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
    )
    .reset_index()
)

# =========================
# 3. Accuracy vs data_ratio
#    (1 figure per SSL method)
# =========================
ssl_methods = grouped["ssl_method"].unique()

for ssl in ssl_methods:
    plt.figure(figsize=(6, 4))

    for finetune in [False, True]:
        sub = grouped[
            (grouped["ssl_method"] == ssl) &
            (grouped["finetune"] == finetune)
        ]

        label = "Fine-tuning" if finetune else "Linear probing"

        plt.errorbar(
            sub["data_ratio"],
            sub["accuracy_mean"],
            yerr=sub["accuracy_std"],
            marker="o",
            capsize=4,
            label=label
        )

    plt.xscale("log")
    plt.xlabel("Data ratio")
    plt.ylabel("Accuracy")
    plt.title(f"{ssl.upper()} – Downstream accuracy (CWRU)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"accuracy_{finetune}.png")

# =========================
# 4. F1-score vs data_ratio
# =========================
for ssl in ssl_methods:
    plt.figure(figsize=(6, 4))

    for finetune in [False, True]:
        sub = grouped[
            (grouped["ssl_method"] == ssl) &
            (grouped["finetune"] == finetune)
        ]

        label = "Fine-tuning" if finetune else "Linear probing"

        plt.errorbar(
            sub["data_ratio"],
            sub["f1_mean"],
            yerr=sub["f1_std"],
            marker="o",
            capsize=4,
            label=label
        )

    plt.xscale("log")
    plt.xlabel("Data ratio")
    plt.ylabel("F1-score")
    plt.title(f"{ssl.upper()} – Downstream F1-score (CWRU)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"f1_score_{finetune}.png")

# =========================
# 5. Bar plot at fixed data ratio
# =========================
fixed_ratio = 0.01   # change si besoin

sub = grouped[grouped["data_ratio"] == fixed_ratio]

plt.figure(figsize=(7, 4))

x_labels = []
y_means = []
y_stds = []

for ssl in ssl_methods:
    for finetune in [False, True]:
        row = sub[
            (sub["ssl_method"] == ssl) &
            (sub["finetune"] == finetune)
        ]

        if len(row) == 0:
            continue

        mode = "FT" if finetune else "LP"
        x_labels.append(f"{ssl.upper()}-{mode}")
        y_means.append(row["accuracy_mean"].values[0])
        y_stds.append(row["accuracy_std"].values[0])

plt.bar(x_labels, y_means, yerr=y_stds, capsize=4)
plt.ylabel("Accuracy")
plt.title(f"Downstream accuracy @ data ratio = {fixed_ratio}")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("bar plot")
