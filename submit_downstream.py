import os
from itertools import product

# Set PYTHONPATH to include current directory
pythonpath = os.getenv("PYTHONPATH", "")
if pythonpath:
    pythonpath = pythonpath + ":" + os.path.abspath(".")
else:
    pythonpath = os.path.abspath(".")
os.environ["PYTHONPATH"] = pythonpath

# Variables for downstream experiment
backbone_inits = ["sap","mae"] # ["random", "mae", "sap"]
pretrain_dataset = "CWRU"  # ["CWRU", "LASPI"]
downstream_dataset = "LASPI"  # ["CWRU", "LASPI"]
data_ratios = [0.01, 0.05, 0.1, 0.2] # [0.01, 0.05, 0.1, 0.2]
split_type = "speed_load_stratified"  # ["independent", "speed_stratified", "speed_load_stratified", "sample_stratified"]
finetune_options = [True, False]
head_type = "linear"  # ["linear", "nonlinear"]
seeds = [0, 1, 2, 3, 4]  
epoch = 10

# =========================
# Downstream experiments
# =========================
for backbone, data_ratio, finetune, seed in product(backbone_inits, data_ratios, finetune_options, seeds):

    if not finetune and backbone == "random":
        print(f"Skipping random backbone without finetuning (not meaningful)")
        continue

    print(f"Starting downstream evaluation with {backbone.upper()} backbone...")
    downstream_cmd = (
        f"python experiments/downstream_{backbone}.py "
        f"--pretrain_dataset {pretrain_dataset} "
        f"--downstream_dataset {downstream_dataset} "
        f"--batch_size 256 "
        f"--window_size 2048 "
        f"--window_stride 256 "
        f"--data_ratio {data_ratio} "
        f"--split_type {split_type} "
        f"--learning_rate 0.0003695 "
        f"--weight_decay 1.1133e-5 "
        f"--epochs {epoch} "
        f"--head_type {head_type} "
        f"--task classification "
        f"--seed {seed}"
    )

    if finetune :
        downstream_cmd += " --finetune"

    print(f"Running: {downstream_cmd}")
    exit_code = os.system(downstream_cmd)

    if exit_code != 0:
        print(f"❌ Command failed: {downstream_cmd}")
