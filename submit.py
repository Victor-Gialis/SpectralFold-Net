import os
from itertools import product

# Set PYTHONPATH to include current directory
pythonpath = os.getenv("PYTHONPATH", "")
if pythonpath:
    pythonpath = pythonpath + ":" + os.path.abspath(".")
else:
    pythonpath = os.path.abspath(".")
os.environ["PYTHONPATH"] = pythonpath

# Number of training epochs for all experiments
epochs = 100

# =========================
# Pretraining experiments
# =========================
ssl_methods = ["mae", "sap"]

for ssl in ssl_methods:

    cmd = (
        f"python experiments/pretrain_{ssl}.py "
        f"--epochs {epochs} "
    )

    print(f"Running: {cmd}")
    exit_code = os.system(cmd)

    if exit_code != 0:
        print(f"❌ Command failed: {cmd}")

# =========================
# Data scarcity experiments
# =========================
backbone_inits = ["random", "mae", "sap"]
pretrain_datasets = ["CWRU"]
downstream_datasets = ["CWRU", "LASPI"]
downstream_tasks = ["classification"]
downstream_head_types = ["linear"]

ratios = [0.01] # train set ratios
seeds = [0] # random seeds  
finetunes = [True, False] 

for (
    backbone_init,
    pretrain_dataset,
    downstream_dataset,
    task,
    head_type,
    ratio,
    seed,
    finetune,
) in product(
    backbone_inits,
    pretrain_datasets,
    downstream_datasets,
    downstream_tasks,
    downstream_head_types,
    ratios,
    seeds,
    finetunes,
):

    cmd = (
        f"python experiments/data_scarcity.py "
        f"--backbone_init {backbone_init} "
        f"--pretrain_dataset {pretrain_dataset} "
        f"--downstream_dataset {downstream_dataset} "
        f"--task {task} "
        f"--head_type {head_type} "
        f"--seed {seed} "
        f"--data_ratio {ratio} "
        f"--epochs {epochs} "
    )

    # Flag finetune (IMPORTANT)
    if finetune:
        cmd += "--finetune "

    print(f"Running: {cmd}")
    exit_code = os.system(cmd)

    if exit_code != 0:
        print(f"❌ Command failed: {cmd}")