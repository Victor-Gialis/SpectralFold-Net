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
pretrain_dataset = "CWRU"  # ["CWRU", "LASPI"]
downstream_dataset = "LASPI"  # ["CWRU", "LASPI"]
data_ratio = 0.01  # [0.01, 0.1, 0.5, 1.0]
split_type = "speed_load_stratified"  # ["independent", "speed_stratified", "speed_load_stratified", "sample_stratified"]
finetune_option = True # [True, False]
head_type = "linear"  # ["linear", "nonlinear"]
seed = 8  # [0, 1, 2, 3, 4]  
epoch = 5

# =========================
# Downstream experiments
# =========================
print(f"Starting downstream evaluation with Random init backbone...")

downstream_cmd = (
    f"python experiments/downstream_random.py "
    f"--pretrain_dataset {pretrain_dataset} "
    f"--downstream_dataset {downstream_dataset} "
    f"--batch_size 128 "
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

if finetune_option:
    downstream_cmd += " --finetune"

print(f"Running: {downstream_cmd}")
exit_code = os.system(downstream_cmd)

if exit_code != 0:
    print(f"❌ Command failed: {downstream_cmd}")
