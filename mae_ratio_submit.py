import os
from itertools import product

# Set PYTHONPATH to include current directory
pythonpath = os.getenv("PYTHONPATH", "")
if pythonpath:
    pythonpath = pythonpath + ":" + os.path.abspath(".")
else:
    pythonpath = os.path.abspath(".")
os.environ["PYTHONPATH"] = pythonpath

# =========================
# Pretraining experiments
# =========================
# for mask_ratio in [0.15]:
for mask_ratio in [0.75]:
    print(f"Starting pretraining with mask_ratio={mask_ratio}")
    
    pretrain_cmd = (
        f"python experiments/pretrain_mae.py "
        f"--epochs 50 "
        f"--mask_ratio {mask_ratio} "
    )

    print(f"Running: {pretrain_cmd}")
    exit_code = os.system(pretrain_cmd)

    if exit_code != 0:
        print(f"❌ Command failed: {pretrain_cmd}")

    # =========================
    # Downstream experiments
    # =========================
    print(f"Starting downstream evaluation with mask_ratio={mask_ratio}")

    downstream_cmd = (
        f"python experiments/downstream_mae.py "
    )

    print(f"Running: {downstream_cmd}")
    exit_code = os.system(downstream_cmd)

    if exit_code != 0:
        print(f"❌ Command failed: {downstream_cmd}")
