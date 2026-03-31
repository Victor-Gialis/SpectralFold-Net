import os
from itertools import product

# Set PYTHONPATH to include current directory
pythonpath = os.getenv("PYTHONPATH", "")
if pythonpath:
    pythonpath = pythonpath + ":" + os.path.abspath(".")
else:
    pythonpath = os.path.abspath(".")
os.environ["PYTHONPATH"] = pythonpath

# Variables for pretrain experiment
pretrain_dataset = "CWRU"  # ["CWRU", "LASPI"]
batch_size = 64
window_size = 2048
window_stride = 256
mask_ratio = 0.40
epochs = 50

# =========================
# Pretraining experiments
# =========================
print(f"Starting MAE pretraining experiments...")

pretrain_cmd = (
    f"python experiments/pretrain_mae.py "
    f"--pretrain_dataset {pretrain_dataset} "
    f"--batch_size {batch_size} "
    f"--window_size {window_size} "
    f"--window_stride {window_stride} "
    f"--mask_ratio {mask_ratio} "
    f"--epochs {epochs} "
)

print(f"Running: {pretrain_cmd}")
exit_code = os.system(pretrain_cmd)

if exit_code != 0:
    print(f"❌ Command failed: {pretrain_cmd}")