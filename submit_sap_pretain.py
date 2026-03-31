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
downsampling_factor = 2
epochs = 50

# =========================
# Pretraining experiments
# =========================
print(f"Starting SAP pretraining experiments...")

pretrain_cmd = (
    f"python experiments/pretrain_sap.py "
    f"--pretrain_dataset {pretrain_dataset} "
    f"--batch_size {batch_size} "
    f"--window_size {window_size} "
    f"--window_stride {window_stride} "
    f"--downsampling_factor {downsampling_factor} "
    f"--epochs {epochs} "
)

print(f"Running: {pretrain_cmd}")
exit_code = os.system(pretrain_cmd)

if exit_code != 0:
    print(f"❌ Command failed: {pretrain_cmd}")