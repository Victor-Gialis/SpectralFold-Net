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
# pretrain_cmd = (
#     f"python experiments/pretrain_sap.py "
#     f"--epochs 50 "
# )

# print(f"Running: {pretrain_cmd}")
# exit_code = os.system(pretrain_cmd)

# if exit_code != 0:
#     print(f"❌ Command failed: {pretrain_cmd}")

# =========================
# Downstream experiments
# =========================
print(f"Starting downstream evaluation with SAP backbone...")

downstream_cmd = (
    f"python experiments/downstream_sap.py "
)

print(f"Running: {downstream_cmd}")
exit_code = os.system(downstream_cmd)

if exit_code != 0:
    print(f"❌ Command failed: {downstream_cmd}")
