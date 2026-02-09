import os
from itertools import product

# Set PYTHONPATH to include current directory
pythonpath = os.getenv("PYTHONPATH", "")
if pythonpath:
    pythonpath = pythonpath + ":" + os.path.abspath(".")
else:
    pythonpath = os.path.abspath(".")
os.environ["PYTHONPATH"] = pythonpath

for (ssl, epochs) in product(["sap"],[50]):

    cmd = (
        f"python experiments/pretrain_{ssl}.py "
        f"--epochs {epochs} "
    )

    print(f"Running: {cmd}")
    exit_code = os.system(cmd)

    if exit_code != 0:
        print(f"❌ Command failed: {cmd}")
