import os
from itertools import product

for (ssl, epochs) in product(["sap","mae"],[2]):

    cmd = (
        f"python experiments/pretrain_{ssl}.py "
        f"--epochs {epochs} "
    )

    print(f"Running: {cmd}")
    exit_code = os.system(cmd)

    if exit_code != 0:
        print(f"❌ Command failed: {cmd}")
