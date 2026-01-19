# from idr_pytools import gpu_jobs_submitter

# ssl_methods = ["random", "mae", "sap"]
# ratios = [0.01, 0.05, 0.1, 0.5, 1.0]
# seeds = [0, 1, 2]

# jobs = []

# for ssl in ssl_methods:
#     for ratio in ratios:
#         for seed in seeds:
#             cmd = (
#                 f"python experiments/data_scarcity.py "
#                 f"--ssl_method {ssl} "
#                 f"--data_ratio {ratio} "
#                 f"--seed {seed}"
#             )
#             jobs.append(cmd)

# gpu_jobs_submitter(
#     jobs,
#     job_name="data_scarcity_ssl",
#     gpus=1,
#     cpus=10,
#     mem="40G",
#     time="02:00:00",
# )

import os
from itertools import product

# =========================
# Experimental space
# =========================
ssl_methods = ["random", "mae", "sap"]
downstream_datasets = ["CWRU", "LASPI"]
downstream_tasks = ["classification"]
downstream_head_types = ["linear", "non-linear"]

ratios = [0.01]          # debug
seeds = [0]              # debug
finetunes = [True, False]
epochs = 1

# =========================
# Cartesian product
# =========================
for (
    ssl,
    dataset,
    task,
    head_type,
    ratio,
    seed,
    finetune,
) in product(
    ssl_methods,
    downstream_datasets,
    downstream_tasks,
    downstream_head_types,
    ratios,
    seeds,
    finetunes,
):

    cmd = (
        f"python experiments/data_scarcity.py "
        f"--ssl_method {ssl} "
        f"--dataset {dataset} "
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

