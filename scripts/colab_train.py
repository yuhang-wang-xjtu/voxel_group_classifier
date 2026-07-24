"""
Colab Cell 4: Train Sonata with specified masking mode.
Runs in conda env via conda run.
"""
import subprocess, sys, os

ENV_NAME = "pointcept"
CONFIG = sys.argv[1] if len(sys.argv) > 1 else "configs/sonata/pretrain-sonata-v1m2-cosine-smoketest.py"
EPOCHS = sys.argv[2] if len(sys.argv) > 2 else "10"
BASE = "/content/voxel_group_classifier"

# Parse optional args
masking_mode = "cosine"
for i, a in enumerate(sys.argv[1:]):
    if a.startswith("--mode="):
        masking_mode = a.split("=")[1]
        if masking_mode == "random":
            CONFIG = "configs/sonata/pretrain-sonata-v1m2-vgc-smoketest.py"
        elif masking_mode == "cosine":
            CONFIG = "configs/sonata/pretrain-sonata-v1m2-cosine-smoketest.py"
        elif masking_mode == "vgc":
            CONFIG = "configs/sonata/pretrain-sonata-v1m2-vgc-smoketest.py"
    elif a.startswith("--epochs="):
        EPOCHS = a.split("=")[1]
    elif a.startswith("--config="):
        CONFIG = a.split("=")[1]

SAVE = f"exp/{masking_mode}_{EPOCHS}ep"
print(f"Masking: {masking_mode}")
print(f"Config: {CONFIG}")
print(f"Epochs: {EPOCHS}")
print(f"Save: {SAVE}")

cmd = (
    f"cd {BASE} && "
    f"conda run -n {ENV_NAME} python tools/train.py "
    f"--config-file {CONFIG} "
    f"--options save_path={SAVE} epoch={EPOCHS} "
    f"data.train.datasets.0.data_root={BASE}/data/s3dis enable_wandb=False "
    f"2>&1"
)
subprocess.run(cmd, shell=True)
