"""
Colab Cell 4: Train Sonata with specified masking mode.
Runs INSIDE conda env 'pointcept' (via 'conda run -n pointcept').
"""
import subprocess, sys

BASE = "/content/voxel_group_classifier"

# Parse args
masking_mode = "cosine"
epochs = "10"
config = None

for a in sys.argv[1:]:
    if a.startswith("--mode="):
        masking_mode = a.split("=")[1]
    elif a.startswith("--epochs="):
        epochs = a.split("=")[1]
    elif a.startswith("--config="):
        config = a.split("=")[1]

if config is None:
    if masking_mode == "cosine":
        config = "configs/sonata/pretrain-sonata-v1m2-cosine-smoketest.py"
    else:
        config = "configs/sonata/pretrain-sonata-v1m2-vgc-smoketest.py"

save = f"exp/{masking_mode}_{epochs}ep"
print(f"Masking: {masking_mode}")
print(f"Config:  {config}")
print(f"Epochs:  {epochs}")
print(f"Save:    {save}")

cmd = [
    "python", f"{BASE}/tools/train.py",
    "--config-file", config,
    "--options", f"save_path={save}", f"epoch={epochs}",
    f"data.train.datasets.0.data_root={BASE}/data/s3dis",
    "enable_wandb=False",
]
print(f"Running: {' '.join(cmd)}")
subprocess.run(cmd, cwd=BASE)
