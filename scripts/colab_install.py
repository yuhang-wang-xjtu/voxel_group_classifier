"""
Colab Cell 2: Install all dependencies in the conda 'pointcept' environment.
Run AFTER restart (conda is available after condacolab.install()).
"""
import subprocess, os, sys

ENV = "pointcept"
REPO = "/content/voxel_group_classifier"

def conda_run(cmd, **kwargs):
    full_cmd = f"conda run -n {ENV} {cmd}"
    return subprocess.run(full_cmd, shell=True, **kwargs)

# Clone repo if not exists
if not os.path.isdir(REPO):
    subprocess.run(f"git clone --depth 1 https://github.com/yuhang-wang-xjtu/voxel_group_classifier.git {REPO}", shell=True)
os.chdir(REPO)

# Create conda env with Python 3.10
print("Creating conda env: pointcept (python=3.10)...")
subprocess.run(f"conda create -n {ENV} python=3.10 -y", shell=True)

# Install PyTorch 2.5.0 with CUDA 12.4
print("Installing PyTorch 2.5.0 + CUDA 12.4...")
conda_run("pip install -q torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124")

# Install spconv (prebuilt wheel, no JIT compilation, no cumm issues)
print("Installing spconv-cu124...")
conda_run("pip install -q spconv-cu124")

# Install other Python deps
print("Installing Python dependencies...")
conda_run("pip install -q ninja h5py addict pyyaml tensorboard timm peft tqdm")
conda_run("pip install -q torch-cluster torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html")

# Compile CUDA extensions (one-time)
print("Compiling CUDA extensions (pointops, pointops2, pointgroup_ops)...")
for lib in ["pointops", "pointops2", "pointgroup_ops"]:
    print(f"  {lib}...")
    conda_run(f"pip install -q ./libs/{lib} --no-build-isolation")

# Verify
print("\nVerification:")
conda_run("python -c \"import torch, spconv; print(f'torch={torch.__version__} spconv={spconv.__version__} cuda={torch.version.cuda}')\"")

print("\nDone. Dependencies installed in conda env 'pointcept'.")
