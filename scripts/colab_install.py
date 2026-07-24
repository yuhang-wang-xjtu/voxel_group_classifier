"""
Colab Cell 2: Install all dependencies in the conda 'pointcept' environment.
This script runs INSIDE the conda env (via 'conda run -n pointcept').
All pip/compile commands use the current Python (Python 3.10).
"""
import subprocess, os

def run(cmd, **kwargs):
    print(f"  > {cmd[:80]}{'...' if len(cmd) > 80 else ''}")
    return subprocess.run(cmd, shell=True, **kwargs)

# Go to repo root
os.chdir("/content/voxel_group_classifier")

# Install PyTorch 2.5.0 with CUDA 12.4
print("Installing PyTorch 2.5.0 + CUDA 12.4...")
run("pip install -q torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124")

# Install spconv (prebuilt wheel, NO JIT compilation, no cumm issues)
print("Installing spconv-cu124...")
run("pip install -q spconv-cu124")

# Install other Python deps
print("Installing Python dependencies...")
run("pip install -q ninja h5py addict pyyaml tensorboard timm peft tqdm")
run("pip install -q torch-cluster torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html")

# Build dependencies for CUDA extensions
print("Installing build tools...")
run("pip install -q setuptools wheel")
# Point CUDA to system CUDA toolkit (Colab has /usr/local/cuda)
os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")

# Compile CUDA extensions
print("Compiling CUDA extensions (pointops, pointops2, pointgroup_ops)...")
for lib in ["pointops", "pointops2", "pointgroup_ops"]:
    print(f"  {lib}...")
    run(f"CUDA_HOME=/usr/local/cuda pip install -e ./libs/{lib}")

# Verify
print("\nVerification:")
run("python -c \"import torch, spconv; print(f'torch={torch.__version__} spconv={spconv.__version__} cuda={torch.version.cuda}')\"")
print("Done.")
