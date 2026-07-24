"""
Colab Cell 1: Set up condacolab, create conda environment.
Run this first, then restart the runtime when prompted.
"""
import subprocess, os

# Install condacolab
subprocess.run(["pip", "install", "-q", "condacolab"])

import condacolab
condacolab.install()  # triggers restart - after this, conda is available
