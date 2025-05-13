#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

echo "Setting up environment for PyTorch with MNIST and CIFAR-10 datasets"

# System update
echo " Updating system packages..."
sudo apt update && sudo apt upgrade -y

# creating an environment
python3 -m venv venv
source  venv/bin/activate

# Essential tools
echo "Installing essential packages..."
sudo apt install -y python3-pip python3-venv git

# Upgrade pip
echo "Upgrading pip..."
pip3 install --upgrade pip

# Check for CUDA
echo "⚡ Checking for CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    echo " CUDA detected! Installing PyTorch with CUDA support..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "CUDA not detected. Installing CPU-only PyTorch..."
    pip3 install torch torchvision torchaudio
fi

# Install core Python packages
echo " Installing core Python packages..."
pip3 install numpy matplotlib scikit-learn

echo "Setup complete! Ready to roll with PyTorch and datasets."
