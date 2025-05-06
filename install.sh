#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

echo "Setting up environment for PyTorch with MNIST and CIFAR-10 datasets"

# System update
echo " Updating system packages..."
sudo apt update && sudo apt upgrade -y

# creating an environment
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

# # Optional: Download MNIST and CIFAR-10 (if needed)
# read -p " Do you want to download CIFAR-10 and MNIST datasets locally? (y/n): " download_data
# if [[ "$download_data" == "y" || "$download_data" == "Y" ]]; then
#     echo " Downloading datasets..."
#     python3 -c "
# import torchvision.datasets as datasets
# datasets.MNIST(root='./data', download=True)
# datasets.CIFAR10(root='./data', download=True)
# "
#     echo "✅ Datasets downloaded to ./data/"
# fi

# # Intel GPU drivers and OpenCL tools
# echo "🖥️ Setting up Intel GPU tools..."
# sudo apt install -y intel-opencl-icd intel-media-va-driver-non-free clinfo vulkan-tools

# echo "✅ Intel OpenCL and Vulkan tools installed."

# Final diagnostic checks
# echo "🔍 Running GPU diagnostics..."
# clinfo | grep 'Device' || echo "⚠️ OpenCL device not found."
# vulkaninfo | grep 'deviceName' || echo "⚠️ Vulkan device not found."

echo "Setup complete! Ready to roll with PyTorch and datasets."
