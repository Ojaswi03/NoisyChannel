#!/bin/bash

set -e  # Exit on any error

echo "⚠️  Starting uninstallation process for PyTorch environment and related tools..."

# Remove Python packages
echo "🧹 Uninstalling Python packages..."
pip3 uninstall -y torch torchvision torchaudio numpy matplotlib scikit-learn

# Optional: Remove datasets
read -p "🗑️  Do you want to remove the downloaded datasets (./data directory)? (y/n): " remove_data
if [[ "$remove_data" == "y" || "$remove_data" == "Y" ]]; then
    rm -rf ./data
    echo "✅ Datasets removed."
fi

# Remove Intel OpenCL/Vulkan tools
echo "🧹 Removing Intel GPU tools (OpenCL/Vulkan)..."
sudo apt remove --purge -y intel-opencl-icd intel-media-va-driver-non-free clinfo vulkan-tools
sudo apt autoremove -y

# Final cleanup
echo "🧹 Final system cleanup..."
sudo apt clean

echo "✅ Uninstallation complete. Your system is back to its pre-PyTorch zen state."
