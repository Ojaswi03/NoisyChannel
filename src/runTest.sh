#!/bin/bash

# Make sure the script exits if any command fails
set -e


#!/bin/bash

echo "==================================="
echo "Starting Full Parameter Sweep Test"
echo "==================================="

lrs=(0.001 0.005 0.01)
sigmas=(0.01 0.02 0.05)
updates=(1 3 5)

for lr in "${lrs[@]}"; do
  for sigma in "${sigmas[@]}"; do
    for update in "${updates[@]}"; do

      echo "-----------------------------------"
      echo "Running with: LR=$lr, Sigma=$sigma, Inner Updates=$update"
      echo "-----------------------------------"

      mkdir -p "diagram-EBM/lr_${lr}_sigma_${sigma}_updates_${update}"

      python3 EBM.py <<EOF
mnist
32
64
$sigma
$lr
10
$update
EOF

      # Move the results into a uniquely named subfolder
      mv diagram-EBM/accuracy_comparison.png "diagram-EBM/lr_${lr}_sigma_${sigma}_updates_${update}/"
      mv diagram-EBM/loss_comparison.png "diagram-EBM/lr_${lr}_sigma_${sigma}_updates_${update}/"

    done
  done
done

echo "==================================="
echo "All parameter sweeps completed!"
echo "Check diagram-EBM/ subfolders for results!"
echo "==================================="



# echo "==================================="
# echo " Starting WCM Test Run "
# echo "==================================="

# # Running WCM (Worst-Case Model)
# python3 WCM.py << EOF
# cifar10
# 30
# 64
# 2.5
# 0.01
# 10
# EOF

# echo "✅ WCM test completed."
# echo ""

echo "🎯 All tests done. Check diagram folders for plots!"
