#!/bin/bash

# Make sure the script exits if any command fails
set -e

# echo "==================================="
# echo " Starting EBM Test Run "
# echo "==================================="

# # Running EBM (Expectation-Based Model)
# python3 EBM.py << EOF
# mnist
# 35
# 64
# 0.1
# 0.05
# 10
# EOF

# echo "✅ EBM test completed."
# echo ""

echo "==================================="
echo " Starting WCM Test Run "
echo "==================================="

# Running WCM (Worst-Case Model)
python3 WCM.py << EOF
cifar10
35
64
2.5
0.05
10
EOF

echo "✅ WCM test completed."
echo ""

echo "🎯 All tests done. Check diagram folders for plots!"
