#!/bin/bash
# Setup script for ACADOS MPC

echo "=========================================="
echo "ACADOS MPC Setup Script"
echo "=========================================="
echo ""

# Check if ACADOS is installed
if [ -z "$ACADOS_SOURCE_DIR" ]; then
    echo "ERROR: ACADOS_SOURCE_DIR environment variable is not set!"
    echo ""
    echo "Please install ACADOS first:"
    echo "  cd ~"
    echo "  git clone https://github.com/acados/acados.git"
    echo "  cd acados"
    echo "  git submodule update --recursive --init"
    echo "  mkdir -p build && cd build"
    echo "  cmake -DACADOS_WITH_QPOASES=ON .."
    echo "  make install -j4"
    echo ""
    echo "Then set the environment variable:"
    echo "  export ACADOS_SOURCE_DIR=\"\$HOME/acados\""
    echo "  echo 'export ACADOS_SOURCE_DIR=\"\$HOME/acados\"' >> ~/.bashrc"
    echo "  export LD_LIBRARY_PATH=\"\$ACADOS_SOURCE_DIR/lib:\$LD_LIBRARY_PATH\""
    echo "  echo 'export LD_LIBRARY_PATH=\"\$ACADOS_SOURCE_DIR/lib:\$LD_LIBRARY_PATH\"' >> ~/.bashrc"
    echo ""
    exit 1
fi

echo "Found ACADOS at: $ACADOS_SOURCE_DIR"
echo ""

# Set library path for ACADOS
export LD_LIBRARY_PATH=$HOME/acados/lib:$LD_LIBRARY_PATH
echo "Set LD_LIBRARY_PATH to include ACADOS libraries"
echo ""

# Check Python dependencies
echo "Checking Python dependencies..."
python3 -c "import acados_template" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing acados_template Python package..."
    pip3 install -e $ACADOS_SOURCE_DIR/interfaces/acados_template
fi

python3 -c "import casadi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing casadi Python package..."
    pip3 install casadi
fi

echo "Python dependencies OK"
echo ""

# Ensure compatible tera renderer is installed
echo "Downloading compatible tera renderer..."
python3 - << EOF
from acados_template import get_tera
get_tera(tera_version='0.0.34', force_download=True)
EOF
echo "Tera renderer ready"
echo ""

# Generate ACADOS solver code
echo "Generating ACADOS solver code..."
cd "$(dirname "$0")"
python3 generate_acados_solver.py

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to generate ACADOS solver code!"
    exit 1
fi

echo ""
echo "Moving generated code to package directory..."
if [ -d "c_generated_code" ]; then
    mv c_generated_code ../
    echo "Generated code moved to ../c_generated_code/"
else
    echo "ERROR: c_generated_code directory not found!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Build your workspace:"
echo "     cd ~/jackal_ws"
echo "     catkin_make"
echo ""
echo "  2. Source the workspace:"
echo "     source devel/setup.bash"
echo ""
echo "  3. Update your launch file to use the C++ MPC node:"
echo "     Change: type=\"mpc_node.py\""
echo "     To:     type=\"mpc_node\""
echo ""
echo "  4. Run the controller:"
echo "     roslaunch jackal_helper move_base_mlda_rviz_auto.launch"
echo ""