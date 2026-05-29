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

# Require Python >=3.8 (acados_template will not install on 3.6 / 3.7)
PY_OK=$(python3 -c 'import sys; print(1 if sys.version_info >= (3, 8) else 0)')
if [ "$PY_OK" != "1" ]; then
    PY_VER=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    echo "ERROR: Python 3.8+ is required for acados_template. Detected: $PY_VER"
    echo ""
    echo "Source your Python 3.8 venv first:"
    echo "  source ~/acados_env/bin/activate"
    echo ""
    echo "See README.md \"For Ubuntu 18.04\" for one-time setup."
    exit 1
fi

# Fail fast if the generated-code destination is already populated, before any
# pip install / network calls run.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -d "$SCRIPT_DIR/../c_generated_code" ]; then
    echo "ERROR: $SCRIPT_DIR/../c_generated_code already exists."
    echo "       Delete or rename it first if you want to regenerate, e.g.:"
    echo "         rm -rf $SCRIPT_DIR/../c_generated_code"
    echo "       (Aborting before regeneration to avoid silently destroying existing code.)"
    exit 1
fi

# Set library path for ACADOS
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH"
echo "Set LD_LIBRARY_PATH to include ACADOS libraries"
echo ""

# Check Python dependencies
echo "Checking Python dependencies..."
if ! python3 -c "import acados_template" 2>/dev/null; then
    echo "Installing acados_template Python package..."
    if ! pip3 install -e "$ACADOS_SOURCE_DIR/interfaces/acados_template"; then
        echo "ERROR: pip install of acados_template failed. Aborting." >&2
        exit 1
    fi
    if ! python3 -c "import acados_template" 2>/dev/null; then
        echo "ERROR: acados_template installed but still cannot be imported. Aborting." >&2
        exit 1
    fi
fi

if ! python3 -c "import casadi" 2>/dev/null; then
    echo "Installing casadi Python package..."
    if ! pip3 install casadi; then
        echo "ERROR: pip install of casadi failed. Aborting." >&2
        exit 1
    fi
    if ! python3 -c "import casadi" 2>/dev/null; then
        echo "ERROR: casadi installed but still cannot be imported. Aborting." >&2
        exit 1
    fi
fi

echo "Python dependencies OK"
echo ""

# Ensure compatible tera renderer is installed
echo "Downloading compatible tera renderer..."
if ! python3 - << EOF
from acados_template import get_tera
get_tera(tera_version='0.0.34', force_download=True)
EOF
then
    echo "ERROR: Tera renderer download failed. Aborting." >&2
    exit 1
fi
echo "Tera renderer ready"
echo ""

# Generate ACADOS solver code
echo "Generating ACADOS solver code..."
cd "$SCRIPT_DIR"

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
echo "  3. Run the controller:"
echo "     roslaunch teamrobo2026 move_base_mlda_2026.launch"
echo ""