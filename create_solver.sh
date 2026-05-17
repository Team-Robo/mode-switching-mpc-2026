export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$HOME/acados/lib"
export ACADOS_SOURCE_DIR="$HOME/acados"

# Absolute directory for script
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd "$SCRIPT_DIR/script"

source ~/env/bin/activate
python3 generate_acados_solver.py