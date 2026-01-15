# Mode Switching MPC 2026
 Note: There are some old commits in the archived 2026 MPC codebase from cpp-convert branch, do check if needed since most are based from this specific branch

## Installation
1) Clone this repo
```
cd ~/jackal_ws/src
git clone git@github.com:Team-Robo/mode-switching-mpc-2026.git
```

2) Install `acados`
```
cd ~
git clone https://github.com/acados/acados.git
cd acados
git submodule update --recursive --init
mkdir -p build && cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make install -j4
```

3) Set environmental variables
```
export ACADOS_SOURCE_DIR="$HOME/acados"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH"
echo 'export ACADOS_SOURCE_DIR="$HOME/acados"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
```

4) For Ubuntu-18.04 user, install [Python 3.8](#for-ubuntu-18-04) and source its virtual environment

5) Install python dependencies and generate acados solver (C code)
```
cd ~/jackal_ws/src/mode-switching-mpc-2026/script
./setup_acados_mpc.sh
```

6) Install ROS dependencies
```
cd ~/jackal_ws
sudo rosdep init; rosdep update --rosdistro $ROS_DISTRO # if never done before
rosdep install -y --from-paths src --ignore-src --rosdistro=$ROS_DISTRO
```

7) Build
```
cd ~/jackal_ws
catkin_make
source devel/setup.bash
```

## Usage
### Step 1: Source workspace first
```
source ~/jackal_ws/devel/setup.bash
```
### Step 2: Run MPC
Launch MPC only
```
roslaunch teamrobo2026 move_base_mlda_2026.launch
```
Run a BARN test (simulation + mpc + send goal)
```
cd ~/jackal_ws/src/the-barn-challenge-robo
python3 run.py -l move_base_mlda_2026.launch # without rviz
python3 run.py -l move_base_mlda_2026.launch -r -rc mpc.rviz # with rviz
```

## For Ubuntu 18.04
As `acados-template` requires `python>3.8` and the default **python3** on Ubuntu 18.04 is **python 3.6**. 
**Python 3.8** needs to be installed and its virtual environment need to be sourced when running `script/setup_acados_mpc.sh` & `script/generate_acados_solver.py`.

```
sudo apt update
sudo apt install python3.8 python3.8-venv python3.8-dev
```

```
python3.8 -m venv ~/acados_env
source ~/acados_env/bin/activate
pip install --upgrade pip setuptools wheel
```

## TODO: edit `CMakeList.txt` & `package.xml` currently `tf2` is not included but used and `python3-casadi` in package.xml is can't be resolved by `rosdep`

