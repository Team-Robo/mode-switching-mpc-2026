#!/usr/bin/env python3
"""
ACADOS code generation script for differential drive MPC
State: [x, y, theta, vr, vl] (5 states)
Control: [ar, al] (2 controls - accelerations of right and left wheels)
"""

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import numpy as np
from casadi import SX, vertcat, sin, cos


def export_robot_model():
    model_name = 'jackal_diff_drive'
    
    # Constants
    L = 0.37558  # Distance between wheels [m]
    
    # State variables: [x, y, theta, vr, vl]
    x = SX.sym('x')
    y = SX.sym('y')
    theta = SX.sym('theta')
    vr = SX.sym('vr')  # right wheel velocity
    vl = SX.sym('vl')  # left wheel velocity
    
    state = vertcat(x, y, theta, vr, vl)
    
    # Control variables: [ar, al]
    ar = SX.sym('ar')  # right wheel acceleration
    al = SX.sym('al')  # left wheel acceleration
    
    control = vertcat(ar, al)
    
    # Differential drive dynamics
    v = (vr + vl) / 2  # linear velocity
    omega = (vr - vl) / L  # angular velocity
    
    # Explicit dynamics: x_dot = f(x, u)
    f_expl = vertcat(
        v * cos(theta),  # dx/dt
        v * sin(theta),  # dy/dt
        omega,           # dtheta/dt
        ar,              # dvr/dt
        al               # dvl/dt
    )
    
    # State derivative symbolic variable
    x_dot = SX.sym('x_dot', 5)
    
    # Implicit dynamics: f_impl = x_dot - f(x, u)
    f_impl = x_dot - f_expl
    
    # Create ACADOS model
    model = AcadosModel()
    model.name = model_name
    model.x = state
    model.xdot = x_dot
    model.u = control
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    
    return model, L


def setup_acados_ocp():
    # Create OCP
    ocp = AcadosOcp()
    
    # Export model
    model, L = export_robot_model()
    ocp.model = model
    
    # Dimensions
    nx = 5  # state dimension
    nu = 2  # control dimension
    N = 25  # prediction horizon (increased for better planning)
    
    # Time horizon (2.5s with N=25 gives dt=0.1s per step)
    Tf = 2.5  # [s]
    ocp.solver_options.tf = Tf
    
    # Cost matrices
    # Stage cost: weighted tracking error + control effort
    Q = np.diag([5.0, 5.0, 1.0, 0.1, 0.1])  # state weights [x, y, theta, vr, vl]
    R = np.diag([1.0, 1.0])  # control weights [ar, al]
    
    # Terminal cost
    Q_e = np.diag([5.0, 5.0, 1.0, 0.0, 0.0])
    
    # Set cost
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    
    # Stage cost
    ocp.cost.W = np.block([[Q, np.zeros((nx, nu))],
                            [np.zeros((nu, nx)), R]])
    
    ny = nx + nu  # output dimension
    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[nx:, :] = np.eye(nu)
    
    # Reference (will be set online)
    ocp.cost.yref = np.zeros(ny)
    
    # Terminal cost
    ocp.cost.W_e = Q_e
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.yref_e = np.zeros(nx)
    
    # Constraints
    # State constraints (only for vr and vl)
    ocp.constraints.idxbx = np.array([3, 4])  # constrain vr, vl
    ocp.constraints.lbx = np.array([-1.0, -1.0])  # bounds only for indices in idxbx
    ocp.constraints.ubx = np.array([1.0, 1.0])    # bounds only for indices in idxbx
    
    # Control constraints
    ocp.constraints.lbu = np.array([-1.0, -1.0])  # min accelerations
    ocp.constraints.ubu = np.array([1.0, 1.0])    # max accelerations
    ocp.constraints.idxbu = np.array([0, 1])
    
    # Nonlinear constraints for total velocity and angular velocity
    # v_total = vr + vl should be in [-2*v_max, 2*v_max]
    # omega = (vr - vl) / L should be in [w_min, w_max]
    v_max_total = 0.8
    w_max = 0.8
    
    from casadi import SX, vertcat
    x = model.x
    h_expr = vertcat(
        x[3] + x[4],  # vr + vl
        (x[3] - x[4]) / L  # omega
    )
    ocp.model.con_h_expr = h_expr
    ocp.constraints.lh = np.array([-2*v_max_total, -w_max])
    ocp.constraints.uh = np.array([2*v_max_total, w_max])
    
    # Initial state constraint (will be set online)
    ocp.constraints.x0 = np.zeros(nx)
    
    # Set dimensions
    ocp.solver_options.N_horizon = N  # Use N_horizon instead of deprecated dims.N
    
    # Solver options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'  # Explicit Runge-Kutta
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # Real-time iteration (best for i3 CPU)
    ocp.solver_options.nlp_solver_max_iter = 1  # RTI uses 1 iteration per call
    ocp.solver_options.qp_solver_iter_max = 20  # Reduced for i3 CPU
    ocp.solver_options.tol = 1e-3  # Slightly relaxed tolerance for speed
    
    # Code generation options
    ocp.code_export_directory = 'c_generated_code'
    
    return ocp


def generate_solver():
    """Generate the ACADOS solver C code"""
    ocp = setup_acados_ocp()
    
    # Create solver
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    
    print("ACADOS solver code generated successfully!")
    print("Generated files in: c_generated_code/")
    print("\nTo compile in your ROS package:")
    print("1. Copy c_generated_code/ to your package")
    print("2. Add ACADOS includes and libs to CMakeLists.txt")
    print("3. Link against acados library")
    
    return acados_ocp_solver


if __name__ == '__main__':
    generate_solver()
