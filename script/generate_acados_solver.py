#!/usr/bin/env python3
"""
ACADOS code generation script for differential drive MPC
State: [x, y, theta, vr, vl] (5 states)
Control: [ar, al] (2 controls - accelerations of right and left wheels)

FIXED:
- Velocity constraints now properly handle v_linear ∈ [-2, 2] m/s
- Clearer variable naming
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
    
    # Parameters: [xL, yL, xR, yR] - obstacle positions
    p = SX.sym('p', 4)
    
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
    
    # Distance constraints using parameters
    p_obs_L = p[0:2]
    p_obs_R = p[2:4]
    dist_L_sq = (x - p_obs_L[0])**2 + (y - p_obs_L[1])**2
    dist_R_sq = (x - p_obs_R[0])**2 + (y - p_obs_R[1])**2
    
    # Implicit dynamics: f_impl = x_dot - f(x, u)
    f_impl = x_dot - f_expl
    
    # Nonlinear constraint expressions
    # NOTE: v_linear = (vr + vl)/2, so to constrain v_linear we constrain (vr+vl)/2
    h_expr = vertcat(
        (vr + vl) / 2.0,           # linear velocity constraint (v_linear)
        (vr - vl) / L,             # angular velocity constraint (omega)
        dist_L_sq,                 # left obstacle distance squared
        dist_R_sq                  # right obstacle distance squared
    )
    
    # Create ACADOS model
    model = AcadosModel()
    model.name = model_name
    model.x = state
    model.xdot = x_dot
    model.u = control
    model.p = p  # Add parameters to model
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.con_h_expr = h_expr  # Nonlinear constraints
    
    return model, L


def setup_acados_ocp():
    # Create OCP
    ocp = AcadosOcp()
    
    # Export model
    model, L = export_robot_model()
    ocp.model = model
    
    # Parameter dimensions
    ocp.dims.np = 4  # Number of parameters
    
    # Initialize parameters with "ghost" values far away
    ocp.parameter_values = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    
    # Dimensions
    nx = 5  # state dimension
    nu = 2  # control dimension
    N = 25  # prediction horizon
    
    Tf = 2.5  # [s]
    ocp.solver_options.tf = Tf
    
    # Cost matrices
    # Stage cost: weighted tracking error + control effort
    Q = np.diag([5.0, 5.0, 1.0, 2.5, 2.5])  # state weights [x, y, theta, vr, vl]
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
    ocp.constraints.lbx = np.array([-2.0, -2.0])  # individual wheel velocity bounds
    ocp.constraints.ubx = np.array([2.0, 2.0])    # individual wheel velocity bounds
    
    # Control constraints
    ocp.constraints.lbu = np.array([-1.0, -1.0])  # min accelerations
    ocp.constraints.ubu = np.array([1.0, 1.0])    # max accelerations
    ocp.constraints.idxbu = np.array([0, 1])
    
    # Nonlinear constraints
    # h_expr = [(vr+vl)/2, (vr-vl)/L, dist_L_sq, dist_R_sq]
    # Index 0: v_linear = (vr+vl)/2 ∈ [-2.0, 2.0] m/s
    # Index 1: omega = (vr-vl)/L ∈ [-0.8, 0.8] rad/s
    # Index 2,3: distance squared constraints
    
    v_linear_max = 2.0  # Maximum linear velocity [m/s]
    omega_max = 0.8     # Maximum angular velocity [rad/s]
    min_dist_sq = 0.37**2  # Minimum distance squared to obstacles
    
    ocp.constraints.lh = np.array([-v_linear_max, -omega_max, min_dist_sq, min_dist_sq])
    ocp.constraints.uh = np.array([v_linear_max, omega_max, 1e9, 1e9])

    # SOFT CONSTRAINTS
    # Indices 2 and 3 correspond to dist_L_sq and dist_R_sq in h_expr
    ocp.constraints.idxsh = np.array([2, 3])

    # Soft constraint Penalty Weight
    slack_weight = 1000.0

    ns = 2  # Number of soft constraints
    ocp.constraints.ns = ns

    # Lower bound slack weights (Used when dist < min_dist)
    ocp.cost.zl = slack_weight * np.ones(ns)
    ocp.cost.Zl = slack_weight * np.ones(ns)

    # Upper bound slack weights (Required by ACADOS structure, unused for distance)
    ocp.cost.zu = np.zeros(ns)
    ocp.cost.Zu = np.zeros(ns)
    
    # Initial state constraint (will be set online)
    ocp.constraints.x0 = np.zeros(nx)
    
    # Set dimensions
    ocp.solver_options.N_horizon = N
    
    # Solver options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'  # Explicit Runge-Kutta
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # Real-time iteration
    ocp.solver_options.nlp_solver_max_iter = 1  # RTI uses 1 iteration per call
    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.tol = 1e-3
    
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
    print("\nVelocity constraints:")
    print("  - Linear velocity: [-2.0, 2.0] m/s")
    print("  - Angular velocity: [-0.8, 0.8] rad/s")
    print("  - Individual wheel velocities: [-2.0, 2.0] m/s")
    
    return acados_ocp_solver


if __name__ == '__main__':
    generate_solver()