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
    
    # Parameters: [x0,y0, x1,y1, ..., x11,y11] - 2 static + 10 dynamic obstacle positions
    p = SX.sym('p', 24)
    
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
    
    # Distance constraints using parameters — 12 obstacles (2 static + 10 dynamic)
    dist_sq_list = []
    for _k in range(12):
        _px = p[2*_k]
        _py = p[2*_k + 1]
        dist_sq_list.append((x - _px)**2 + (y - _py)**2)
    
    # Implicit dynamics: f_impl = x_dot - f(x, u)
    f_impl = x_dot - f_expl
    
    # Nonlinear constraint expressions: [v_linear, omega, dist_sq_0 .. dist_sq_11]
    h_expr = vertcat(
        (vr + vl) / 2.0,  # linear velocity constraint (v_linear)
        (vr - vl) / L,    # angular velocity constraint (omega)
        *dist_sq_list      # 12 obstacle distance constraints
    )
    
    # Create ACADOS model
    model = AcadosModel()
    model.name = model_name
    model.x = state
    model.xdot = x_dot
    model.u = control
    model.p = p
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.con_h_expr = h_expr
    
    return model, L


def setup_acados_ocp():
    ocp = AcadosOcp()
    
    model, L = export_robot_model()
    ocp.model = model
    
    ocp.dims.np = 24
    ocp.parameter_values = np.array([1000.0] * 24)
    
    nx = 5   # [x, y, theta, vr, vl]
    nu = 2   # [ar, al]
    N  = 20  # prediction horizon
    Tf = 2.0 # [s]

    # =========================================================================
    # Cost: [x, y, theta, v_linear, ar, al]  ny = 6
    # =========================================================================
    ny = 4 + nu  # 6

    velocity_weight = 20.0
    Q = np.diag([85.0, 85.0, 41.0, velocity_weight])  # [x, y, theta, v_linear]
    R = np.diag([0.005, 0.005])

    ocp.cost.cost_type   = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'

    ocp.cost.W = np.block([[Q,               np.zeros((4, nu))],
                            [np.zeros((nu, 4)), R             ]])

    # Vx: map states -> cost outputs
    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[0, 0] = 1.0   # x
    ocp.cost.Vx[1, 1] = 1.0   # y
    ocp.cost.Vx[2, 2] = 1.0   # theta
    ocp.cost.Vx[3, 3] = 0.5   # v_linear = (vr + vl) / 2
    ocp.cost.Vx[3, 4] = 0.5

    # Vu: controls start at index 4
    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[4:, :] = np.eye(nu)

    ocp.cost.yref = np.zeros(ny)  # set online: [x, y, theta, v_ref, 0, 0]

    # Terminal cost: [x, y, theta] only — no velocity term
    Q_e = np.diag([85.0, 85.0, 41.0])
    ocp.cost.W_e = Q_e
    ocp.cost.Vx_e = np.zeros((3, nx))
    ocp.cost.Vx_e[0, 0] = 1.0
    ocp.cost.Vx_e[1, 1] = 1.0
    ocp.cost.Vx_e[2, 2] = 1.0
    ocp.cost.yref_e = np.zeros(3)

    # =========================================================================
    # Constraints
    # =========================================================================
    ocp.constraints.idxbx = np.array([3, 4])
    ocp.constraints.lbx   = np.array([-2.0, -2.0])
    ocp.constraints.ubx   = np.array([ 2.0,  2.0])

    ocp.constraints.lbu    = np.array([-2.0, -2.0])
    ocp.constraints.ubu    = np.array([ 2.0,  2.0])
    ocp.constraints.idxbu  = np.array([0, 1])

    v_linear_max = 2.0
    omega_max    = 1.8
    robot_radius=0.22
    safety_margin=0.00
    min_dist_sq = (robot_radius + safety_margin)**2

    # 14 constraints: v_linear, omega, 12 × distance_sq
    ocp.constraints.lh = np.array([-v_linear_max, -omega_max] + [min_dist_sq] * 12)
    ocp.constraints.uh = np.array([ v_linear_max,  omega_max] + [1e9]         * 12)

    # Soft constraints on all 12 obstacle distance constraints (indices 2..13)
    n_obs = 12
    ocp.constraints.idxsh = np.arange(2, 2 + n_obs)
    ns = n_obs
    ocp.constraints.ns = ns
    slack_weight   = 1000.0
    ocp.cost.zl    = slack_weight * np.ones(ns)
    ocp.cost.Zl    = slack_weight * np.ones(ns)
    ocp.cost.zu    = np.zeros(ns)
    ocp.cost.Zu    = np.zeros(ns)

    ocp.constraints.x0 = np.zeros(nx)

    # =========================================================================
    # Solver options
    # =========================================================================
    ocp.solver_options.N_horizon         = N
    ocp.solver_options.tf                = Tf
    ocp.solver_options.qp_solver         = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx    = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type   = 'ERK'
    ocp.solver_options.nlp_solver_type   = 'SQP_RTI'
    ocp.solver_options.nlp_solver_max_iter = 1
    ocp.solver_options.qp_solver_iter_max  = 20
    ocp.solver_options.tol               = 1e-3

    ocp.code_export_directory = 'c_generated_code'

    return ocp


def generate_solver():
    ocp = setup_acados_ocp()
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    print("ACADOS solver code generated successfully!")
    print("Generated files in: c_generated_code/")
    return acados_ocp_solver


if __name__ == '__main__':
    generate_solver()
