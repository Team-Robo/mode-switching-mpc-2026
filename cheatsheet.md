# MPC Failure Causes + Parameter Table (Current)

## 1) Possible failure causes

1. **QP MINSTEP / QP non-convergence (`ACADOS_MINSTEP`)**
   - Current solver config is `SQP_RTI` with `nlp_solver_max_iter=1`; difficult local linearizations can fail in one RTI step.
2. **Stage-0 hard infeasibility**
   - Stage-0 state is hard-fixed to measured `current_state` (`lbx=ubx` at stage 0). If measured state conflicts with hard bounds (`v`, `omega`, `vr`, `vl`), solve can fail immediately.
3. **Hard bound violations**
   - Hard constraints remain active for `v`, `omega`, wheel speed, and controls (`ar`, `al`) even though obstacle-distance constraints are soft.
4. **Obstacle-distance pressure**
   - If predicted/static obstacle geometry plus hard kinematics makes progress infeasible, QP can fail despite soft obstacle slack.
5. **Invalid numeric data**
   - NaN/Inf in state, reference, or obstacle parameters can break the NLP/QP.
6. **Runtime ACADOS call failures**
   - Any nonzero return in setter/update calls (`lh/uh`, `W`, `yref`, `update_params`) triggers early fail path.

---

## 2) MPC parameter table (constraints, weights, cost/reference)

### 2.1 Constraints

| Parameter | Script value (`generate_acados_solver.py`) | Overwritten in `mpc_controller.cpp`? | Runtime default (controller) |
|---|---|---|---|
| Wheel speed bounds (`idxbx=[3,4]`, `lbx/ubx`) | `vr,vl in [-2, 2]` | No | `[-2, 2]` |
| Control bounds (`idxbu=[0,1]`, `lbu/ubu`) | `ar,al in [-2, 2]` | No | `[-2, 2]` |
| Nonlinear bound `h[0]` (linear speed) | `[-2, 2]` (via `v_linear_max=2.0`) | Yes (`lh/uh` set every cycle) | `[-v_cap, +v_cap]`; defaults: `v_cap=v_linear_max=2.0` in `NORMAL/STATIC/DYNAMIC/RUSH`, `v_cap=0` in `ROTATION_SHIM` |
| Nonlinear bound `h[1]` (angular speed) | `[-1.8, 1.8]` (via `omega_max=1.8`) | Yes (`lh/uh` set every cycle) | `[-omega_cap, +omega_cap]`; defaults: `omega_cap=omega_max=1.8` in `NORMAL/STATIC/DYNAMIC/RUSH`, `omega_cap=shim_omega=0.8` in `ROTATION_SHIM` |
| Nonlinear bounds `h[2..13]` obstacle distance lower bounds | `min_dist_sq=(robot_radius+safety_margin)^2=(0.22+0.01)^2=0.0529` | Yes (`lh` rebuilt every cycle) | Non-dynamic formula: `(robot_radius+safety_margin)^2` default `(0.37+0.01)^2=0.1444`; dynamic formula: `(robot_radius+dynamic_obs_radius+safety_margin)^2` default `(0.37+0.5+0.01)^2=0.7744` |
| Obstacle soft-constraint indices (`idxsh`) | `idxsh=[2..13]`, `ns=12` | No | Same as script |
| Obstacle slack weights (`zl/Zl/zu/Zu`) | `zl=1000`, `Zl=1000`, `zu=0`, `Zu=0` | No | Same as script |
| Initial state (`x0`) | `x0=zeros(nx)` template | Yes | Stage-0 hard-set each cycle to measured `current_state` (`lbx=ubx`) |
| Runtime obstacle parameter vector (`p`, size 24) | initialized to all `1000.0` | Yes (`update_params` each stage) | Per-stage selected obstacles are injected each cycle |

### 2.2 Weights (stage + terminal cost matrices)

| Parameter | Script value (`generate_acados_solver.py`) | Overwritten in `mpc_controller.cpp`? | Runtime default (controller) |
|---|---|---|---|
| Stage cost matrix `W` | diag from `Q=[85,85,41,20]`, `R=[0.005,0.005]` | Yes (`W` set each stage) | `W=diag([pos, pos, heading, vel, accel, accel])` with mode-dependent values below |
| Terminal cost matrix `W_e` | `diag([85,85,41])` | Yes (`W_e` set at terminal stage) | `W_e=diag([pos, pos, heading])`, mode-dependent (RUSH uses rush weights) |
| Base position weight | `85` (inside script `Q`) | Yes | `weight_position_error=128.0` |
| Base heading weight | `41` (inside script `Q`) | Yes | `weight_heading_error=57.0` |
| Base velocity weight | `20` (inside script `Q`) | Yes | `weight_velocity=33.0` |
| Base accel weights | `0.005, 0.005` (inside script `R`) | Yes | `weight_acceleration=0.01803665193219243` |
| STATIC accel multiplier | N/A | Yes | `accel_weight_mult_static=3.0` |
| DYNAMIC multipliers | N/A | Yes | `position=0.7`, `heading=0.45`, `velocity=1.113057650495854`, `accel=4.9711669468300554` |
| RUSH absolute weights | N/A | Yes | `rush_weight_position=0.0`, `rush_weight_heading=3030.0`, `rush_weight_velocity=3050.0`, `rush_weight_accel=0.0` |
| Retry profile scales | N/A | Yes | `retry_v_ref_scale=0.7`, `retry_heading_weight_scale=0.7`, `retry_accel_weight_scale=1.5` (applied on retry path) |

### 2.3 Cost/reference parameters

| Parameter | Script value (`generate_acados_solver.py`) | Overwritten in `mpc_controller.cpp`? | Runtime default (controller) |
|---|---|---|---|
| Stage reference `yref` shape | `[x, y, theta, v_ref, 0, 0]` (initialized zeros) | Yes (set per stage online) | Same shape, set each cycle |
| Terminal reference `yref_e` shape | `[x, y, theta]` (initialized zeros) | Yes (set at stage `N`) | Same shape, set each cycle |
| `x/y` reference source | N/A (online) | Yes | Built from global plan subsampling with `min_spacing_global_plan=0.14` default |
| Heading reference (`theta`) | N/A (online) | Yes | Built from path tangent (`buildHeadingRefFromPath`); terminal heading is explicitly aligned to stored `goal_yaw_` |
| `v_ref` (NORMAL) | N/A (online) | Yes | `v_ref=v_linear_max=2.0` |
| `v_ref` (STATIC_OBS) | N/A (online) | Yes | `v_ref=clamp(v_ref_static, 0, v_linear_max)` with default `v_ref_static=1.0` |
| `v_ref` (RUSH_GOAL) | N/A (online) | Yes | `v_ref=rush_vref=2.0` |
| `v_ref` (ROTATION_SHIM) | N/A (online) | Yes | `v_ref=0.0` |
| Goal pose/yaw tracking | N/A | Yes | Goal pose is stored from `plan.back()` and used for terminal heading alignment + goal checks |

---

### Ownership quick view

- **Script-only (not runtime-overwritten):**
  - `lbx/ubx` (wheel speed), `lbu/ubu` (control bounds), `idxsh/ns`, slack weights (`zl/Zl/zu/Zu`).
- **Script + runtime-overwritten in controller:**
  - Nonlinear bounds `lh/uh`, `W`, `W_e`, `yref`, `yref_e`, stage-0 initial-state enforcement, runtime obstacle parameter vector `p`.
# MPC Cheatsheet (Current)