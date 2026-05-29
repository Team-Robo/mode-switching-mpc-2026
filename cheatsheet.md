# MPC Cheatsheet (Current)

This file summarizes the current behavior and effective parameters of the MPC stack in this repository.

## 1) Solver/codegen status

Primary sources checked:

- `script/generate_acados_solver.py`
- `src/mpc_controller.cpp`
- `src/mpc_local_planner_ros.cpp`
- `configs/mpc_local_planner_params.yaml`
- `launch/move_base_mlda_2026.launch`

Current ACADOS codegen settings (from `generate_acados_solver.py`):

- Horizon: `N=20`, `Tf=2.0`
- State: `[x, y, theta, vr, vl]`
- Control: `[ar, al]`
- Runtime parameter vector size: `np=24` (2 static + 10 dynamic obstacle points)
- Solver type: `SQP_RTI`
- `nlp_solver_max_iter=1`
- `qp_solver_iter_max=40`

Generated template constraints (before runtime overwrite):

- Wheel speed bounds: `vr, vl in [-2, 2]`
- Control bounds: `ar, al in [-2, 2]`
- Nonlinear `h`:
  - `h[0]=v_linear` in `[-2, 2]`
  - `h[1]=omega` in `[-1.8, 1.8]`
  - `h[2..13]=distance_sq` lower-bounded by template `min_dist_sq`
- Obstacle soft constraints enabled on `h[2..13]` with slack penalty `1000`

## 2) Runtime mode behavior

Current solve-path behavior in `mpc_controller.cpp`:

- `DYNAMIC_OBS` mode logic is disabled (`dynamic_obs_active=false`).
- `RUSH_GOAL` mode logic is disabled (`rush_goal_latched_=false`).
- Rotation-shim mode-selection block is commented out (startup scan still uses the label temporarily).
- Effective solve-time modes are therefore typically:
  - `NORMAL`
  - `STATIC_OBS`
- Reversal is an overlay (heading-reference overlay), not a separate mode.

## 3) Runtime constraints (effective)

The controller overwrites nonlinear bounds (`lh/uh`) every cycle.

| Item | Template value (codegen) | Runtime effective now |
|---|---|---|
| `h[0]` linear speed cap | `[-2, 2]` | `[-v_cap, +v_cap]`, where `v_cap=v_linear_max` for both `NORMAL` and `STATIC_OBS` |
| `h[1]` angular speed cap | `[-1.8, 1.8]` | `[-omega_cap, +omega_cap]`, where `omega_cap=omega_max` for both `NORMAL` and `STATIC_OBS` |
| `h[2..13]` obstacle distance lower bounds | template `min_dist_sq` | runtime `min_dist_sq=(robot_radius+safety_margin)^2` in active modes |
| Stage-0 state | template `x0=zeros` | hard-fixed each cycle to measured `current_state` via stage-0 `lbx=ubx` |

Current values with `configs/mpc_local_planner_params.yaml`:

- `robot_radius=0.22`
- `safety_margin=0.01`
- active-mode `min_dist_sq=(0.22+0.01)^2=0.0529`

If dynamic mode is re-enabled, code uses:

- `min_dist_sq=(robot_radius+dynamic_obs_radius+safety_margin)^2`
- with current defaults (`dynamic_obs_radius=0.5` unless set), that is `(0.22+0.5+0.01)^2=0.5329`

Obstacle parameter packing now uses:

- one closest static obstacle on the **left**
- one closest static obstacle on the **right**
- up to 10 dynamic obstacles

## 4) References and goal handling

Reference generation path:

- Local reference points are subsampled from global plan.
- Exact global goal point is explicitly appended if missing.
- Heading reference is built from path tangent (`buildHeadingRefFromPath`).

Current goal-orientation handling:

- Final goal yaw is stored from `plan.back().pose.orientation`.
- MPC terminal heading is explicitly forced to goal yaw:
  - terminal `theta_ref_` is aligned in `setPlan(...)`
  - `theta_sub.back()` is aligned before `solveOCP(...)`
- Near-goal rotate branch in `MpcLocalPlannerROS` is also active by config:
  - `enable_rotate_to_goal: true`
  - in-place rotation with costmap footprint safety checks

## 5) Cost weights and retry profile

Current base weights from `configs/mpc_local_planner_params.yaml`:

- `weight_position_error=155.0`
- `weight_heading_error=92.0`
- `weight_velocity=50.0`
- `weight_acceleration=0.0003088547426299309`

Multipliers:

- `accel_weight_mult_static` is not set in this YAML (controller default used)
- `accel_weight_mult_dynamic=4.9711669468300554`
- `position_weight_mult_dynamic=0.7000000000000001`
- `heading_weight_mult_dynamic=0.44999999999999996`
- `vel_weight_mult_dynamic=1.113057650495854`

Retry profile status:

- `retry_profile_enabled=false` in current YAML (disabled unless toggled)
- When enabled, retry profile applies to `STATIC_OBS` and `NORMAL` for:
  - heading weight scaling
  - accel weight scaling
- `retry_v_ref_scale` affects static-mode `v_ref` path (normal-mode `v_ref` remains `v_linear_max`).

## 6) Key ROS params currently loaded

Loaded under `move_base` namespace `MpcLocalPlannerROS`:

- `configs/mpc_local_planner_params.yaml`

Notable values currently set there:

- `v_ref_static: 0.8`
- `min_spacing_global_plan: 0.05`
- `static_obs_safe_dist: 1.2`
- `dynamic_obs_safe_dist: 8.8`
- `xy_goal_tolerance: 0.25`
- `yaw_goal_tolerance: 0.157`
- `enable_rotate_to_goal: true` and rotate tuning params
- `bench_log_enabled: false`

Separate node params loaded from `configs/teamrobo2026_mpc_params.yaml`:

- `map_to_cloud.bbox_length: 3.0`
- `map_to_cloud.bbox_viz_enabled: true`

## 7) Build/runtime linkage note

`CMakeLists.txt` and `package.xml` currently include `base_local_planner` dependencies, and plugin link uses:

- `-Wl,--no-as-needed` before `${catkin_LIBRARIES}`
- `-Wl,--as-needed` after link list

This is to keep `libbase_local_planner.so` as a required runtime dependency for the local planner plugin.

## 8) Current failure checklist

Even with obstacle soft constraints, solver can fail due to hard constraints and QP numerical issues:

- Stage-0 fixed state can conflict with hard nonlinear limits in that cycle.
- Hard `omega`/`v` bounds can be violated by measured state transients.
- RTI single-NLP-step setup (`SQP_RTI`, `nlp_solver_max_iter=1`) is sensitive to poor local linearization.
- Large heading mismatch and low-speed turning can trigger repeated `ACADOS_MINSTEP`.
- Invalid numerical inputs (NaN/Inf) still break solve path.

Observed practical pattern:

- bursts of `QP status=3 (MINSTEP)` with `status=4` in controller logs,
- intermittent recovery next cycle when local linearization/warm-start improves.
