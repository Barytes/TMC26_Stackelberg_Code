# Figure 6 start-point explanation

## Root cause
Figure 6 marks `Start` using the first element of the exported Algorithm-5 trajectory (`traj_pE[0], traj_pN[0]`) in `scripts/run_stage1_boundary_visualization.py` (lines 221–222).

However, the trajectory itself is recorded **after** Stage-2 warm start and price refinement in Algorithm 5:
- Initial configured price is read as `(initial_pE, initial_pN)` at `src/tmc26_exp/stackelberg.py:945`.
- Stage-2 warm start chooses an initial offloading set at `:946–948`.
- Then price is immediately refined on that set at `:949`.
- Inside the main loop, price is refined again at `:961` before the first `trajectory.append(...)` at `:986`.

So the first recorded point is the first refined/warm-started point, not the raw configured initial point (e.g., `(0.01, 0.01)`).

## Where the first recorded point is produced
- **Producer:** `src/tmc26_exp/stackelberg.py:986` (`trajectory.append(SearchStep(... pE=current_price[0], pN=current_price[1], ...))`)
- **Consumer for plotting Start marker:** `scripts/run_stage1_boundary_visualization.py:221–222`

## Suggested fix (to plot true configured initial point)
Option A (minimal visualization-only fix):
- In `run_stage1_boundary_visualization.py`, plot `Start` from config/system-clipped initial price:
  - `p0 = (max(stackelberg_cfg.initial_pE, system.cE), max(stackelberg_cfg.initial_pN, system.cN))`
- Keep trajectory curve unchanged (still algorithm-iteration points).

Option B (algorithm-trajectory fix):
- Add an explicit pre-iteration trajectory entry before warm-start refinement (or at least before first loop-time refinement), e.g., iteration `-1` with the configured initial price.
- Then existing `Start = traj[0]` remains semantically correct.

Option A is safer for reproducing existing algorithm behavior while correcting figure semantics.
