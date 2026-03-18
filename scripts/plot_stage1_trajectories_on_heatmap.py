from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tmc26_exp.baselines import run_stage1_epec_diagonalization
from tmc26_exp.config import load_config
from tmc26_exp.simulator import sample_users
from tmc26_exp.stackelberg import run_stage1_solver


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("Value must be >= 1.")
    return value


def _nonnegative_float(raw: str) -> float:
    value = float(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("Value must be >= 0.")
    return value


def _load_summary_kv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _load_grid_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")

    required = {"pE", "pN", "esp_revenue", "nsp_revenue", "eps"}
    if not required.issubset(set(rows[0].keys() if rows else [])):
        raise ValueError(f"CSV missing required columns: {required}")
    has_eps_proxy = "epsilon_proxy" in set(rows[0].keys() if rows else [])

    pE_vals = sorted({float(r["pE"]) for r in rows})
    pN_vals = sorted({float(r["pN"]) for r in rows})
    pE_grid = np.asarray(pE_vals, dtype=float)
    pN_grid = np.asarray(pN_vals, dtype=float)
    n_rows, n_cols = pN_grid.size, pE_grid.size

    e_map = {v: i for i, v in enumerate(pE_vals)}
    n_map = {v: j for j, v in enumerate(pN_vals)}
    esp_rev = np.full((n_rows, n_cols), np.nan, dtype=float)
    nsp_rev = np.full((n_rows, n_cols), np.nan, dtype=float)
    eps = np.full((n_rows, n_cols), np.nan, dtype=float)
    eps_proxy = np.full((n_rows, n_cols), np.nan, dtype=float) if has_eps_proxy else None

    for r in rows:
        i = e_map[float(r["pE"])]
        j = n_map[float(r["pN"])]
        esp_rev[j, i] = float(r["esp_revenue"])
        nsp_rev[j, i] = float(r["nsp_revenue"])
        eps[j, i] = float(r["eps"])
        if eps_proxy is not None:
            eps_proxy[j, i] = float(r["epsilon_proxy"])

    if np.any(~np.isfinite(esp_rev)) or np.any(~np.isfinite(nsp_rev)) or np.any(~np.isfinite(eps)):
        raise ValueError("CSV grid is incomplete.")
    if eps_proxy is not None and np.any(~np.isfinite(eps_proxy)):
        raise ValueError("CSV epsilon_proxy grid is incomplete.")
    return pE_grid, pN_grid, esp_rev, nsp_rev, eps, eps_proxy


def _nearest_idx(grid: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(grid - value)))


def _nearest_positive_idx(grid: np.ndarray, target: float) -> int:
    positive_idx = np.flatnonzero(grid > 0.0)
    if positive_idx.size == 0:
        return _nearest_idx(grid, target)
    pos_vals = grid[positive_idx]
    local = int(np.argmin(np.abs(pos_vals - target)))
    return int(positive_idx[local])


def _snap_to_grid(
    pE_grid: np.ndarray,
    pN_grid: np.ndarray,
    pE: float,
    pN: float,
) -> tuple[int, int, float, float]:
    i = _nearest_idx(pE_grid, pE)
    j = _nearest_idx(pN_grid, pN)
    return i, j, float(pE_grid[i]), float(pN_grid[j])


def _rbf_kernel(x1: np.ndarray, x2: np.ndarray, length_scale: float) -> np.ndarray:
    ls = max(float(length_scale), 1e-9)
    x1_scaled = np.asarray(x1, dtype=float) / ls
    x2_scaled = np.asarray(x2, dtype=float) / ls
    d2 = np.sum((x1_scaled[:, None, :] - x2_scaled[None, :, :]) ** 2, axis=2)
    return np.exp(-0.5 * d2)


def _gp_predict(
    observed_x: np.ndarray,
    observed_y: np.ndarray,
    query_x: np.ndarray,
    length_scale: float,
    noise: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    if observed_x.shape[0] == 0:
        raise ValueError("At least one observed sample is required for GP prediction.")

    y = np.asarray(observed_y, dtype=float)
    y_mean = float(np.mean(y))
    y_scale = float(np.std(y))
    if y_scale < 1e-9:
        y_scale = 1.0
    y_norm = (y - y_mean) / y_scale

    k_xx = _rbf_kernel(observed_x, observed_x, length_scale)
    eye = np.eye(k_xx.shape[0], dtype=float)
    jitter = max(float(noise), 1e-9)
    chol: np.ndarray | None = None
    for _ in range(6):
        try:
            chol = np.linalg.cholesky(k_xx + jitter * eye)
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0

    if chol is None:
        mu = np.full(query_x.shape[0], y_mean, dtype=float)
        sigma = np.full(query_x.shape[0], y_scale, dtype=float)
        return mu, sigma

    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_norm))
    k_xs = _rbf_kernel(observed_x, query_x, length_scale)
    mu_norm = k_xs.T @ alpha
    v = np.linalg.solve(chol, k_xs)
    var_norm = np.maximum(1.0 - np.sum(v * v, axis=0), 1e-9)
    mu = y_mean + y_scale * mu_norm
    sigma = y_scale * np.sqrt(var_norm)
    return mu, sigma


def _parse_baselines(raw: str) -> list[str]:
    if not raw.strip():
        return []
    cleaned = raw.strip().replace(" ", "")
    if cleaned.lower() in {"none", "null"}:
        return []
    out: list[str] = []
    for part in cleaned.split(","):
        name = part.upper()
        if not name:
            continue
        if name not in {"BO", "BO-ONLINE", "DRL", "EPEC-DIAG"}:
            raise ValueError(f"Unsupported baseline: {name}. Allowed: BO, BO-ONLINE, DRL, EPEC-DIAG.")
        if name not in out:
            out.append(name)
    return out


def _trajectory_points(result) -> list[tuple[float, float]]:
    points = [(float(step.pE), float(step.pN)) for step in result.trajectory]
    final_p = (float(result.price[0]), float(result.price[1]))
    if not points:
        return [final_p]
    last = points[-1]
    if abs(last[0] - final_p[0]) + abs(last[1] - final_p[1]) > 1e-12:
        points.append(final_p)
    return points


def _load_trajectory_csv(path: Path) -> list[tuple[float, float]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Trajectory CSV has no rows: {path}")
    required = {"pE", "pN"}
    if not required.issubset(set(rows[0].keys() if rows else [])):
        raise ValueError(f"Trajectory CSV missing columns: {required}")
    points = [(float(r["pE"]), float(r["pN"])) for r in rows]
    if not points:
        raise ValueError(f"Trajectory CSV has no points: {path}")
    return points


def _simulate_bo_trajectory(
    pE_grid: np.ndarray,
    pN_grid: np.ndarray,
    surface: np.ndarray,
    bo_init_points: int,
    bo_iters: int,
    bo_candidate_pool: int,
    bo_kernel_bandwidth: float,
    bo_ucb_beta: float,
    seed: int,
    trace_mode: str,
    start_pE: float,
    start_pN: float,
) -> tuple[list[tuple[float, float]], dict[str, float | int | str]]:
    rng = np.random.default_rng(seed)
    x: list[tuple[float, float]] = []
    y: list[float] = []
    sampled: list[tuple[float, float]] = []
    incumbent: list[tuple[float, float]] = []
    best_idx: tuple[int, int] | None = None
    best_val = float("inf")
    init_points = max(1, int(bo_init_points))
    candidate_pool = max(1, int(bo_candidate_pool))
    n_iters = max(0, int(bo_iters))

    pE_min, pE_max = float(pE_grid[0]), float(pE_grid[-1])
    pN_min, pN_max = float(pN_grid[0]), float(pN_grid[-1])

    def evaluate_point(pE: float, pN: float) -> tuple[float, tuple[int, int], tuple[float, float]]:
        i, j, pE_snap, pN_snap = _snap_to_grid(pE_grid, pN_grid, pE, pN)
        return float(surface[j, i]), (j, i), (pE_snap, pN_snap)

    def record_sample(pE: float, pN: float) -> None:
        nonlocal best_idx, best_val
        val, idx, snapped = evaluate_point(pE, pN)
        x.append((pE, pN))
        y.append(val)
        sampled.append(snapped)
        if val < best_val or best_idx is None:
            best_val = val
            best_idx = idx
        assert best_idx is not None
        incumbent.append((float(pE_grid[best_idx[1]]), float(pN_grid[best_idx[0]])))

    start_eval_pE = min(max(float(start_pE), pE_min), pE_max)
    start_eval_pN = min(max(float(start_pN), pN_min), pN_max)
    record_sample(start_eval_pE, start_eval_pN)

    for _ in range(init_points - 1):
        record_sample(
            float(rng.uniform(pE_min, pE_max)),
            float(rng.uniform(pN_min, pN_max)),
        )

    for t in range(n_iters):
        cand = np.column_stack(
            [
                rng.uniform(pE_min, pE_max, size=candidate_pool),
                rng.uniform(pN_min, pN_max, size=candidate_pool),
            ]
        )
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        mu, sigma = _gp_predict(
            x_arr,
            y_arr,
            cand,
            length_scale=bo_kernel_bandwidth,
        )
        beta = max(float(bo_ucb_beta), 0.0) * math.sqrt(2.0 * math.log(t + 2.0))
        acquisition = mu - beta * sigma
        order = np.argsort(acquisition)
        pick = int(order[0])
        for idx in order:
            d2 = np.sum((x_arr - cand[int(idx)]) ** 2, axis=1)
            if float(np.min(d2)) > 1e-12:
                pick = int(idx)
                break
        record_sample(float(cand[pick, 0]), float(cand[pick, 1]))

    traj = sampled if trace_mode == "samples" else incumbent
    unique_count = len(set(sampled))
    meta = {
        "trace_mode": trace_mode,
        "evals": len(sampled),
        "unique_points": unique_count,
        "best_epsilon": float(best_val),
    }
    return traj, meta


def _simulate_drl_trajectory(
    pE_grid: np.ndarray,
    pN_grid: np.ndarray,
    esp_revenue: np.ndarray,
    nsp_revenue: np.ndarray,
    drl_price_levels: int,
    drl_episodes: int,
    drl_steps_per_episode: int,
    drl_alpha: float,
    drl_gamma: float,
    drl_epsilon: float,
    seed: int,
    start_pE: float,
    start_pN: float,
    rollout_steps: int,
) -> tuple[list[tuple[float, float]], dict[str, float | int | str]]:
    dgrid_e = np.linspace(float(pE_grid[0]), float(pE_grid[-1]), int(drl_price_levels))
    dgrid_n = np.linspace(float(pN_grid[0]), float(pN_grid[-1]), int(drl_price_levels))
    action_deltas = (-1, 0, 1)
    q_esp = np.zeros((dgrid_e.size, dgrid_n.size, len(action_deltas), len(action_deltas)), dtype=float)
    q_nsp = np.zeros((dgrid_e.size, dgrid_n.size, len(action_deltas), len(action_deltas)), dtype=float)
    rng = np.random.default_rng(seed)

    def clamp(idx: int, n: int) -> int:
        return min(max(idx, 0), n - 1)

    def greedy_joint_action(e_idx: int, n_idx: int) -> tuple[int, int, bool]:
        q_e = q_esp[e_idx, n_idx]
        q_n = q_nsp[e_idx, n_idx]
        best_e = np.max(q_e, axis=0, keepdims=True)
        best_n = np.max(q_n, axis=1, keepdims=True)
        pure_nash_mask = (q_e >= best_e - 1e-12) & (q_n >= best_n - 1e-12)
        if np.any(pure_nash_mask):
            candidates = np.argwhere(pure_nash_mask)
            scores = (q_e + q_n)[pure_nash_mask]
            pick = candidates[int(np.argmax(scores))]
            return int(pick[0]), int(pick[1]), True
        flat_idx = int(np.argmax(q_e + q_n))
        a_e, a_n = np.unravel_index(flat_idx, q_e.shape)
        return int(a_e), int(a_n), False

    def rewards_at(e_idx: int, n_idx: int) -> tuple[float, float]:
        i, j, _, _ = _snap_to_grid(pE_grid, pN_grid, float(dgrid_e[e_idx]), float(dgrid_n[n_idx]))
        return float(esp_revenue[j, i]), float(nsp_revenue[j, i])

    for _ in range(int(drl_episodes)):
        i = int(rng.integers(0, dgrid_e.size))
        j = int(rng.integers(0, dgrid_n.size))
        for _step in range(int(drl_steps_per_episode)):
            if float(rng.uniform()) < float(drl_epsilon):
                a_esp = int(rng.integers(0, len(action_deltas)))
                a_nsp = int(rng.integers(0, len(action_deltas)))
            else:
                a_esp, a_nsp, _ = greedy_joint_action(i, j)
            ni = clamp(i + action_deltas[a_esp], dgrid_e.size)
            nj = clamp(j + action_deltas[a_nsp], dgrid_n.size)
            reward_esp, reward_nsp = rewards_at(ni, nj)
            next_a_esp, next_a_nsp, _ = greedy_joint_action(ni, nj)
            td_target_esp = reward_esp + float(drl_gamma) * float(q_esp[ni, nj, next_a_esp, next_a_nsp])
            td_target_nsp = reward_nsp + float(drl_gamma) * float(q_nsp[ni, nj, next_a_esp, next_a_nsp])
            q_esp[i, j, a_esp, a_nsp] += float(drl_alpha) * (td_target_esp - q_esp[i, j, a_esp, a_nsp])
            q_nsp[i, j, a_esp, a_nsp] += float(drl_alpha) * (td_target_nsp - q_nsp[i, j, a_esp, a_nsp])
            i, j = ni, nj

    i = _nearest_idx(dgrid_e, start_pE)
    j = _nearest_idx(dgrid_n, start_pN)
    points: list[tuple[float, float]] = []
    visited_states: set[tuple[int, int]] = set()

    def append_point(e_idx: int, n_idx: int) -> None:
        e_snap = float(pE_grid[_nearest_idx(pE_grid, float(dgrid_e[e_idx]))])
        n_snap = float(pN_grid[_nearest_idx(pN_grid, float(dgrid_n[n_idx]))])
        points.append((e_snap, n_snap))

    append_point(i, j)
    visited_states.add((i, j))
    for _ in range(int(rollout_steps)):
        a_esp, a_nsp, _ = greedy_joint_action(i, j)
        ni = clamp(i + action_deltas[a_esp], dgrid_e.size)
        nj = clamp(j + action_deltas[a_nsp], dgrid_n.size)
        if ni == i and nj == j:
            break
        i, j = ni, nj
        append_point(i, j)
        state = (i, j)
        if state in visited_states:
            break
        visited_states.add(state)

    end_esp, end_nsp = rewards_at(i, j)
    meta = {
        "episodes": int(drl_episodes),
        "steps_per_episode": int(drl_steps_per_episode),
        "rollout_points": len(points),
        "final_esp_revenue": float(end_esp),
        "final_nsp_revenue": float(end_nsp),
        "final_joint_revenue": float(end_esp + end_nsp),
    }
    return points, meta


def _plot_compare(
    surface: np.ndarray,
    pE_grid: np.ndarray,
    pN_grid: np.ndarray,
    trajectories: dict[str, list[tuple[float, float]]],
    eps_tol: float,
    out_path: Path,
    cbar_label: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=150)
    im = ax.imshow(
        surface,
        origin="lower",
        aspect="auto",
        extent=[float(pE_grid[0]), float(pE_grid[-1]), float(pN_grid[0]), float(pN_grid[-1])],
        cmap="magma",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    se_mask = surface <= float(eps_tol)
    se_n_idx, se_e_idx = np.nonzero(se_mask)
    if se_n_idx.size > 0:
        ax.scatter(
            pE_grid[se_e_idx],
            pN_grid[se_n_idx],
            s=10,
            facecolors="none",
            edgecolors="white",
            linewidths=0.6,
            alpha=0.8,
            label=f"low-gap set (surface<={eps_tol:.2g})",
        )

    styles = {
        "VBBR": {"color": "cyan", "linestyle": "-", "linewidth": 1.8},
        "BO": {"color": "gold", "linestyle": "--", "linewidth": 1.5},
        "BO-ONLINE": {"color": "deepskyblue", "linestyle": ":", "linewidth": 1.7},
        "EPEC-DIAG": {"color": "white", "linestyle": "-", "linewidth": 1.6},
        "DRL": {"color": "lime", "linestyle": "-.", "linewidth": 1.5},
    }
    for name in ["VBBR", "BO", "BO-ONLINE", "EPEC-DIAG", "DRL"]:
        if name not in trajectories:
            continue
        traj = trajectories[name]
        if not traj:
            continue
        pE = np.asarray([pt[0] for pt in traj], dtype=float)
        pN = np.asarray([pt[1] for pt in traj], dtype=float)
        style = styles[name]
        ax.plot(
            pE,
            pN,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=0.95,
            label=f"{name} path",
        )
        ax.scatter([pE[0]], [pN[0]], c=style["color"], s=50, marker="o", edgecolors="black", linewidths=0.5)
        ax.scatter([pE[-1]], [pN[-1]], c=style["color"], s=70, marker="X", edgecolors="black", linewidths=0.5)

    ax.set_title(title)
    ax.set_xlabel("pE")
    ax.set_ylabel("pN")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_trajectory_csv(
    out_path: Path,
    trajectory: list[tuple[float, float]],
    pE_grid: np.ndarray,
    pN_grid: np.ndarray,
    eps: np.ndarray,
    surface: np.ndarray,
    surface_label: str,
) -> None:
    rows = [f"step,pE,pN,nearest_grid_pE,nearest_grid_pN,nearest_grid_eps,{surface_label}"]
    for k, (pE, pN) in enumerate(trajectory):
        i = _nearest_idx(pE_grid, pE)
        j = _nearest_idx(pN_grid, pN)
        rows.append(
            f"{k},{pE:.10g},{pN:.10g},{float(pE_grid[i]):.10g},{float(pN_grid[j]):.10g},{float(eps[j, i]):.10g},{float(surface[j, i]):.10g}"
        )
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot BO/BO-online/EPEC-diag/DRL/VBBR trajectory comparison on a known Stage-I heatmap CSV."
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to price_grid_metrics.csv.")
    parser.add_argument("--config", type=str, default="configs/default.toml", help="Path to TOML config.")
    parser.add_argument("--summary", type=str, default=None, help="Optional summary.txt for seed/config fallback.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed for VBBR user sampling.")
    parser.add_argument(
        "--eps-tol",
        type=_nonnegative_float,
        default=None,
        help="SE tolerance for highlighting points. Default: read from summary or 1e-12.",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default="BO,DRL",
        help="Comma-separated baseline trajectories to draw. Supported: BO,BO-ONLINE,EPEC-DIAG,DRL. Use 'none' to skip.",
    )
    parser.add_argument(
        "--bo-trace-mode",
        type=str,
        choices=["best", "samples"],
        default="best",
        help="BO trajectory visualization mode: incumbent-best path or sampled-point path.",
    )
    parser.add_argument(
        "--surface",
        type=str,
        choices=["real_gap", "epsilon_proxy"],
        default="real_gap",
        help="Heatmap surface to visualize.",
    )
    parser.add_argument(
        "--bo-surface",
        type=str,
        choices=["real_gap", "epsilon_proxy"],
        default=None,
        help="Optional BO objective surface override. Default: same as --surface.",
    )
    parser.add_argument("--bo-seed", type=int, default=None, help="Optional BO RNG seed override.")
    parser.add_argument("--drl-seed", type=int, default=None, help="Optional DRL RNG seed override.")
    parser.add_argument(
        "--drl-rollout-steps",
        type=_positive_int,
        default=200,
        help="Max greedy rollout steps when visualizing DRL trajectory.",
    )
    parser.add_argument("--drl-start-pE", type=float, default=None, help="Optional DRL rollout start pE.")
    parser.add_argument("--drl-start-pN", type=float, default=None, help="Optional DRL rollout start pN.")
    parser.add_argument(
        "--vbbr-csv",
        type=str,
        default=None,
        help="Optional precomputed vbbr_trajectory.csv. If not set, VBBR will be recomputed.",
    )
    parser.add_argument(
        "--search-max-iters",
        type=_positive_int,
        default=None,
        help="Optional override for VBBR search_max_iters when recomputing.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/stage1_traj_compare_<timestamp>).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    pE_grid, pN_grid, esp_rev, nsp_rev, eps, eps_proxy = _load_grid_csv(csv_path)
    joint_revenue = esp_rev + nsp_rev
    bo_surface_name = str(args.bo_surface) if args.bo_surface is not None else str(args.surface)
    if args.surface == "epsilon_proxy" and eps_proxy is None:
        raise ValueError("Requested epsilon_proxy heatmap surface, but CSV does not contain column 'epsilon_proxy'.")
    if bo_surface_name == "epsilon_proxy" and eps_proxy is None:
        raise ValueError("Requested epsilon_proxy BO surface, but CSV does not contain column 'epsilon_proxy'.")
    heatmap_surface = eps if args.surface == "real_gap" else eps_proxy
    bo_surface = eps if bo_surface_name == "real_gap" else eps_proxy
    assert heatmap_surface is not None
    assert bo_surface is not None

    summary_path = Path(args.summary) if args.summary else None
    if summary_path is None:
        candidate = csv_path.parent / "summary.txt"
        if candidate.exists():
            summary_path = candidate
    summary = _load_summary_kv(summary_path) if summary_path else {}

    config_path = str(args.config)
    if args.config == "configs/default.toml" and "config" in summary:
        config_path = summary["config"]
    cfg = load_config(config_path)

    seed = int(args.seed) if args.seed is not None else int(summary.get("seed", cfg.seed))
    eps_tol = float(args.eps_tol) if args.eps_tol is not None else float(summary.get("eps_tol", 1e-12))
    baselines = _parse_baselines(args.baselines)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out = Path("outputs") / f"stage1_traj_compare_{timestamp}"
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    trajectories: dict[str, list[tuple[float, float]]] = {}
    users = None

    def ensure_users():
        nonlocal users
        if users is None:
            rng = np.random.default_rng(seed)
            users = sample_users(cfg, rng)
        return users

    meta_lines: list[str] = [
        f"csv = {csv_path}",
        f"config = {config_path}",
        f"seed = {seed}",
        f"eps_tol = {eps_tol}",
        f"heatmap_surface = {args.surface}",
        f"bo_objective = {'real_revenue_deviation_gap' if bo_surface_name == 'real_gap' else 'epsilon_proxy'}",
        "drl_rewards = esp_revenue,nsp_revenue",
        f"baselines_selected = {','.join(baselines) if baselines else 'none'}",
    ]

    if args.vbbr_csv is not None:
        vbbr_path = Path(args.vbbr_csv)
        vbbr_traj = _load_trajectory_csv(vbbr_path)
        trajectories["VBBR"] = vbbr_traj
        meta_lines.append(f"vbbr_source = csv:{vbbr_path}")
    else:
        stack_cfg = replace(cfg.stackelberg, stage1_solver_variant="vbbr_brd")
        if args.search_max_iters is not None:
            stack_cfg = replace(stack_cfg, search_max_iters=int(args.search_max_iters))
        res = run_stage1_solver(ensure_users(), cfg.system, stack_cfg)
        vbbr_traj = _trajectory_points(res)
        trajectories["VBBR"] = vbbr_traj
        meta_lines.extend(
            [
                "vbbr_source = recomputed",
                f"vbbr_outer_iterations = {res.outer_iterations}",
                f"vbbr_stage2_oracle_calls = {res.stage2_oracle_calls}",
                f"vbbr_stopping_reason = {res.stopping_reason}",
            ]
        )

    bo_seed = int(args.bo_seed) if args.bo_seed is not None else int(cfg.baselines.random_seed + 101)
    bo_online_seed = int(cfg.baselines.random_seed + 151)
    drl_seed = int(args.drl_seed) if args.drl_seed is not None else int(cfg.baselines.random_seed + 303)

    if "EPEC-DIAG" in baselines:
        epec_out, epec_traj = run_stage1_epec_diagonalization(
            ensure_users(),
            cfg.system,
            cfg.stackelberg,
            cfg.baselines,
            outcome_name="EPEC-DIAG",
        )
        trajectories["EPEC-DIAG"] = epec_traj
        meta_lines.extend(
            [
                f"epec_diag_iters = {epec_out.meta['iters']}",
                f"epec_diag_converged = {epec_out.meta['converged']}",
                f"epec_diag_grid_points = {epec_out.meta['grid_points']}",
                f"epec_diag_trajectory_len = {epec_out.meta['trajectory_len']}",
                f"epec_diag_stage2_unique_prices = {epec_out.meta['stage2_unique_prices']}",
                f"epec_diag_final_eps_proxy = {epec_out.epsilon_proxy}",
            ]
        )

    if "BO" in baselines:
        start_pE = float(cfg.stackelberg.initial_pE)
        start_pN = float(cfg.stackelberg.initial_pN)
        bo_traj, bo_meta = _simulate_bo_trajectory(
            pE_grid=pE_grid,
            pN_grid=pN_grid,
            surface=bo_surface,
            bo_init_points=int(cfg.baselines.bo_init_points),
            bo_iters=int(cfg.baselines.bo_iters),
            bo_candidate_pool=int(cfg.baselines.bo_candidate_pool),
            bo_kernel_bandwidth=float(cfg.baselines.bo_kernel_bandwidth),
            bo_ucb_beta=float(cfg.baselines.bo_ucb_beta),
            seed=bo_seed,
            trace_mode=args.bo_trace_mode,
            start_pE=start_pE,
            start_pN=start_pN,
        )
        trajectories["BO"] = bo_traj
        meta_lines.extend(
            [
                f"bo_seed = {bo_seed}",
                f"bo_trace_mode = {args.bo_trace_mode}",
                f"bo_evals = {bo_meta['evals']}",
                f"bo_unique_points = {bo_meta['unique_points']}",
                (
                    f"bo_best_epsilon = {bo_meta['best_epsilon']}"
                    if bo_surface_name == "real_gap"
                    else f"bo_best_epsilon_proxy = {bo_meta['best_epsilon']}"
                ),
            ]
        )

    if "BO-ONLINE" in baselines:
        start_pE = float(cfg.stackelberg.initial_pE)
        start_pN = float(cfg.stackelberg.initial_pN)
        bo_online_traj, bo_online_meta = _simulate_bo_trajectory(
            pE_grid=pE_grid,
            pN_grid=pN_grid,
            surface=bo_surface,
            bo_init_points=1,
            bo_iters=int(cfg.baselines.bo_iters),
            bo_candidate_pool=int(cfg.baselines.bo_candidate_pool),
            bo_kernel_bandwidth=float(cfg.baselines.bo_kernel_bandwidth),
            bo_ucb_beta=float(cfg.baselines.bo_ucb_beta),
            seed=bo_online_seed,
            trace_mode=args.bo_trace_mode,
            start_pE=start_pE,
            start_pN=start_pN,
        )
        trajectories["BO-ONLINE"] = bo_online_traj
        meta_lines.extend(
            [
                f"bo_online_seed = {bo_online_seed}",
                f"bo_online_trace_mode = {args.bo_trace_mode}",
                f"bo_online_evals = {bo_online_meta['evals']}",
                f"bo_online_unique_points = {bo_online_meta['unique_points']}",
                (
                    f"bo_online_best_epsilon = {bo_online_meta['best_epsilon']}"
                    if bo_surface_name == 'real_gap'
                    else f"bo_online_best_epsilon_proxy = {bo_online_meta['best_epsilon']}"
                ),
            ]
        )

    if "DRL" in baselines:
        start_pE = float(args.drl_start_pE) if args.drl_start_pE is not None else float(cfg.stackelberg.initial_pE)
        start_pN = float(args.drl_start_pN) if args.drl_start_pN is not None else float(cfg.stackelberg.initial_pN)
        if start_pE <= float(pE_grid[0]):
            start_i = _nearest_positive_idx(pE_grid, start_pE)
            start_pE = float(pE_grid[start_i])
        if start_pN <= float(pN_grid[0]):
            start_j = _nearest_positive_idx(pN_grid, start_pN)
            start_pN = float(pN_grid[start_j])
        drl_traj, drl_meta = _simulate_drl_trajectory(
            pE_grid=pE_grid,
            pN_grid=pN_grid,
            esp_revenue=esp_rev,
            nsp_revenue=nsp_rev,
            drl_price_levels=int(cfg.baselines.drl_price_levels),
            drl_episodes=int(cfg.baselines.drl_episodes),
            drl_steps_per_episode=int(cfg.baselines.drl_steps_per_episode),
            drl_alpha=float(cfg.baselines.drl_alpha),
            drl_gamma=float(cfg.baselines.drl_gamma),
            drl_epsilon=float(cfg.baselines.drl_epsilon),
            seed=drl_seed,
            start_pE=start_pE,
            start_pN=start_pN,
            rollout_steps=int(args.drl_rollout_steps),
        )
        trajectories["DRL"] = drl_traj
        meta_lines.extend(
            [
                f"drl_seed = {drl_seed}",
                f"drl_rollout_start = ({start_pE:.10g},{start_pN:.10g})",
                f"drl_episodes = {drl_meta['episodes']}",
                f"drl_steps_per_episode = {drl_meta['steps_per_episode']}",
                f"drl_rollout_points = {drl_meta['rollout_points']}",
                f"drl_final_esp_revenue = {drl_meta['final_esp_revenue']}",
                f"drl_final_nsp_revenue = {drl_meta['final_nsp_revenue']}",
                f"drl_final_joint_revenue = {drl_meta['final_joint_revenue']}",
            ]
        )

    fig_name = "eps_heatmap_trajectory_compare.png" if args.surface == "real_gap" else "eps_proxy_heatmap_trajectory_compare.png"
    fig_path = out_dir / fig_name
    _plot_compare(
        surface=heatmap_surface,
        pE_grid=pE_grid,
        pN_grid=pN_grid,
        trajectories=trajectories,
        eps_tol=eps_tol,
        out_path=fig_path,
        cbar_label="epsilon" if args.surface == "real_gap" else "epsilon_proxy",
        title=f"Stage-I trajectories on {'epsilon' if args.surface == 'real_gap' else 'epsilon proxy'} heatmap",
    )

    for name, traj in trajectories.items():
        surface = joint_revenue
        surface_label = "nearest_grid_joint_revenue"
        if name in {"BO", "BO-ONLINE"}:
            surface = bo_surface
            surface_label = (
                (
                    "nearest_grid_bo_online_real_gap"
                    if name == "BO-ONLINE" and bo_surface_name == "real_gap"
                    else "nearest_grid_bo_online_epsilon_proxy"
                )
                if name == "BO-ONLINE"
                else (
                    "nearest_grid_bo_real_gap" if bo_surface_name == "real_gap" else "nearest_grid_bo_epsilon_proxy"
                )
            )
        elif name == "EPEC-DIAG":
            surface = joint_revenue
            surface_label = "nearest_grid_joint_revenue"
        stem = name.lower().replace("-", "_")
        _save_trajectory_csv(
            out_dir / f"{stem}_trajectory.csv",
            traj,
            pE_grid,
            pN_grid,
            eps,
            surface,
            surface_label,
        )
        meta_lines.append(f"{stem}_points = {len(traj)}")

    (out_dir / "summary.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
    print(f"Done. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
