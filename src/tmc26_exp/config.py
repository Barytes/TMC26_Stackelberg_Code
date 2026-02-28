from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass(frozen=True)
class DistributionSpec:
    kind: str
    params: dict[str, Any]


@dataclass(frozen=True)
class UserDistributions:
    w: DistributionSpec
    d: DistributionSpec
    fl: DistributionSpec
    alpha: DistributionSpec
    beta: DistributionSpec
    rho: DistributionSpec
    varpi: DistributionSpec
    kappa: DistributionSpec
    sigma: DistributionSpec


@dataclass(frozen=True)
class PriceAxis:
    min: float
    max: float
    points: int


@dataclass(frozen=True)
class SystemConfig:
    F: float
    B: float
    cE: float
    cN: float


@dataclass(frozen=True)
class StackelbergConfig:
    enabled: bool
    initial_pE: float
    initial_pN: float
    inner_eta_F0: float
    inner_eta_B0: float
    inner_eta_mu0: float
    inner_max_iters: int
    inner_tol: float
    greedy_max_iters: int
    rne_directions: int
    rne_root_tol: float
    rne_max_expand_steps: int
    search_max_iters: int
    search_improvement_tol: float


@dataclass(frozen=True)
class BaselineConfig:
    enabled: bool
    random_seed: int
    exact_max_users: int
    stage2_solver_for_pricing: str
    max_price_E: float
    max_price_N: float
    gso_grid_points: int
    pbdr_grid_points: int
    pbdr_max_iters: int
    pbdr_tol: float
    bo_init_points: int
    bo_iters: int
    bo_candidate_pool: int
    bo_kernel_bandwidth: float
    bo_ucb_beta: float
    drl_price_levels: int
    drl_episodes: int
    drl_steps_per_episode: int
    drl_alpha: float
    drl_gamma: float
    drl_epsilon: float
    market_max_iters: int
    market_step_size: float
    single_sp_max_iters: int
    random_offloading_trials: int
    random_offloading_prob: float
    ubrd_max_iters: int
    vi_max_iters: int
    vi_step_size: float
    vi_tol: float
    penalty_outer_iters: int
    penalty_inner_iters: int
    penalty_init_rho: float
    penalty_rho_scale: float
    penalty_tol: float


@dataclass(frozen=True)
class DetailedExperimentConfig:
    emit_plan: bool
    output_subdir: str
    suggested_trials: int
    heavy_trials: int


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    output_dir: str
    n_users: int
    n_trials: int
    seed: int
    pE: PriceAxis
    pN: PriceAxis
    metrics: list[str]
    users: UserDistributions
    system: SystemConfig
    stackelberg: StackelbergConfig
    baselines: BaselineConfig
    detailed_experiment: DetailedExperimentConfig


def _parse_distribution(raw: dict[str, Any]) -> DistributionSpec:
    if "kind" not in raw:
        raise ValueError("Distribution entry requires field 'kind'.")
    kind = str(raw["kind"]).strip().lower()
    params = {k: v for k, v in raw.items() if k != "kind"}
    return DistributionSpec(kind=kind, params=params)


def _parse_price_axis(raw: dict[str, Any], name: str) -> PriceAxis:
    try:
        axis = PriceAxis(
            min=float(raw["min"]),
            max=float(raw["max"]),
            points=int(raw["points"]),
        )
    except KeyError as exc:
        raise ValueError(f"Missing field in [{name}] axis config: {exc}") from exc
    if axis.min <= 0 or axis.max <= axis.min or axis.points < 2:
        raise ValueError(f"Invalid [{name}] axis config: {axis}")
    return axis


def _parse_system(raw: dict[str, Any]) -> SystemConfig:
    sys = SystemConfig(
        F=float(raw.get("F", 20.0e9)),
        B=float(raw.get("B", 50.0e6)),
        cE=float(raw.get("cE", 0.1)),
        cN=float(raw.get("cN", 0.1)),
    )
    if sys.F <= 0 or sys.B <= 0:
        raise ValueError("System capacities F and B must be positive.")
    if sys.cE < 0 or sys.cN < 0:
        raise ValueError("System costs cE and cN must be non-negative.")
    return sys


def _parse_stackelberg(raw: dict[str, Any]) -> StackelbergConfig:
    cfg = StackelbergConfig(
        enabled=bool(raw.get("enabled", True)),
        initial_pE=float(raw.get("initial_pE", 0.5)),
        initial_pN=float(raw.get("initial_pN", 0.5)),
        inner_eta_F0=float(raw.get("inner_eta_F0", 0.5)),
        inner_eta_B0=float(raw.get("inner_eta_B0", 0.5)),
        inner_eta_mu0=float(raw.get("inner_eta_mu0", 0.5)),
        inner_max_iters=int(raw.get("inner_max_iters", 500)),
        inner_tol=float(raw.get("inner_tol", 1e-4)),
        greedy_max_iters=int(raw.get("greedy_max_iters", 256)),
        rne_directions=int(raw.get("rne_directions", 20)),
        rne_root_tol=float(raw.get("rne_root_tol", 1e-5)),
        rne_max_expand_steps=int(raw.get("rne_max_expand_steps", 50)),
        search_max_iters=int(raw.get("search_max_iters", 20)),
        search_improvement_tol=float(raw.get("search_improvement_tol", 1e-9)),
    )
    if cfg.initial_pE <= 0 or cfg.initial_pN <= 0:
        raise ValueError("Initial prices must be positive.")
    if cfg.inner_eta_F0 <= 0 or cfg.inner_eta_B0 <= 0 or cfg.inner_eta_mu0 <= 0:
        raise ValueError("Inner algorithm initial step sizes must be positive.")
    if cfg.inner_max_iters <= 0 or cfg.greedy_max_iters <= 0 or cfg.search_max_iters <= 0:
        raise ValueError("Algorithm iteration limits must be positive integers.")
    if cfg.inner_tol <= 0 or cfg.rne_root_tol <= 0:
        raise ValueError("Algorithm tolerances must be positive.")
    if cfg.rne_directions <= 0 or cfg.rne_max_expand_steps <= 0:
        raise ValueError("RNE sampling settings must be positive integers.")
    if cfg.search_improvement_tol < 0:
        raise ValueError("search_improvement_tol must be non-negative.")
    return cfg


def _parse_baselines(raw: dict[str, Any], stack: StackelbergConfig, seed: int) -> BaselineConfig:
    cfg = BaselineConfig(
        enabled=bool(raw.get("enabled", False)),
        random_seed=int(raw.get("random_seed", seed + 17)),
        exact_max_users=int(raw.get("exact_max_users", 16)),
        stage2_solver_for_pricing=str(raw.get("stage2_solver_for_pricing", "DG")).upper(),
        max_price_E=float(raw.get("max_price_E", 8.0)),
        max_price_N=float(raw.get("max_price_N", 8.0)),
        gso_grid_points=int(raw.get("gso_grid_points", 25)),
        pbdr_grid_points=int(raw.get("pbdr_grid_points", 31)),
        pbdr_max_iters=int(raw.get("pbdr_max_iters", 30)),
        pbdr_tol=float(raw.get("pbdr_tol", 1e-4)),
        bo_init_points=int(raw.get("bo_init_points", 12)),
        bo_iters=int(raw.get("bo_iters", 40)),
        bo_candidate_pool=int(raw.get("bo_candidate_pool", 96)),
        bo_kernel_bandwidth=float(raw.get("bo_kernel_bandwidth", 0.25)),
        bo_ucb_beta=float(raw.get("bo_ucb_beta", 2.5)),
        drl_price_levels=int(raw.get("drl_price_levels", 15)),
        drl_episodes=int(raw.get("drl_episodes", 120)),
        drl_steps_per_episode=int(raw.get("drl_steps_per_episode", 40)),
        drl_alpha=float(raw.get("drl_alpha", 0.1)),
        drl_gamma=float(raw.get("drl_gamma", 0.95)),
        drl_epsilon=float(raw.get("drl_epsilon", 0.2)),
        market_max_iters=int(raw.get("market_max_iters", 80)),
        market_step_size=float(raw.get("market_step_size", 0.2)),
        single_sp_max_iters=int(raw.get("single_sp_max_iters", 200)),
        random_offloading_trials=int(raw.get("random_offloading_trials", 64)),
        random_offloading_prob=float(raw.get("random_offloading_prob", 0.5)),
        ubrd_max_iters=int(raw.get("ubrd_max_iters", 200)),
        vi_max_iters=int(raw.get("vi_max_iters", 200)),
        vi_step_size=float(raw.get("vi_step_size", 0.5)),
        vi_tol=float(raw.get("vi_tol", 1e-5)),
        penalty_outer_iters=int(raw.get("penalty_outer_iters", 8)),
        penalty_inner_iters=int(raw.get("penalty_inner_iters", 50)),
        penalty_init_rho=float(raw.get("penalty_init_rho", 0.1)),
        penalty_rho_scale=float(raw.get("penalty_rho_scale", 4.0)),
        penalty_tol=float(raw.get("penalty_tol", 1e-4)),
    )
    if cfg.stage2_solver_for_pricing not in {"CS", "UBRD", "VI", "PEN", "DG"}:
        raise ValueError("stage2_solver_for_pricing must be one of CS/UBRD/VI/PEN/DG.")
    if cfg.exact_max_users <= 0:
        raise ValueError("exact_max_users must be positive.")
    if cfg.max_price_E <= stack.initial_pE or cfg.max_price_N <= stack.initial_pN:
        # Keep this soft but safe for pricing search.
        raise ValueError("max_price_E/max_price_N should be greater than initial prices.")
    if cfg.random_offloading_trials <= 0:
        raise ValueError("random_offloading_trials must be positive.")
    if not (0 < cfg.random_offloading_prob < 1):
        raise ValueError("random_offloading_prob must be in (0,1).")
    if cfg.vi_max_iters <= 0 or cfg.penalty_outer_iters <= 0 or cfg.penalty_inner_iters <= 0:
        raise ValueError("VI/Penalty iteration limits must be positive.")
    if cfg.vi_step_size <= 0 or cfg.vi_tol <= 0 or cfg.penalty_tol <= 0:
        raise ValueError("VI/Penalty tolerances and step sizes must be positive.")
    if cfg.penalty_init_rho <= 0 or cfg.penalty_rho_scale <= 1:
        raise ValueError("Penalty rho settings must satisfy init_rho > 0 and rho_scale > 1.")
    return cfg


def _parse_detailed_experiment(raw: dict[str, Any]) -> DetailedExperimentConfig:
    cfg = DetailedExperimentConfig(
        emit_plan=bool(raw.get("emit_plan", True)),
        output_subdir=str(raw.get("output_subdir", "detailed_plan")),
        suggested_trials=int(raw.get("suggested_trials", 100)),
        heavy_trials=int(raw.get("heavy_trials", 200)),
    )
    if cfg.suggested_trials <= 0 or cfg.heavy_trials <= 0:
        raise ValueError("detailed_experiment trial counts must be positive.")
    return cfg


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))

    user_raw = data["user_distributions"]
    users = UserDistributions(
        w=_parse_distribution(user_raw["w"]),
        d=_parse_distribution(user_raw["d"]),
        fl=_parse_distribution(user_raw["fl"]),
        alpha=_parse_distribution(user_raw["alpha"]),
        beta=_parse_distribution(user_raw["beta"]),
        rho=_parse_distribution(user_raw["rho"]),
        varpi=_parse_distribution(user_raw["varpi"]),
        kappa=_parse_distribution(user_raw["kappa"]),
        sigma=_parse_distribution(user_raw["sigma"]),
    )

    pE = _parse_price_axis(data["price_grid"]["pE"], "price_grid.pE")
    pN = _parse_price_axis(data["price_grid"]["pN"], "price_grid.pN")
    system = _parse_system(data.get("system", {}))
    stackelberg = _parse_stackelberg(data.get("stackelberg", {}))
    seed = int(data.get("seed", 2026))
    baselines = _parse_baselines(data.get("baselines", {}), stackelberg, seed)
    detailed_experiment = _parse_detailed_experiment(data.get("detailed_experiment", {}))

    return ExperimentConfig(
        run_name=str(data.get("run_name", "default_run")),
        output_dir=str(data.get("output_dir", "outputs")),
        n_users=int(data.get("n_users", 100)),
        n_trials=int(data.get("n_trials", 5)),
        seed=seed,
        pE=pE,
        pN=pN,
        metrics=[str(x) for x in data.get("metrics", [])],
        users=users,
        system=system,
        stackelberg=stackelberg,
        baselines=baselines,
        detailed_experiment=detailed_experiment,
    )
