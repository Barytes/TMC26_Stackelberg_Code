# Multi-Provider Experiment Scripts

This folder contains standalone scripts for the multi-ESP, multi-NSP extension experiment.

Run a default 3-ESP, 3-NSP instance:

```bash
UV_CACHE_DIR=/private/tmp/tmc26-uv-cache \
UV_PROJECT_ENVIRONMENT=/private/tmp/tmc26-mp-venv \
uv run python scripts/multi_provider/run_multi_provider_solver.py --n-users 30 --num-esp 3 --num-nsp 3 --setup code_default_heterogeneous
```

The solver prints the output directory. Then plot the three companion figures:

```bash
UV_CACHE_DIR=/private/tmp/tmc26-uv-cache \
UV_PROJECT_ENVIRONMENT=/private/tmp/tmc26-mp-venv \
uv run python scripts/multi_provider/plot_multi_provider_results.py outputs/multi_provider/<run_dir>
```

Main outputs:

- `multi_provider_trajectory.csv`
- `multi_provider_assignment.csv`
- `multi_provider_provider_metrics.csv`
- `multi_provider_assignment_heatmap.png`
- `multi_provider_convergence.png`
- `multi_provider_provider_metrics.png`
