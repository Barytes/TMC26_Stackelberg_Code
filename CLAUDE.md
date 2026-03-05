# CLAUDE.md - Agent Guidelines

For detailed information about the paper and code, read:
- `TMC26_Stackelberg.tex` - Full paper with mathematical derivations
- `README.md` - Paper summary and code usage instructions
- `PROGRESS.md` - Log of changes and progress (read before starting new work)

## Behavior Rules

1. **Prefer suite files over ad hoc scripts** - Use TOML suite files for experiments (`suites/*.toml`)

2. **Don't modify core algorithms** - Unless fixing a confirmed bug:
   - `src/tmc26_exp/stackelberg.py`
   - `src/tmc26_exp/model.py`
   - `src/tmc26_exp/baselines.py`

3. **Be explicit about limitations** - Suite runner only supports Stage-II methods (`DG`, `CS`, `UBRD`, `VI`, `PEN`). State if request exceeds this.

4. **Use DG as default** - Recommended Stage II method for general use

5. **Avoid CS for large n** - Centralized solver only works for n ≤ 15

## Suite Runner

```bash
uv run python scripts/run_suite.py --config configs/default.toml --suite suites/<name>.toml
```

Output: `outputs/suites/<experiment.name>/results.csv`, `summary.csv`