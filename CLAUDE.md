# CLAUDE.md - Agent Guidelines

For detailed information about the paper and code, read:
- `TMC26_Stackelberg.tex` - Full paper with mathematical derivations
- `README.md` - Paper summary and code usage instructions
- `SPEC.md` - Specification of experiment objectives, core claims to validate, and experiment design.
- `PROGRESS.md` - Log of changes and progress (read before starting new work)

## Behavior Rules

1. **Follow SPEC.md** - Adhere to the specification in `SPEC.md` for all experiments.

2. **Use scripts files over ad hoc scripts** - Use one script for one experiment figure (`suites/*.toml`)

3. **Don't modify core algorithms** - Unless fixing a confirmed bug:
   - `src/tmc26_exp/stackelberg.py`
   - `src/tmc26_exp/model.py`
   - `src/tmc26_exp/baselines.py`

4. **Be explicit about limitations** - Suite runner only supports Stage-II methods (`DG`, `CS`, `UBRD`, `VI`, `PEN`). State if request exceeds this.

5. **Use DG as default** - Recommended Stage II method for general use

6. **Avoid CS for large n** - Centralized solver only works for n ≤ 15

7. **Update Your Progess** - Summarize and update your progress in `PROGRESS.md`, follow the format in the file.