# CLAUDE.md - Agent Guidelines

For detailed information about the paper and code, read:
- `TMC26_Stackelberg.tex` - Primary source of truth for the latest paper
- `docs/SPEC.md` - Experiment specification aligned to the latest paper
- `docs/DEV.md` - Implementation-oriented notes mapped to the latest paper

## Behavior Rules

1. **Paper first** - If any script name, config key, or note conflicts with `TMC26_Stackelberg.tex`, follow the paper.

2. **Use SPEC and DEV as supporting docs** - Treat `docs/SPEC.md` and `docs/DEV.md` as secondary documents that must stay consistent with the paper.

3. **Read before changing** - Understand the relevant paper section before implementing code or rewriting experiment logic.

4. **Surface legacy naming explicitly** - Some scripts or config keys may still use older names. Do not silently adopt those names as paper terminology.

5. **Ask the User for Help** - When the paper and the code conflict in a way that cannot be resolved safely from local context, ask the user.
