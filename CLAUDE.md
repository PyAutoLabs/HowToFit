# HowToFit

This is the **HowToFit** tutorial lecture series for `PyAutoFit`, a Python probabilistic programming library for Bayesian model fitting. Tutorials teach new users what model fitting is and how to compose, fit, and interpret probabilistic models from first principles.

## Repository Structure

- `scripts/` — Runnable Python tutorial scripts
  - `chapter_1_introduction/` — Models, fitting data, non-linear searches, why modeling is hard, results and samples
  - `chapter_2_scientific_workflow/` — Reserved stub for future material (empty except for README)
  - `chapter_3_graphical_models/` — Individual models, graphical models, hierarchical models, Expectation Propagation
  - `simulators/` — Simulator scripts that generate the tutorial 1D datasets at runtime
- `notebooks/` — Jupyter notebook versions of scripts (generated from `scripts/`, do not edit directly)
- `config/` — `PyAutoFit` configuration YAML files
- `dataset/` — Empty in the repo; tutorial datasets are written here at runtime by the simulator scripts
- `output/` — Model-fit results (generated at runtime, not committed)

## Running Scripts

Scripts are run from the repository root so relative paths to `dataset/` and `output/` resolve correctly:

```bash
python scripts/chapter_1_introduction/tutorial_1_models.py
```

Tutorials that need a dataset invoke `scripts/simulators/simulators.py` via `subprocess` if the dataset folder does not already exist — there is no manual simulate-then-run step.

**Integration testing / fast mode**: set `PYAUTO_TEST_MODE=1` to skip non-linear search sampling:

```bash
PYAUTO_TEST_MODE=1 python scripts/chapter_1_introduction/tutorial_3_non_linear_search.py
```

**Fast smoke tests**: combine test mode with the skip flags:

```bash
PYAUTO_TEST_MODE=2 PYAUTO_SKIP_FIT_OUTPUT=1 PYAUTO_SKIP_VISUALIZATION=1 PYAUTO_SKIP_CHECKS=1 python scripts/chapter_1_introduction/tutorial_3_non_linear_search.py
```

**Codex / sandboxed runs**: set writable cache directories so `numba` and `matplotlib` do not fail on unwritable home paths:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python scripts/chapter_1_introduction/tutorial_1_models.py
```

## Core API Patterns

Imports used throughout the tutorials:

```python
import autofit as af
```

Most tutorials use the `af.ex` namespace for example model components (e.g. `af.ex.Gaussian`, `af.ex.Analysis`) and the generic `af.Model` / `af.Collection` API for model composition.

## Notebooks vs Scripts

Notebooks in `notebooks/` are generated from the `.py` files in `scripts/` using `generate.py` from the `PyAutoBuild` repo. **Always edit the `.py` scripts**, never the notebooks directly. The `# %%` marker alternates between code and markdown cells.

### Building Notebooks

Run from the workspace root:

```bash
PYTHONPATH=../PyAutoBuild/autobuild python3 ../PyAutoBuild/autobuild/generate.py howtofit
```

The `howtofit` project target in `PyAutoBuild/autobuild/config/` is what drives this.

## Relationship to autofit_workspace

HowToFit is the teaching companion to `autofit_workspace`. Several tutorials point users to `autofit_workspace` scripts (e.g. `scripts/cookbooks/...`, `scripts/overview/overview_2_science_workflow.py`, `scripts/searches/...`) as the next destination after the relevant concept has been introduced. Those cross-references use absolute paths like `autofit_workspace/scripts/...` and refer to the separate `autofit_workspace` repository — not to anything inside HowToFit.

## Related Repos

- **PyAutoFit** source: `../PyAutoFit`
- **autofit_workspace**: `../autofit_workspace` — main user-facing workspace
- **PyAutoBuild**: `../PyAutoBuild` — notebook generation and CI/CD tooling
## Never rewrite history

NEVER perform these operations on any repo with a remote:

- `git init` in a directory already tracked by git
- `rm -rf .git && git init`
- Commit with subject "Initial commit", "Fresh start", "Start fresh", "Reset
  for AI workflow", or any equivalent message on a branch with a remote
- `git push --force` to `main` (or any branch tracked as `origin/HEAD`)
- `git filter-repo` / `git filter-branch` on shared branches
- `git rebase -i` rewriting commits already pushed to a shared branch

If the working tree needs a clean state, the **only** correct sequence is:

    git fetch origin
    git reset --hard origin/main
    git clean -fd

This applies equally to humans, local Claude Code, cloud Claude agents, Codex,
and any other agent. The "Initial commit — fresh start for AI workflow" pattern
that appeared independently on origin and local for three workspace repos is
exactly what this rule prevents — it costs ~40 commits of redundant local work
every time it happens.
