"""
Start Here: HowToFit
====================

Welcome to **HowToFit** — the `PyAutoFit` tutorial lecture series on probabilistic model fitting
and Bayesian inference.

This script gives you a one-page overview of the series and points you to the first tutorial.

__HowToFit__

**HowToFit** is a three-chapter guide which takes you from knowing nothing about model fitting
to being able to compose, fit, and interpret complex graphical and hierarchical models with
**PyAutoFit** for scientific research.

- Chapter 1: Introduction — what a model is, how we fit data, how non-linear searches work,
  why model fitting is hard, and how to inspect the results.
- Chapter 2: Scientific workflow — placeholder for future material. For now, the equivalent
  overview lives in ``autofit_workspace/scripts/overview/overview_2_science_workflow.py``.
- Chapter 3: Graphical models — fitting many datasets simultaneously with graphical models,
  hierarchical models, and Expectation Propagation.

Each chapter is organised into numbered tutorial files: `chapter_<N>_<name>/tutorial_<M>_<topic>.py`
(Python script) or the matching `.ipynb` in `notebooks/`. Tutorials build on each other within a
chapter and assume you have completed the earlier ones.

__Recommended Path__

We recommend working through the tutorials in order:

1. Start with `scripts/chapter_1_introduction/start_here.py`, then `tutorial_1_models.py` and
   continue through the rest of chapter 1.
2. At this point you can start applying **PyAutoFit** to your own data using scripts in the
   separate `autofit_workspace` repository — you know enough to be productive.
3. Come back for `chapter_3_graphical_models` when you need to fit many datasets simultaneously
   or build hierarchical inference pipelines.

__Notebooks vs Scripts__

Every tutorial exists as both a Python script (under `scripts/`) and a Jupyter notebook (under
`notebooks/`). The notebooks are ideal for reading because plots render inline between small
blocks of code. The Python scripts are more convenient for actual **PyAutoFit** use.

The notebooks are auto-generated from the Python scripts, so if you want to make changes, **edit
the Python scripts, not the notebooks.**

__Next Step__

Open `scripts/chapter_1_introduction/start_here.py` (or the `.ipynb` equivalent) and start there.
"""

print(__doc__)
