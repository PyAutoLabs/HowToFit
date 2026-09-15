# HowToFit

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PyAutoLabs/HowToFit/blob/main/start_here.ipynb)

[Start Here on Colab](https://colab.research.google.com/github/PyAutoLabs/HowToFit/blob/main/start_here.ipynb) |
[Installation Guide](https://pyautofit.readthedocs.io/en/latest/installation/overview.html) |
[PyAutoFit readthedocs](https://pyautofit.readthedocs.io/en/latest/index.html) |
[Browse Chapter 1 With Images](markdown/README.md) |
[autofit_workspace](https://github.com/PyAutoLabs/autofit_workspace)

<img src="https://github.com/Jammy2211/PyAutoLogo/blob/main/gifs/pyautofit.gif?raw=true" width="900" />

Welcome to **HowToFit**, the tutorial lecture series for [PyAutoFit](https://github.com/PyAutoLabs/PyAutoFit),
an open-source framework for scientific inference.

**PyAutoFit** is designed so scientists can bring their models, data and likelihood code, then fit models,
explore results and develop analyses **using natural language** with an AI coding agent. **HowToFit** teaches
the core principles behind this workflow, so you understand the inference being performed rather than
treating it as a black box.

The tutorials assume minimal prior knowledge of statistics and begin from first principles: models, priors,
likelihood functions and non-linear searches. They then progress to model comparison, graphical models and
hierarchical inference across large datasets.

With these foundations in place, **PyAutoFit** can be used through its natural-language workflow to compose
models, choose searches, perform inference and interpret results conversationally.

For experienced scientists who already know these concepts, the
[natural-language inference page](https://pyautofit.readthedocs.io/en/latest/overview/natural_language.html)
may be the better starting point: it walks a complete fit through **PyAutoFit** as a conversation with an AI
coding agent, assuming the principles taught in **HowToFit** as background.

## Chapters

- `chapter_1_introduction` — Models, likelihoods, non-linear searches, why modeling is hard, and how to
  interpret the results of a fit, ending with a short guide to building a scientific workflow.
- `chapter_advanced` — Fitting many datasets simultaneously with graphical models,
  hierarchical models, and Expectation Propagation.

Each chapter is a folder of numbered tutorial files — `tutorial_<M>_<topic>.py` (Python script) or the
matching `.ipynb` in `notebooks/`. Tutorials build on each other within a chapter and assume you have
completed the earlier ones.

## Getting Started

### Study with the assistant

Use the [Jupyter notebooks](notebooks/) if you want to run the code (recommended), or read the
available [Markdown lectures](markdown/README.md) directly on GitHub.

For help alongside the lectures, open the [autofit_assistant](https://github.com/PyAutoLabs/autofit_assistant)
repository in your AI coding agent, following its setup instructions, and paste:

```text
Enter HowToFit mode.

I want to work through the HowToFit lectures. Show me where to find them
and how to use Jupyter Notebook or Markdown, then help me with questions
as I go.
```

The assistant will answer questions about concepts, equations, code and results as you study, and help with
notebook errors. Share the lecture link and section or the cell you are working on; you choose when to move on.

### Run in Google Colab (nothing to install)

Every tutorial opens in Google Colab in one click. There is nothing to install and no local Python
environment to set up — **PyAutoFit** installs itself in the notebook's first cell. In Colab you *run*
the tutorial: edit the code, change the model, and see the output for yourself.

Whilst in Colab, we recommend opening **Gemini** — the button at the bottom of the notebook — and using it
as a study assistant alongside the lecture. It can see the notebook you have open, so you can ask it to
explain an equation, unpack what a cell is doing, or interpret the output of a fit, without leaving the
tutorial.

The `markdown` links are the same tutorial already executed and rendered on GitHub, with its real
output figures inline. Nothing runs and nothing installs — you just read it. They are good for
skimming a tutorial before running it, or for reading on a phone. Markdown pages currently exist only
for the chapter 1 tutorials listed with a `markdown` link below; every other tutorial is Colab-only.

**[Start Here](https://colab.research.google.com/github/PyAutoLabs/HowToFit/blob/main/start_here.ipynb)** — a one-page overview of the whole series.

- **[Chapter 1: Introduction](scripts/chapter_1_introduction/README.md)** — Models, likelihoods, non-linear searches, why modeling is hard, and how to interpret the results of a fit, ending with a short guide to building a scientific workflow.
  - Start Here: HowToFit Lectures — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToFit/blob/main/notebooks/chapter_1_introduction/start_here.ipynb) / [markdown](markdown/chapter_1_introduction/start_here.md))
  - Tutorial 1: Models — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToFit/blob/main/notebooks/chapter_1_introduction/tutorial_1_models.ipynb) / [markdown](markdown/chapter_1_introduction/tutorial_1_models.md))
  - Tutorial 2: Fitting Data — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToFit/blob/main/notebooks/chapter_1_introduction/tutorial_2_fitting_data.ipynb) / [markdown](markdown/chapter_1_introduction/tutorial_2_fitting_data.md))
  - Tutorial 3: Non Linear Search — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToFit/blob/main/notebooks/chapter_1_introduction/tutorial_3_non_linear_search.ipynb) / [markdown](markdown/chapter_1_introduction/tutorial_3_non_linear_search.md))
  - Tutorial 4: Why Modeling Is Hard — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToFit/blob/main/notebooks/chapter_1_introduction/tutorial_4_why_modeling_is_hard.ipynb) / [markdown](markdown/chapter_1_introduction/tutorial_4_why_modeling_is_hard.md))
  - Tutorial 5: Results and Samples — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToFit/blob/main/notebooks/chapter_1_introduction/tutorial_5_results_and_samples.ipynb) / [markdown](markdown/chapter_1_introduction/tutorial_5_results_and_samples.md))
  - Tutorial 6: Gradients — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToFit/blob/main/notebooks/chapter_1_introduction/tutorial_6_gradients.ipynb))
  - Tutorial 7: The Details — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToFit/blob/main/notebooks/chapter_1_introduction/tutorial_7_the_details.ipynb))
  - Tutorial 8: Scientific Workflow — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToFit/blob/main/notebooks/chapter_1_introduction/tutorial_8_scientific_workflow.ipynb) / [markdown](markdown/chapter_1_introduction/tutorial_8_scientific_workflow.md))
  - Tutorial Optional: Bayesian Formalism — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToFit/blob/main/notebooks/chapter_1_introduction/tutorial_optional_bayesian_formalism.ipynb))
- **[Advanced Chapter: Graphical & Hierarchical Models](scripts/chapter_advanced/README.md)** — Fitting many datasets simultaneously with graphical models, hierarchical models, and Expectation Propagation.
  - Tutorial 1: Individual Models — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToFit/blob/main/notebooks/chapter_advanced/tutorial_1_individual_models.ipynb))
  - Tutorial 2: Graphical Model — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToFit/blob/main/notebooks/chapter_advanced/tutorial_2_graphical_model.ipynb))
  - Tutorial 3: Graphical Benefits — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToFit/blob/main/notebooks/chapter_advanced/tutorial_3_graphical_benefits.ipynb))
  - Tutorial 4: Hierarchical Models — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToFit/blob/main/notebooks/chapter_advanced/tutorial_4_hierachical_models.ipynb))
  - Tutorial 5: Expectation Propagation — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToFit/blob/main/notebooks/chapter_advanced/tutorial_5_expectation_propagation.ipynb))
  - Tutorial Optional: Hierarchical Expectation Propagation — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToFit/blob/main/notebooks/chapter_advanced/tutorial_optional_hierarchical_ep.ipynb))
  - Tutorial Optional: Hierarchical Individual — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToFit/blob/main/notebooks/chapter_advanced/tutorial_optional_hierarchical_individual.ipynb))

Model-fits run considerably faster on a GPU. In Colab, enable one via *Runtime* → *Change runtime type*
→ *Hardware accelerator* before running a notebook.

### Run on your own machine

Follow the
[PyAutoFit installation guide](https://pyautofit.readthedocs.io/en/latest/installation/overview.html),
then clone this repository:

```bash
git clone https://github.com/PyAutoLabs/HowToFit.git
cd HowToFit
```

The tutorials are distributed as both Jupyter notebooks (`notebooks/`) and Python scripts (`scripts/`).
We recommend the notebooks for reading — figures render inline, and you can step through small code blocks
interactively. Use the Python scripts for actual **PyAutoFit** use.

## Before Chapter 1

Before starting chapter 1, open `start_here.py` for a one-page overview of the series, then begin
`scripts/chapter_1_introduction/tutorial_1_models.py`.

## Repository Structure

- `scripts/` — Runnable Python tutorial scripts, one subfolder per chapter.
- `notebooks/` — Jupyter notebook versions of the scripts (auto-generated; see below).
- `config/` — **PyAutoFit** configuration YAML files used by the tutorials.
- `dataset/` — Tutorial 1D datasets are generated at runtime by `scripts/simulators/simulators.py` —
  no data files are committed.
- `output/` — Model-fit results (generated at runtime, not committed).

## Notebooks vs Scripts

Notebooks in `notebooks/` are generated from the Python files in `scripts/`. **Always edit the ``.py``
scripts, never the notebooks directly.** The `# %%` markers in each script alternate between code and
markdown cells, which [PyAutoHands](https://github.com/PyAutoLabs/PyAutoHands) uses to produce the
`.ipynb` files.

## Relationship to autofit_workspace

[autofit_workspace](https://github.com/PyAutoLabs/autofit_workspace) is the main user-facing workspace
for **PyAutoFit** — concise examples, cookbooks, and search templates aimed at users who already understand
probabilistic modeling. **HowToFit** is the teaching companion. Tutorials in chapters 1 and 3 reference
`autofit_workspace` scripts as the next place to go after the relevant concept has been introduced.

## Citations

If you use **HowToFit** or **PyAutoFit** in your research, please cite the references listed in
`CITATIONS.rst`.

## Community & Support

Support for **PyAutoFit** is available via our Slack workspace. Slack is invitation-only; send an email
if you'd like an invite.

For installation issues, bug reports, or feature requests, raise an issue on the
[PyAutoFit GitHub issues page](https://github.com/PyAutoLabs/PyAutoFit/issues) (for library issues)
or the [HowToFit GitHub issues page](https://github.com/PyAutoLabs/HowToFit/issues) (for tutorial
content issues).
