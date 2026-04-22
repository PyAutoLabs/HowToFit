HowToFit
========

`Installation Guide <https://pyautofit.readthedocs.io/en/latest/installation/overview.html>`_ |
`PyAutoFit readthedocs <https://pyautofit.readthedocs.io/en/latest/index.html>`_ |
`autofit_workspace <https://github.com/PyAutoLabs/autofit_workspace>`_

.. image:: https://github.com/Jammy2211/PyAutoLogo/blob/main/gifs/pyautofit.gif?raw=true
  :width: 900

Welcome to **HowToFit** — the tutorial lecture series for `PyAutoFit <https://github.com/PyAutoLabs/PyAutoFit>`_,
an open-source probabilistic programming library for Bayesian model fitting.

**HowToFit** teaches new users how to compose, fit, and interpret probabilistic models with **PyAutoFit**.
It assumes minimal prior knowledge of statistics and takes you from first principles — what a model is,
what a non-linear search does, how a likelihood function is built — through to graphical and hierarchical
models for fitting large datasets simultaneously.

For experienced scientists who already know the fundamentals of Bayesian modeling, the
`autofit_workspace <https://github.com/PyAutoLabs/autofit_workspace>`_ examples will be more appropriate —
they are concise, API-focused, and assume the concepts taught in **HowToFit** as background.

Chapters
--------

- ``chapter_1_introduction`` — Models, likelihoods, non-linear searches, why modeling is hard, and how to
  interpret the results of a fit.
- ``chapter_2_scientific_workflow`` — Reserved for future material on building a full scientific
  model-fitting workflow. Currently a stub; the corresponding overview lives in
  ``autofit_workspace/scripts/overview/overview_2_science_workflow.py``.
- ``chapter_3_graphical_models`` — Fitting many datasets simultaneously with graphical models,
  hierarchical models, and Expectation Propagation.

Each chapter is organised into numbered tutorial files: ``chapter_<N>_<name>/tutorial_<M>_<topic>.py``
(Python script) or the matching ``.ipynb`` in ``notebooks/``. Tutorials build on each other within a
chapter and assume you have completed the earlier ones.

Getting Started
---------------

You can run the tutorials on your own machine by following the
`PyAutoFit installation guide <https://pyautofit.readthedocs.io/en/latest/installation/overview.html>`_,
then cloning this repository:

.. code-block:: bash

    git clone https://github.com/PyAutoLabs/HowToFit.git
    cd HowToFit

Alternatively, every tutorial can be opened directly in Google Colab via the links in each chapter's
``README.rst``.

The tutorials are distributed as both Jupyter notebooks (``notebooks/``) and Python scripts (``scripts/``).
We recommend the notebooks for reading — figures render inline, and you can step through small code blocks
interactively. Use the Python scripts for actual **PyAutoFit** use.

Before Chapter 1
----------------

Before starting chapter 1, open ``start_here.py`` for a one-page overview of the series, then begin
``scripts/chapter_1_introduction/tutorial_1_models.py``.

Repository Structure
--------------------

- ``scripts/`` — Runnable Python tutorial scripts, one subfolder per chapter.
- ``notebooks/`` — Jupyter notebook versions of the scripts (auto-generated; see below).
- ``config/`` — **PyAutoFit** configuration YAML files used by the tutorials.
- ``dataset/`` — Tutorial 1D datasets are generated at runtime by ``scripts/simulators/simulators.py`` —
  no data files are committed.
- ``output/`` — Model-fit results (generated at runtime, not committed).

Notebooks vs Scripts
--------------------

Notebooks in ``notebooks/`` are generated from the Python files in ``scripts/``. **Always edit the ``.py``
scripts, never the notebooks directly.** The ``# %%`` markers in each script alternate between code and
markdown cells, which `PyAutoBuild <https://github.com/PyAutoLabs/PyAutoBuild>`_ uses to produce the
``.ipynb`` files.

Relationship to autofit_workspace
---------------------------------

`autofit_workspace <https://github.com/PyAutoLabs/autofit_workspace>`_ is the main user-facing workspace
for **PyAutoFit** — concise examples, cookbooks, and search templates aimed at users who already understand
probabilistic modeling. **HowToFit** is the teaching companion. Tutorials in chapters 1 and 3 reference
``autofit_workspace`` scripts as the next place to go after the relevant concept has been introduced.

Citations
---------

If you use **HowToFit** or **PyAutoFit** in your research, please cite the references listed in
``CITATIONS.rst``.

Community & Support
-------------------

Support for **PyAutoFit** is available via our Slack workspace. Slack is invitation-only; send an email
if you'd like an invite.

For installation issues, bug reports, or feature requests, raise an issue on the
`PyAutoFit GitHub issues page <https://github.com/PyAutoLabs/PyAutoFit/issues>`_ (for library issues)
or the `HowToFit GitHub issues page <https://github.com/PyAutoLabs/HowToFit/issues>`_ (for tutorial
content issues).
