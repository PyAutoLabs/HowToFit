The ``scripts`` folder contains **HowToFit** lectures, which teach a new user how to compose and fit models in **PyAutoFit**.

Folders
-------

- ``chapter_1_introduction``: Introduction lectures describing how to compose and fit models in **PyAutoFit**.
- ``chapter_2_scientific_workflow``: Reserved stub for future material on building a full scientific model-fitting workflow. The equivalent overview currently lives in ``autofit_workspace/scripts/overview/overview_2_science_workflow.py``.
- ``chapter_3_graphical_models``: How to compose and fit graphical models which fit many datasets simultaneously, including hierarchical models and Expectation Propagation.
- ``simulators``: Simulator scripts that generate the tutorial 1D datasets at runtime.

Jupyter Notebooks
-----------------

The tutorials are supplied as *Jupyter notebooks*, which come with a ``.ipynb`` suffix. For those new to
Python, *Jupyter notebooks* are a different way to write, view and use Python code. Compared to traditional
Python scripts, they allow:

- Small blocks of code to be viewed and run at a time
- Images and visualization from a code block to be displayed directly underneath it.
- Text script to appear between the blocks of code.

This makes them an ideal way for us to present the **HowToFit** lecture series. The notebooks live under
``notebooks/`` and are auto-generated from the Python scripts in ``scripts/`` — **edit the Python scripts,
not the notebooks.**

For actual **PyAutoFit** use we recommend the Python scripts. Chapter 3 onwards assumes that transition.

Code Style and Formatting
-------------------------

You may notice the style and formatting of our Python code looks different to what you are used to. For
example, it is common for brackets to be placed on their own line at the end of function calls, and the
inputs of a function or class may be listed over many separate lines.

This is intentional, because we believe it makes the cleanest, most readable code possible. The codebase
is auto-formatted with ``black`` — see https://github.com/python/black.

How to Approach HowToFit
------------------------

**HowToFit** currently consists of three chapters (chapter 2 is a stub). Chapter 1 will take a couple of
hours to work through. The concepts in chapter 3 are challenging, and familiarity with **PyAutoFit** and
model fitting is desirable before tackling them.

We recommend that you complete chapter 1 and then apply what you've learnt to a model-fitting problem you
are interested in, building on the scripts found in the
`autofit_workspace <https://github.com/PyAutoLabs/autofit_workspace>`_ repository. Once you're confident
with your use of **PyAutoFit**, return for chapter 3.

Overview of Chapter 1 (Beginner)
--------------------------------

**Model Fitting with PyAutoFit**

In chapter 1, we'll learn about model composition and fitting with **PyAutoFit**. By the end, you'll be
able to:

1) Compose a model in **PyAutoFit**.
2) Define a ``log_likelihood_function()`` via an ``Analysis`` class to fit that model to data.
3) Understand the concept of a non-linear search and non-linear parameter space.
4) Fit a model to data using a non-linear search.
5) Compose and fit more complex models using **PyAutoFit**'s model composition API.
6) Analyse the results of a model-fit, including parameter estimates and errors.

Overview of Chapter 3 (Advanced)
--------------------------------

**Fitting Graphical Models to Large Datasets**

Chapter 3 covers how to compose and fit graphical models to extremely large datasets. You'll learn:

1) Why fitting a model to many datasets one-by-one is suboptimal.
2) How to fit a graphical model to all datasets simultaneously and why this improves the model results.
3) How to scale graphical model fits up to extremely large datasets via Expectation Propagation.
4) How to fit a hierarchical model using the graphical modeling framework and Expectation Propagation.
