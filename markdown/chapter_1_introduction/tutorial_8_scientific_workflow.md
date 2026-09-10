> ✏️ **This page is auto-generated from [`scripts/chapter_1_introduction/tutorial_8_scientific_workflow.py`](../../scripts/chapter_1_introduction/tutorial_8_scientific_workflow.py) — do not edit it directly.**
> It shows the example fully executed, with its real output images.
> Run it yourself via the [Python script](../../scripts/chapter_1_introduction/tutorial_8_scientific_workflow.py) or the [Jupyter notebook](../../notebooks/chapter_1_introduction/tutorial_8_scientific_workflow.ipynb).

Tutorial 8: Scientific Workflow
===============================

You can now compose a model, fit it to data and interpret its results. A scientific study often repeats
these steps for many datasets, competing models and different non-linear searches. The next challenge is
keeping those fits organized so that you can understand and compare them.

PyAutoFit saves each fit's model, search settings, samples and summaries alongside its visualizations.
These outputs let you revisit a fit without rerunning inference, record quantities specific to your
science, and trace a comparison back to its original results. Live progress and model-fit plots also
help you develop intuition for how inference is progressing while a search runs.

Imagine five datasets, each fitted with several models and searches. A scientific workflow makes that
collection navigable: you can inspect the folders yourself or ask an assistant to compare parameter
constraints, fit quality and runtime, and identify results that need closer attention. Model composition
provides the competing models; the workflow makes their results feasible to interpret together.

Continue with the [Scientific Workflow guide](https://pyautofit.readthedocs.io/en/latest/overview/scientific_workflow.html)
for natural language prompts that build this workflow with an assistant.

For the corresponding Python implementation, follow the
[workspace scientific workflow example](https://github.com/PyAutoLabs/autofit_workspace/blob/main/scripts/overview/overview_2_scientific_workflow.py).

__Wrap Up__

Apply the workflow to your own data before returning to chapter 3 for graphical and hierarchical models.
