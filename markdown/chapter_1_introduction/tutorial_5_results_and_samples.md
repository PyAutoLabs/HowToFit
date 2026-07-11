> ✏️ **This page is auto-generated from [`scripts/chapter_1_introduction/tutorial_5_results_and_samples.py`](../../scripts/chapter_1_introduction/tutorial_5_results_and_samples.py) — do not edit it directly.**
> It shows the example fully executed, with its real output images.
> Run it yourself via the [Python script](../../scripts/chapter_1_introduction/tutorial_5_results_and_samples.py) or the [Jupyter notebook](../../notebooks/chapter_1_introduction/tutorial_5_results_and_samples.ipynb).

Tutorial 5: Results And Samples
===============================

In this tutorial, we'll cover all of the output that comes from a non-linear search's `Result`  object.

We used this object at various points in the chapter. The bulk of material covered here is described in the example
script `autofit_workspace/overview/simple/result.py`. Nevertheless, it is a good idea to refresh ourselves about how
results in **PyAutoFit** work before covering more advanced material.

__Contents__

This tutorial is split into the following sections:

- **Data**: Load the dataset from the HowToFit/dataset folder.
- **Reused Functions**: Reuse the `plot_profile_1d` and `Analysis` classes from the previous tutorial.
- **Model Fit**: Run a non-linear search to generate a `Result` object.
- **Result**: Examine the `Result` object and its info attribute.
- **Samples**: Introduce the `Samples` object containing the non-linear search samples.
- **Parameters**: Access parameter values from the samples.
- **Figures of Merit**: Examine log likelihood, log prior, and log posterior values.
- **Instances**: Return results as model instances from samples.
- **Vectors**: Return results as 1D parameter vectors.
- **Labels**: Access the paths, names, and labels for model parameters.
- **Posterior / PDF**: Access median PDF estimates for the model parameters.
- **Plot**: Visualize model fit results using instances.
- **Errors**: Compute parameter error estimates at specified sigma confidence limits.
- **PDF**: Plot Probability Density Functions using corner.py.
- **Other Results**: Access maximum log posterior and other sample statistics.
- **Sample Instance**: Create instances from individual samples in the sample list.
- **Bayesian Evidence**: Access the log evidence for nested sampling searches.
- **Derived Errors (PDF from samples)**: Compute errors on derived quantities from sample PDFs.
- **Samples Filtering**: Filter samples by parameter paths for specific parameter analysis.
- **Latex**: Generate LaTeX table code for modeling results.


```python

from autoconf import setup_notebook; setup_notebook()

import autofit as af
import autofit.plot as aplt
import os
from os import path
import numpy as np
import matplotlib.pyplot as plt
```

    Working Directory has been set to `HowToFit`


__Data__

Load the dataset from the `HowToFit/dataset` folder.


```python
dataset_path = path.join("dataset", "example_1d", "gaussian_x1__exponential_x1")
```

__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.


```python
if not path.exists(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/simulators/simulators.py"],
        check=True,
    )

data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)
```

__Reused Functions__

We'll reuse the `plot_profile_1d` and `Analysis` classes of the previous tutorial.


```python


def plot_profile_1d(
    xvalues,
    profile_1d,
    title=None,
    ylabel=None,
    errors=None,
    color="k",
    output_path=None,
    output_filename=None,
):
    plt.errorbar(
        x=xvalues,
        y=profile_1d,
        yerr=errors,
        linestyle="",
        color=color,
        ecolor="k",
        elinewidth=1,
        capsize=2,
    )
    plt.title(title)
    plt.xlabel("x value of profile")
    plt.ylabel(ylabel)
    if not path.exists(output_path):
        os.makedirs(output_path)
    plt.savefig(path.join(output_path, f"{output_filename}.png"))
    plt.clf()


class Analysis(af.Analysis):
    def __init__(self, data, noise_map):
        super().__init__()

        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance):
        model_data = self.model_data_from_instance(instance=instance)

        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        chi_squared = sum(chi_squared_map)
        noise_normalization = np.sum(np.log(2 * np.pi * noise_map**2.0))
        log_likelihood = -0.5 * (chi_squared + noise_normalization)

        return log_likelihood

    def model_data_from_instance(self, instance):
        """
        To create the summed profile of all individual profiles in an instance, we can use a dictionary comprehension
        to iterate over all profiles in the instance.
        """
        xvalues = np.arange(self.data.shape[0])

        return sum([profile.model_data_from(xvalues=xvalues) for profile in instance])

    def visualize(self, paths, instance, during_analysis):
        """
        This method is identical to the previous tutorial, except it now uses the `model_data_from_instance` method
        to create the profile.
        """
        xvalues = np.arange(self.data.shape[0])

        model_data = self.model_data_from_instance(instance=instance)

        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0

        """The visualizer now outputs images of the best-fit results to hard-disk (checkout `visualizer.py`)."""
        plot_profile_1d(
            xvalues=xvalues,
            profile_1d=self.data,
            title="Data",
            ylabel="Data Values",
            color="k",
            output_path=paths.image_path,
            output_filename="data",
        )

        plot_profile_1d(
            xvalues=xvalues,
            profile_1d=model_data,
            title="Model Data",
            ylabel="Model Data Values",
            color="k",
            output_path=paths.image_path,
            output_filename="model_data",
        )

        plot_profile_1d(
            xvalues=xvalues,
            profile_1d=residual_map,
            title="Residual Map",
            ylabel="Residuals",
            color="k",
            output_path=paths.image_path,
            output_filename="residual_map",
        )

        plot_profile_1d(
            xvalues=xvalues,
            profile_1d=chi_squared_map,
            title="Chi-Squared Map",
            ylabel="Chi-Squareds",
            color="k",
            output_path=paths.image_path,
            output_filename="chi_squared_map",
        )

```

__Model Fit__

Now lets run the non-linear search to get ourselves a `Result`.


```python


class Gaussian:
    def __init__(
        self,
        centre=30.0,  # <- **PyAutoFit** recognises these constructor arguments
        normalization=1.0,  # <- are the Gaussian`s model parameters.
        sigma=5.0,
    ):
        """
        Represents a 1D Gaussian profile.

        This is a model-component of example models in the **HowToFit** lectures and is used to fit example datasets
        via a non-linear search.

        Parameters
        ----------
        centre
            The x coordinate of the profile centre.
        normalization
            Overall normalization of the profile.
        sigma
            The sigma value controlling the size of the Gaussian.
        """
        self.centre = centre
        self.normalization = normalization
        self.sigma = sigma

    def model_data_from(self, xvalues: np.ndarray):
        """

        Returns a 1D Gaussian on an input list of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Gaussian, via its `centre`.

        The output is referred to as the `model_data` to signify that it is a representation of the data from the
        model.

        Parameters
        ----------
        xvalues
            The x coordinates in the original reference frame of the data.
        """
        transformed_xvalues = np.subtract(xvalues, self.centre)
        return np.multiply(
            np.divide(self.normalization, self.sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(transformed_xvalues, self.sigma))),
        )


class Exponential:
    def __init__(
        self,
        centre=30.0,  # <- **PyAutoFit** recognises these constructor arguments
        normalization=1.0,  # <- are the Exponential`s model parameters.
        rate=0.01,
    ):
        """
        Represents a 1D Exponential profile.

        This is a model-component of example models in the **HowToFit** lectures and is used to fit example datasets
        via a non-linear search.

        Parameters
        ----------
        centre
            The x coordinate of the profile centre.
        normalization
            Overall normalization of the profile.
        ratw
            The decay rate controlling has fast the Exponential declines.
        """
        self.centre = centre
        self.normalization = normalization
        self.rate = rate

    def model_data_from(self, xvalues: np.ndarray):
        """
        Returns a 1D Gaussian on an input list of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Gaussian, via its `centre`.

        The output is referred to as the `model_data` to signify that it is a representation of the data from the
        model.

        Parameters
        ----------
        xvalues
            The x coordinates in the original reference frame of the data.
        """
        transformed_xvalues = np.subtract(xvalues, self.centre)
        return self.normalization * np.multiply(
            self.rate, np.exp(-1.0 * self.rate * abs(transformed_xvalues))
        )


model = af.Collection(gaussian=af.Model(Gaussian), exponential=af.Model(Exponential))

analysis = Analysis(data=data, noise_map=noise_map)

search = af.Emcee(
    name="tutorial_5_results_and_samples",
    path_prefix="chapter_1_introduction",
)

print(
    """
    The non-linear search has begun running.
    Checkout the HowToFit/output/chapter_1_introduction/tutorial_5_results_and_samples
    folder for live output of the results.
    This Jupyter notebook cell with progress once the search has completed - this could take a few minutes!
    """
)

result = search.fit(model=model, analysis=analysis)

print("The search has finished run - you may now continue the notebook.")
```

    
        The non-linear search has begun running.
        Checkout the HowToFit/output/chapter_1_introduction/tutorial_5_results_and_samples
        folder for live output of the results.
        This Jupyter notebook cell with progress once the search has completed - this could take a few minutes!
        
    2026-07-11 16:29:32,999 - autofit.non_linear.search.abstract_search - INFO - Starting non-linear search with 1 cores.


    2026-07-11 16:29:33,011 - tutorial_5_results_and_samples - INFO - The output path of this fit is HowToFit/output/chapter_1_introduction/tutorial_5_results_and_samples/f7b96c133609a46d15a374506304464c


    2026-07-11 16:29:33,012 - tutorial_5_results_and_samples - INFO - Outputting pre-fit files (e.g. model.info, visualization).


    2026-07-11 16:29:33,575 - autofit.non_linear.initializer - INFO - Generating initial samples of model using JAX LH Function cores


    2026-07-11 16:29:33,597 - autofit.non_linear.initializer - INFO - Initial samples generated, starting non-linear search


    2026-07-11 16:29:33,598 - tutorial_5_results_and_samples - INFO - Visualizing Starting Point Model in image_start folder.


    2026-07-11 16:29:34,071 - tutorial_5_results_and_samples - INFO - Starting new Emcee non-linear search (no previous samples found).


      0%|          | 0/2000 [00:00<?, ?it/s]

    .../PyAutoFit/autofit/non_linear/fitness.py:299: RuntimeWarning: invalid value encountered in scalar subtract
      log_likelihood -= np.sum(log_prior_list)
      0%|          | 6/2000 [00:00<00:34, 57.91it/s]

      1%|          | 12/2000 [00:00<00:37, 52.79it/s]

      1%|          | 18/2000 [00:00<00:36, 54.87it/s]

      1%|▏         | 25/2000 [00:00<00:34, 58.00it/s]

      2%|▏         | 32/2000 [00:00<00:32, 60.45it/s]

      2%|▏         | 39/2000 [00:00<00:31, 62.40it/s]

      2%|▏         | 46/2000 [00:00<00:30, 63.09it/s]

      3%|▎         | 53/2000 [00:00<00:30, 62.83it/s]

      3%|▎         | 60/2000 [00:00<00:31, 60.73it/s]

      3%|▎         | 67/2000 [00:01<00:31, 61.10it/s]

      4%|▎         | 74/2000 [00:01<00:31, 61.79it/s]

      4%|▍         | 81/2000 [00:01<00:30, 63.32it/s]

      4%|▍         | 88/2000 [00:01<00:29, 64.94it/s]

      5%|▍         | 95/2000 [00:01<00:30, 62.83it/s]

      5%|▌         | 102/2000 [00:01<00:31, 60.33it/s]

      5%|▌         | 109/2000 [00:01<00:31, 59.30it/s]

      6%|▌         | 116/2000 [00:01<00:31, 60.31it/s]

      6%|▌         | 123/2000 [00:02<00:30, 60.97it/s]

      6%|▋         | 130/2000 [00:02<00:32, 57.28it/s]

      7%|▋         | 137/2000 [00:02<00:31, 59.18it/s]

      7%|▋         | 143/2000 [00:02<00:31, 58.85it/s]

      8%|▊         | 150/2000 [00:02<00:31, 59.25it/s]

      8%|▊         | 157/2000 [00:02<00:30, 60.24it/s]

      8%|▊         | 164/2000 [00:02<00:30, 61.00it/s]

      9%|▊         | 171/2000 [00:02<00:31, 57.78it/s]

      9%|▉         | 177/2000 [00:02<00:32, 56.67it/s]

      9%|▉         | 183/2000 [00:03<00:31, 57.34it/s]

     10%|▉         | 190/2000 [00:03<00:30, 58.48it/s]

     10%|▉         | 197/2000 [00:03<00:30, 59.20it/s]

     10%|█         | 204/2000 [00:03<00:29, 60.58it/s]

     11%|█         | 211/2000 [00:03<00:30, 59.58it/s]

     11%|█         | 218/2000 [00:03<00:29, 60.93it/s]

     11%|█▏        | 225/2000 [00:03<00:28, 61.58it/s]

     12%|█▏        | 232/2000 [00:03<00:29, 60.11it/s]

     12%|█▏        | 239/2000 [00:03<00:28, 61.03it/s]

     12%|█▏        | 246/2000 [00:04<00:29, 59.95it/s]

     13%|█▎        | 253/2000 [00:04<00:28, 60.50it/s]

     13%|█▎        | 260/2000 [00:04<00:28, 61.83it/s]

     13%|█▎        | 267/2000 [00:04<00:29, 58.38it/s]

     14%|█▎        | 274/2000 [00:04<00:28, 59.82it/s]

     14%|█▍        | 281/2000 [00:04<00:28, 60.38it/s]

     14%|█▍        | 288/2000 [00:04<00:29, 59.02it/s]

     15%|█▍        | 294/2000 [00:04<00:29, 57.36it/s]

     15%|█▌        | 300/2000 [00:05<00:29, 56.89it/s]

     15%|█▌        | 306/2000 [00:05<00:29, 57.05it/s]

     16%|█▌        | 312/2000 [00:05<00:31, 54.34it/s]

     16%|█▌        | 319/2000 [00:05<00:29, 56.67it/s]

     16%|█▋        | 326/2000 [00:05<00:28, 59.12it/s]

     17%|█▋        | 332/2000 [00:05<00:28, 58.85it/s]

     17%|█▋        | 338/2000 [00:05<00:28, 58.07it/s]

     17%|█▋        | 344/2000 [00:05<00:28, 58.07it/s]

     18%|█▊        | 351/2000 [00:05<00:27, 59.77it/s]

     18%|█▊        | 358/2000 [00:05<00:26, 61.57it/s]

     18%|█▊        | 365/2000 [00:06<00:25, 63.20it/s]

     19%|█▊        | 372/2000 [00:06<00:25, 64.83it/s]

     19%|█▉        | 380/2000 [00:06<00:24, 66.60it/s]

     19%|█▉        | 387/2000 [00:06<00:23, 67.24it/s]

     20%|█▉        | 395/2000 [00:06<00:23, 68.00it/s]

     20%|██        | 402/2000 [00:06<00:23, 66.79it/s]

     20%|██        | 409/2000 [00:06<00:24, 65.96it/s]

     21%|██        | 416/2000 [00:06<00:24, 65.67it/s]

     21%|██        | 423/2000 [00:06<00:23, 66.76it/s]

     22%|██▏       | 430/2000 [00:07<00:23, 65.95it/s]

     22%|██▏       | 438/2000 [00:07<00:23, 67.64it/s]

     22%|██▏       | 445/2000 [00:07<00:22, 67.76it/s]

     23%|██▎       | 453/2000 [00:07<00:22, 68.72it/s]

     23%|██▎       | 460/2000 [00:07<00:22, 68.75it/s]

     23%|██▎       | 467/2000 [00:07<00:22, 67.53it/s]

     24%|██▎       | 474/2000 [00:07<00:24, 61.88it/s]

     24%|██▍       | 481/2000 [00:07<00:25, 60.62it/s]

     24%|██▍       | 488/2000 [00:07<00:25, 60.07it/s]

     25%|██▍       | 495/2000 [00:08<00:25, 58.88it/s]

     25%|██▌       | 501/2000 [00:08<00:25, 58.23it/s]

     25%|██▌       | 507/2000 [00:08<00:25, 57.48it/s]

     26%|██▌       | 513/2000 [00:08<00:25, 57.40it/s]

     26%|██▌       | 520/2000 [00:08<00:25, 58.60it/s]

     26%|██▋       | 527/2000 [00:08<00:24, 60.76it/s]

     27%|██▋       | 535/2000 [00:08<00:22, 64.14it/s]

     27%|██▋       | 543/2000 [00:08<00:21, 66.81it/s]

     28%|██▊       | 551/2000 [00:08<00:21, 68.35it/s]

     28%|██▊       | 559/2000 [00:09<00:20, 70.72it/s]

     28%|██▊       | 567/2000 [00:09<00:20, 71.57it/s]

     29%|██▉       | 575/2000 [00:09<00:19, 72.43it/s]

     29%|██▉       | 583/2000 [00:09<00:19, 71.87it/s]

     30%|██▉       | 591/2000 [00:09<00:19, 71.09it/s]

     30%|██▉       | 599/2000 [00:09<00:19, 70.46it/s]

     30%|███       | 607/2000 [00:09<00:19, 71.00it/s]

     31%|███       | 615/2000 [00:09<00:19, 71.26it/s]

     31%|███       | 623/2000 [00:09<00:19, 72.41it/s]

     32%|███▏      | 631/2000 [00:10<00:18, 72.54it/s]

     32%|███▏      | 639/2000 [00:10<00:18, 71.94it/s]

     32%|███▏      | 647/2000 [00:10<00:19, 70.01it/s]

     33%|███▎      | 655/2000 [00:10<00:19, 68.65it/s]

     33%|███▎      | 662/2000 [00:10<00:19, 67.85it/s]

     33%|███▎      | 669/2000 [00:10<00:19, 68.21it/s]

     34%|███▍      | 677/2000 [00:10<00:19, 69.07it/s]

     34%|███▍      | 685/2000 [00:10<00:18, 70.52it/s]

     35%|███▍      | 693/2000 [00:10<00:18, 71.18it/s]

     35%|███▌      | 701/2000 [00:11<00:18, 71.45it/s]

     35%|███▌      | 709/2000 [00:11<00:18, 71.06it/s]

     36%|███▌      | 717/2000 [00:11<00:18, 71.10it/s]

     36%|███▋      | 725/2000 [00:11<00:17, 71.37it/s]

     37%|███▋      | 733/2000 [00:11<00:17, 71.31it/s]

     37%|███▋      | 741/2000 [00:11<00:17, 72.06it/s]

     37%|███▋      | 749/2000 [00:11<00:17, 71.37it/s]

     38%|███▊      | 757/2000 [00:11<00:17, 69.09it/s]

     38%|███▊      | 764/2000 [00:11<00:17, 69.07it/s]

     39%|███▊      | 771/2000 [00:12<00:17, 69.24it/s]

     39%|███▉      | 779/2000 [00:12<00:17, 69.46it/s]

     39%|███▉      | 787/2000 [00:12<00:17, 69.95it/s]

     40%|███▉      | 795/2000 [00:12<00:17, 70.30it/s]

     40%|████      | 803/2000 [00:12<00:16, 70.91it/s]

     41%|████      | 811/2000 [00:12<00:16, 70.81it/s]

     41%|████      | 819/2000 [00:12<00:17, 69.32it/s]

     41%|████▏     | 826/2000 [00:12<00:17, 68.48it/s]

     42%|████▏     | 833/2000 [00:12<00:17, 66.36it/s]

     42%|████▏     | 840/2000 [00:13<00:18, 63.55it/s]

     42%|████▏     | 847/2000 [00:13<00:18, 62.73it/s]

     43%|████▎     | 854/2000 [00:13<00:17, 63.81it/s]

     43%|████▎     | 861/2000 [00:13<00:17, 64.30it/s]

     43%|████▎     | 868/2000 [00:13<00:17, 65.06it/s]

     44%|████▍     | 875/2000 [00:13<00:17, 65.14it/s]

     44%|████▍     | 882/2000 [00:13<00:16, 66.11it/s]

     44%|████▍     | 889/2000 [00:13<00:16, 66.65it/s]

     45%|████▍     | 896/2000 [00:13<00:16, 66.79it/s]

     45%|████▌     | 903/2000 [00:14<00:16, 66.85it/s]

     46%|████▌     | 910/2000 [00:14<00:16, 67.64it/s]

     46%|████▌     | 918/2000 [00:14<00:15, 68.42it/s]

     46%|████▋     | 926/2000 [00:14<00:15, 68.98it/s]

     47%|████▋     | 933/2000 [00:14<00:15, 68.26it/s]

     47%|████▋     | 940/2000 [00:14<00:15, 66.87it/s]

     47%|████▋     | 947/2000 [00:14<00:15, 66.34it/s]

     48%|████▊     | 954/2000 [00:14<00:15, 65.56it/s]

     48%|████▊     | 961/2000 [00:14<00:15, 65.52it/s]

     48%|████▊     | 968/2000 [00:15<00:15, 65.34it/s]

     49%|████▉     | 975/2000 [00:15<00:15, 65.19it/s]

     49%|████▉     | 982/2000 [00:15<00:15, 65.29it/s]

     49%|████▉     | 989/2000 [00:15<00:15, 65.61it/s]

     50%|████▉     | 996/2000 [00:15<00:15, 66.32it/s]

     50%|█████     | 1003/2000 [00:15<00:14, 66.83it/s]

     51%|█████     | 1011/2000 [00:15<00:14, 68.15it/s]

     51%|█████     | 1019/2000 [00:15<00:14, 69.29it/s]

     51%|█████▏    | 1027/2000 [00:15<00:13, 69.95it/s]

     52%|█████▏    | 1034/2000 [00:16<00:13, 69.26it/s]

     52%|█████▏    | 1041/2000 [00:16<00:13, 69.25it/s]

     52%|█████▏    | 1049/2000 [00:16<00:13, 69.62it/s]

     53%|█████▎    | 1057/2000 [00:16<00:13, 70.30it/s]

     53%|█████▎    | 1065/2000 [00:16<00:13, 70.39it/s]

     54%|█████▎    | 1073/2000 [00:16<00:13, 69.35it/s]

     54%|█████▍    | 1081/2000 [00:16<00:13, 70.09it/s]

     54%|█████▍    | 1089/2000 [00:16<00:12, 70.38it/s]

     55%|█████▍    | 1097/2000 [00:16<00:13, 68.60it/s]

     55%|█████▌    | 1104/2000 [00:17<00:13, 67.80it/s]

     56%|█████▌    | 1111/2000 [00:17<00:13, 68.02it/s]

     56%|█████▌    | 1119/2000 [00:17<00:12, 68.99it/s]

     56%|█████▋    | 1127/2000 [00:17<00:12, 69.50it/s]

     57%|█████▋    | 1135/2000 [00:17<00:12, 69.94it/s]

     57%|█████▋    | 1142/2000 [00:17<00:12, 69.51it/s]

     57%|█████▋    | 1149/2000 [00:17<00:12, 69.65it/s]

     58%|█████▊    | 1156/2000 [00:17<00:12, 69.72it/s]

     58%|█████▊    | 1164/2000 [00:17<00:11, 70.85it/s]

     59%|█████▊    | 1172/2000 [00:17<00:11, 70.15it/s]

     59%|█████▉    | 1180/2000 [00:18<00:11, 69.00it/s]

     59%|█████▉    | 1187/2000 [00:18<00:12, 67.45it/s]

     60%|█████▉    | 1194/2000 [00:18<00:11, 67.54it/s]

     60%|██████    | 1202/2000 [00:18<00:11, 68.74it/s]

     60%|██████    | 1210/2000 [00:18<00:11, 69.65it/s]

     61%|██████    | 1217/2000 [00:18<00:11, 69.47it/s]

     61%|██████▏   | 1225/2000 [00:18<00:11, 70.35it/s]

     62%|██████▏   | 1233/2000 [00:18<00:10, 70.66it/s]

     62%|██████▏   | 1241/2000 [00:18<00:10, 70.94it/s]

     62%|██████▏   | 1249/2000 [00:19<00:10, 71.29it/s]

     63%|██████▎   | 1257/2000 [00:19<00:10, 70.08it/s]

     63%|██████▎   | 1265/2000 [00:19<00:10, 69.72it/s]

     64%|██████▎   | 1272/2000 [00:19<00:10, 69.60it/s]

     64%|██████▍   | 1279/2000 [00:19<00:10, 69.45it/s]

     64%|██████▍   | 1286/2000 [00:19<00:10, 69.50it/s]

     65%|██████▍   | 1294/2000 [00:19<00:10, 70.11it/s]

     65%|██████▌   | 1302/2000 [00:19<00:09, 69.86it/s]

     65%|██████▌   | 1309/2000 [00:19<00:09, 69.47it/s]

     66%|██████▌   | 1316/2000 [00:20<00:09, 69.15it/s]

     66%|██████▌   | 1323/2000 [00:20<00:09, 67.75it/s]

     66%|██████▋   | 1330/2000 [00:20<00:09, 67.24it/s]

     67%|██████▋   | 1337/2000 [00:20<00:10, 66.07it/s]

     67%|██████▋   | 1344/2000 [00:20<00:09, 65.87it/s]

     68%|██████▊   | 1351/2000 [00:20<00:09, 66.04it/s]

     68%|██████▊   | 1358/2000 [00:20<00:09, 67.18it/s]

     68%|██████▊   | 1365/2000 [00:20<00:09, 67.77it/s]

     69%|██████▊   | 1373/2000 [00:20<00:09, 68.87it/s]

     69%|██████▉   | 1380/2000 [00:21<00:08, 69.06it/s]

     69%|██████▉   | 1387/2000 [00:21<00:08, 69.10it/s]

     70%|██████▉   | 1395/2000 [00:21<00:08, 69.58it/s]

     70%|███████   | 1402/2000 [00:21<00:08, 69.67it/s]

     70%|███████   | 1410/2000 [00:21<00:08, 70.60it/s]

     71%|███████   | 1418/2000 [00:21<00:08, 71.19it/s]

     71%|███████▏  | 1426/2000 [00:21<00:08, 71.37it/s]

     72%|███████▏  | 1434/2000 [00:21<00:07, 71.62it/s]

     72%|███████▏  | 1442/2000 [00:21<00:07, 71.10it/s]

     72%|███████▎  | 1450/2000 [00:22<00:07, 69.04it/s]

     73%|███████▎  | 1457/2000 [00:22<00:07, 67.97it/s]

     73%|███████▎  | 1465/2000 [00:22<00:07, 68.66it/s]

     74%|███████▎  | 1472/2000 [00:22<00:07, 68.97it/s]

     74%|███████▍  | 1480/2000 [00:22<00:07, 69.68it/s]

     74%|███████▍  | 1487/2000 [00:22<00:07, 68.70it/s]

     75%|███████▍  | 1494/2000 [00:22<00:07, 68.36it/s]

     75%|███████▌  | 1502/2000 [00:22<00:07, 69.09it/s]

     75%|███████▌  | 1509/2000 [00:22<00:07, 67.74it/s]

     76%|███████▌  | 1516/2000 [00:22<00:07, 66.32it/s]

     76%|███████▌  | 1523/2000 [00:23<00:07, 65.58it/s]

     76%|███████▋  | 1530/2000 [00:23<00:07, 61.97it/s]

     77%|███████▋  | 1537/2000 [00:23<00:08, 57.81it/s]

     77%|███████▋  | 1543/2000 [00:23<00:07, 57.34it/s]

     78%|███████▊  | 1550/2000 [00:23<00:07, 58.67it/s]

     78%|███████▊  | 1557/2000 [00:23<00:07, 60.94it/s]

     78%|███████▊  | 1564/2000 [00:23<00:06, 63.37it/s]

     79%|███████▊  | 1571/2000 [00:23<00:06, 64.73it/s]

     79%|███████▉  | 1579/2000 [00:24<00:06, 67.11it/s]

     79%|███████▉  | 1587/2000 [00:24<00:05, 69.62it/s]

     80%|███████▉  | 1594/2000 [00:24<00:05, 69.68it/s]

     80%|████████  | 1602/2000 [00:24<00:05, 70.66it/s]

     80%|████████  | 1610/2000 [00:24<00:05, 70.61it/s]

     81%|████████  | 1618/2000 [00:24<00:05, 70.65it/s]

     81%|████████▏ | 1626/2000 [00:24<00:05, 71.39it/s]

     82%|████████▏ | 1634/2000 [00:24<00:05, 72.20it/s]

     82%|████████▏ | 1642/2000 [00:24<00:05, 70.89it/s]

     82%|████████▎ | 1650/2000 [00:24<00:04, 71.02it/s]

     83%|████████▎ | 1658/2000 [00:25<00:04, 71.54it/s]

     83%|████████▎ | 1666/2000 [00:25<00:04, 70.34it/s]

     84%|████████▎ | 1674/2000 [00:25<00:04, 69.76it/s]

     84%|████████▍ | 1681/2000 [00:25<00:04, 69.28it/s]

     84%|████████▍ | 1688/2000 [00:25<00:04, 67.22it/s]

     85%|████████▍ | 1695/2000 [00:25<00:04, 62.82it/s]

     85%|████████▌ | 1702/2000 [00:25<00:04, 60.60it/s]

     85%|████████▌ | 1709/2000 [00:25<00:04, 59.00it/s]

     86%|████████▌ | 1715/2000 [00:26<00:04, 59.13it/s]

     86%|████████▌ | 1722/2000 [00:26<00:04, 60.18it/s]

     86%|████████▋ | 1729/2000 [00:26<00:04, 61.90it/s]

     87%|████████▋ | 1736/2000 [00:26<00:04, 62.79it/s]

     87%|████████▋ | 1743/2000 [00:26<00:04, 63.83it/s]

     88%|████████▊ | 1750/2000 [00:26<00:03, 64.97it/s]

     88%|████████▊ | 1757/2000 [00:26<00:03, 65.49it/s]

     88%|████████▊ | 1764/2000 [00:26<00:03, 65.57it/s]

     89%|████████▊ | 1771/2000 [00:26<00:03, 64.88it/s]

     89%|████████▉ | 1778/2000 [00:26<00:03, 64.14it/s]

     89%|████████▉ | 1785/2000 [00:27<00:03, 62.71it/s]

     90%|████████▉ | 1792/2000 [00:27<00:03, 61.70it/s]

     90%|████████▉ | 1799/2000 [00:27<00:03, 61.54it/s]

     90%|█████████ | 1806/2000 [00:27<00:03, 62.53it/s]

     91%|█████████ | 1813/2000 [00:27<00:02, 63.32it/s]

     91%|█████████ | 1820/2000 [00:27<00:02, 63.79it/s]

     91%|█████████▏| 1827/2000 [00:27<00:02, 64.06it/s]

     92%|█████████▏| 1834/2000 [00:27<00:02, 64.13it/s]

     92%|█████████▏| 1841/2000 [00:27<00:02, 64.76it/s]

     92%|█████████▏| 1848/2000 [00:28<00:02, 64.36it/s]

     93%|█████████▎| 1855/2000 [00:28<00:02, 64.15it/s]

     93%|█████████▎| 1862/2000 [00:28<00:02, 64.37it/s]

     93%|█████████▎| 1869/2000 [00:28<00:02, 62.47it/s]

     94%|█████████▍| 1876/2000 [00:28<00:02, 61.79it/s]

     94%|█████████▍| 1883/2000 [00:28<00:01, 62.94it/s]

     95%|█████████▍| 1891/2000 [00:28<00:01, 65.28it/s]

     95%|█████████▍| 1899/2000 [00:28<00:01, 67.40it/s]

     95%|█████████▌| 1907/2000 [00:28<00:01, 69.35it/s]

     96%|█████████▌| 1915/2000 [00:29<00:01, 70.36it/s]

     96%|█████████▌| 1923/2000 [00:29<00:01, 71.17it/s]

     97%|█████████▋| 1931/2000 [00:29<00:00, 71.06it/s]

     97%|█████████▋| 1939/2000 [00:29<00:00, 70.50it/s]

     97%|█████████▋| 1947/2000 [00:29<00:00, 69.12it/s]

     98%|█████████▊| 1955/2000 [00:29<00:00, 69.87it/s]

     98%|█████████▊| 1962/2000 [00:29<00:00, 69.55it/s]

     98%|█████████▊| 1970/2000 [00:29<00:00, 69.80it/s]

     99%|█████████▉| 1978/2000 [00:30<00:00, 69.79it/s]

     99%|█████████▉| 1986/2000 [00:30<00:00, 70.08it/s]

    100%|█████████▉| 1994/2000 [00:30<00:00, 70.04it/s]

    100%|██████████| 2000/2000 [00:30<00:00, 65.96it/s]

    


    2026-07-11 16:30:04,816 - tutorial_5_results_and_samples - INFO - Fit Running: Updating results (see output folder).


    2026-07-11 16:30:05,229 - autofit.non_linear.samples.samples - INFO - Samples with weight less than 1e-10 removed from samples.csv.


    2026-07-11 16:30:05,245 - autofit.non_linear.search.updater - INFO - Creating latent samples by drawing 100 from the PDF.


    2026-07-11 16:30:06,204 - arviz - INFO - Found 'auto' as default backend, checking available backends


    2026-07-11 16:30:06,205 - arviz - INFO - Matplotlib is available, defining as default backend


    2026-07-11 16:30:06,210 - arviz - INFO - arviz_base 1.0.0 available, exposing its functions as part of the `arviz` namespace


    2026-07-11 16:30:06,410 - arviz - INFO - arviz_stats 1.0.0 available, exposing its functions as part of the `arviz` namespace


    2026-07-11 16:30:06,425 - arviz - INFO - arviz_plots 1.0.0 available, exposing its functions as part of the `arviz` namespace


    2026-07-11 16:30:06,657 - root - WARNING - Too few points to create valid contours


    2026-07-11 16:30:06,691 - root - WARNING - Too few points to create valid contours


    2026-07-11 16:30:06,714 - root - WARNING - Too few points to create valid contours


    2026-07-11 16:30:06,744 - root - WARNING - Too few points to create valid contours


    2026-07-11 16:30:06,765 - root - WARNING - Too few points to create valid contours


    2026-07-11 16:30:06,786 - root - WARNING - Too few points to create valid contours


    2026-07-11 16:30:06,817 - root - WARNING - Too few points to create valid contours


    2026-07-11 16:30:06,837 - root - WARNING - Too few points to create valid contours


    2026-07-11 16:30:06,856 - root - WARNING - Too few points to create valid contours


    2026-07-11 16:30:06,880 - root - WARNING - Too few points to create valid contours


    2026-07-11 16:30:06,910 - root - WARNING - Too few points to create valid contours


    2026-07-11 16:30:06,932 - root - WARNING - Too few points to create valid contours


    2026-07-11 16:30:06,954 - root - WARNING - Too few points to create valid contours


    2026-07-11 16:30:07,067 - root - WARNING - Too few points to create valid contours


    2026-07-11 16:30:07,088 - root - WARNING - Too few points to create valid contours


    2026-07-11 16:30:07,791 - tutorial_5_results_and_samples - INFO - Removing all files except for .zip file


    2026-07-11 16:30:08,040 - tutorial_5_results_and_samples - INFO - Search complete, returning result


    The search has finished run - you may now continue the notebook.



    <Figure size 640x480 with 0 Axes>


__Result__

Here, we'll look in detail at what information is contained in the `Result`.

It contains an `info` attribute which prints the result in readable format.


```python
print(result.info)
```

    Maximum Log Likelihood                                                          157.53479909
    
    model                                                                           Collection (N=6)
        gaussian                                                                    Gaussian (N=3)
        exponential                                                                 Exponential (N=3)
    
    Maximum Log Likelihood Model:
    
    gaussian
        centre                                                                      -294101561365780736.000
    ... [15 lines of output truncated] ...
        centre                                                                      50.10 (49.60, 157.08)
        normalization                                                               51.74 (0.37, 54.07)
        rate                                                                        0.06 (0.05, 1.15)
    
    
    Summary (1.0 sigma limits):
    
    gaussian
        centre                                                                      50.14 (-10295526.34, 192063.19)
        normalization                                                               8.73 (2.66, 78.09)
        sigma                                                                       10.80 (-936478.33, 26484.73)
    exponential
        centre                                                                      50.10 (50.05, 50.15)
        normalization                                                               51.74 (51.54, 51.92)
        rate                                                                        0.06 (0.06, 0.06)
    
    instances
    
    


__Samples__

The result contains a `Samples` object, which contains all of the non-linear search samples. 

Each sample corresponds to a set of model parameters that were evaluated and accepted by our non linear search, 
in this example emcee. 

This also includes their log likelihoods, which are used for computing additional information about the model-fit,
for example the error on every parameter. 

Our model-fit used the MCMC algorithm Emcee, so the `Samples` object returned is a `SamplesMCMC` object.


```python
samples = result.samples

print("MCMC Samples: \n")
print(samples)
```

    MCMC Samples: 
    
    SamplesMCMC(500)


__Parameters__

The parameters are stored as a list of lists, where:

 - The outer list is the size of the total number of samples.
 - The inner list is the size of the number of free parameters in the fit.

Below, we print the first sample — its second parameter (Gaussian -> normalization) and its third
parameter (Gaussian -> sigma). Any other sample index would work the same way; index `0` is used
here so the example remains valid regardless of how many samples the search produced.


```python
samples = result.samples
print("Sample 0's second parameter value (Gaussian -> normalization):")
print(samples.parameter_lists[0][1])
print("Sample 0's third parameter value (Gaussian -> sigma)")
print(samples.parameter_lists[0][2], "\n")
```

    Sample 0's second parameter value (Gaussian -> normalization):
    0.11335924010781939
    Sample 0's third parameter value (Gaussian -> sigma)
    3099.028904610647 
    


__Figures of Merit__

The Samples class also contains the log likelihood, log prior, log posterior and weight_list of every accepted sample, 
where:

- The log likelihood is the value evaluated from the likelihood function (e.g. -0.5 * chi_squared + the noise 
normalized).

- The log prior encodes information on how the priors on the parameters maps the log likelihood value to the log 
posterior value.

- The log posterior is log_likelihood + log_prior.

- The weight gives information on how samples should be combined to estimate the posterior. The weight values depend on
the sampler used, for MCMC samples they are all 1 (e.g. all weighted equally).

Below, we inspect the first sample. Any sample index would work the same way; index `0` is used here
so the example is valid regardless of how many samples the search produced.


```python
print("log(likelihood), log(prior), log(posterior) and weight of the first sample.")
print(samples.log_likelihood_list[0])
print(samples.log_prior_list[0])
print(samples.log_posterior_list[0])
print(samples.weight_list[0])
```

    log(likelihood), log(prior), log(posterior) and weight of the first sample.
    68.35981822308696
    -1.7760622584685666
    66.58375596461839
    1.0


__Instances__

The `Samples` contains many results which are returned as an instance of the model, using the Python class structure
of the model composition.

For example, we can return the model parameters corresponding to the maximum log likelihood sample.


```python
max_lh_instance = samples.max_log_likelihood()

print("Max Log Likelihood `Gaussian` Instance:")
print("Centre = ", max_lh_instance.gaussian.centre)
print("Normalization = ", max_lh_instance.gaussian.normalization)
print("Sigma = ", max_lh_instance.gaussian.sigma, "\n")

print("Max Log Likelihood Exponential Instance:")
print("Centre = ", max_lh_instance.exponential.centre)
print("Normalization = ", max_lh_instance.exponential.normalization)
print("Sigma = ", max_lh_instance.exponential.rate, "\n")
```

    Max Log Likelihood `Gaussian` Instance:
    Centre =  -2.9410156136578074e+17
    Normalization =  523915696.11279744
    Sigma =  -3.023887584844618e+16 
    
    Max Log Likelihood Exponential Instance:
    Centre =  50.2399798087453
    Normalization =  51.65027804241896
    Sigma =  0.0623539817262462 
    


__Vectors__

All results can alternatively be returned as a 1D vector of values, by passing `as_instance=False`:


```python
max_lh_vector = samples.max_log_likelihood(as_instance=False)
print("Max Log Likelihood Model Parameters: \n")
print(max_lh_vector, "\n\n")
```

    Max Log Likelihood Model Parameters: 
    
    [-2.9410156136578074e+17, 523915696.11279744, -3.023887584844618e+16, 50.2399798087453, 51.65027804241896, 0.0623539817262462] 
    
    


__Labels__

Vectors return a lists of all model parameters, but do not tell us which values correspond to which parameters.

The following quantities are available in the `Model`, where the order of their entries correspond to the parameters 
in the `ml_vector` above:
 
 - `paths`: a list of tuples which give the path of every parameter in the `Model`.
 - `parameter_names`: a list of shorthand parameter names derived from the `paths`.
 - `parameter_labels`: a list of parameter labels used when visualizing non-linear search results (see below).


```python
model = samples.model

print(model.paths)
print(model.parameter_names)
print(model.parameter_labels)
print(model.model_component_and_parameter_names)
print("\n")
```

    [('gaussian', 'centre'), ('gaussian', 'normalization'), ('gaussian', 'sigma'), ('exponential', 'centre'), ('exponential', 'normalization'), ('exponential', 'rate')]
    ['centre', 'normalization', 'sigma', 'centre', 'normalization', 'rate']
    ['x', 'norm', '\\sigma', 'x', 'norm', '\\lambda']
    ['gaussian.centre', 'gaussian.normalization', 'gaussian.sigma', 'exponential.centre', 'exponential.normalization', 'exponential.rate']
    
    


From here on, we will returned all results information as instances, but every method below can be returned as a
vector via the `as_instance=False` input.

__Posterior / PDF__

The ``Result`` object contains the full posterior information of our non-linear search, which can be used for
parameter estimation. 

The median pdf vector is available from the `Samples` object, which estimates the every parameter via 1D 
marginalization of their PDFs.


```python
median_pdf_instance = samples.median_pdf()

print("Max Log Likelihood `Gaussian` Instance:")
print("Centre = ", median_pdf_instance.gaussian.centre)
print("Normalization = ", median_pdf_instance.gaussian.normalization)
print("Sigma = ", median_pdf_instance.gaussian.sigma, "\n")

print("Max Log Likelihood Exponential Instance:")
print("Centre = ", median_pdf_instance.exponential.centre)
print("Normalization = ", median_pdf_instance.exponential.normalization)
print("Sigma = ", median_pdf_instance.exponential.rate, "\n")
```

    Max Log Likelihood `Gaussian` Instance:
    Centre =  50.13507900894478
    Normalization =  8.730203550030652
    Sigma =  10.797678770589716 
    
    Max Log Likelihood Exponential Instance:
    Centre =  50.1016222399324
    Normalization =  51.74042118842236
    Sigma =  0.06213782821625456 
    


__Plot__

Because results are returned as instances, it is straight forward to use them and their associated functionality
to make plots of the results:


```python
model_gaussian = max_lh_instance.gaussian.model_data_from(
    xvalues=np.arange(data.shape[0])
)
model_exponential = max_lh_instance.exponential.model_data_from(
    xvalues=np.arange(data.shape[0])
)
model_data = model_gaussian + model_exponential

plt.plot(range(data.shape[0]), data)
plt.plot(range(data.shape[0]), model_data)
plt.plot(range(data.shape[0]), model_gaussian, "--")
plt.plot(range(data.shape[0]), model_exponential, "--")
plt.title("Illustrative model fit to 1D `Gaussian` + Exponential profile data.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()
```


    
![png](tutorial_5_results_and_samples_files/tutorial_5_results_and_samples_27_0.png)
    


__Errors__

The samples include methods for computing the error estimates of all parameters, via 1D marginalization at an 
input sigma confidence limit. 


```python
errors_at_upper_sigma_instance = samples.errors_at_upper_sigma(sigma=3.0)
errors_at_lower_sigma_instance = samples.errors_at_lower_sigma(sigma=3.0)

print("Upper Error values of Gaussian (at 3.0 sigma confidence):")
print("Centre = ", errors_at_upper_sigma_instance.gaussian.centre)
print("Normalization = ", errors_at_upper_sigma_instance.gaussian.normalization)
print("Sigma = ", errors_at_upper_sigma_instance.gaussian.sigma, "\n")

print("lower Error values of Gaussian (at 3.0 sigma confidence):")
print("Centre = ", errors_at_lower_sigma_instance.gaussian.centre)
print("Normalization = ", errors_at_lower_sigma_instance.gaussian.normalization)
print("Sigma = ", errors_at_lower_sigma_instance.gaussian.sigma, "\n")
```

    Upper Error values of Gaussian (at 3.0 sigma confidence):
    Centre =  1.9035565111885748e+16
    Normalization =  876330725.5982871
    Sigma =  1954631025976492.2 
    
    lower Error values of Gaussian (at 3.0 sigma confidence):
    Centre =  6.468665924711259e+17
    Normalization =  8.730203390689214
    Sigma =  6.650671449666327e+16 
    


They can also be returned at the values of the parameters at their error values:


```python
values_at_upper_sigma_instance = samples.values_at_upper_sigma(sigma=3.0)
values_at_lower_sigma_instance = samples.values_at_lower_sigma(sigma=3.0)

print("Upper Parameter values w/ error of Gaussian (at 3.0 sigma confidence):")
print("Centre = ", values_at_upper_sigma_instance.gaussian.centre)
print("Normalization = ", values_at_upper_sigma_instance.gaussian.normalization)
print("Sigma = ", values_at_upper_sigma_instance.gaussian.sigma, "\n")

print("lower Parameter values w/ errors of Gaussian (at 3.0 sigma confidence):")
print("Centre = ", values_at_lower_sigma_instance.gaussian.centre)
print("Normalization = ", values_at_lower_sigma_instance.gaussian.normalization)
print("Sigma = ", values_at_lower_sigma_instance.gaussian.sigma, "\n")
```

    Upper Parameter values w/ error of Gaussian (at 3.0 sigma confidence):
    Centre =  1.90355651118858e+16
    Normalization =  876330734.3284906
    Sigma =  1954631025976503.0 
    
    lower Parameter values w/ errors of Gaussian (at 3.0 sigma confidence):
    Centre =  -6.468665924711259e+17
    Normalization =  1.5934143739195753e-07
    Sigma =  -6.650671449666326e+16 
    


__PDF__

The Probability Density Functions (PDF's) of the results can be plotted using the Emcee's visualization
tool `corner.py`, which is wrapped via the `aplt.corner_cornerpy` function.


```python
aplt.corner_cornerpy(samples=result.samples)
```

    2026-07-11 16:30:08,517 - root - WARNING - Too few points to create valid contours


    2026-07-11 16:30:08,545 - root - WARNING - Too few points to create valid contours


    2026-07-11 16:30:08,566 - root - WARNING - Too few points to create valid contours


    2026-07-11 16:30:08,593 - root - WARNING - Too few points to create valid contours


    2026-07-11 16:30:08,613 - root - WARNING - Too few points to create valid contours


    2026-07-11 16:30:08,633 - root - WARNING - Too few points to create valid contours


    2026-07-11 16:30:08,663 - root - WARNING - Too few points to create valid contours


    2026-07-11 16:30:08,682 - root - WARNING - Too few points to create valid contours


    2026-07-11 16:30:08,700 - root - WARNING - Too few points to create valid contours


    2026-07-11 16:30:08,717 - root - WARNING - Too few points to create valid contours


    2026-07-11 16:30:08,743 - root - WARNING - Too few points to create valid contours


    2026-07-11 16:30:08,763 - root - WARNING - Too few points to create valid contours


    2026-07-11 16:30:08,781 - root - WARNING - Too few points to create valid contours


    2026-07-11 16:30:08,802 - root - WARNING - Too few points to create valid contours


    2026-07-11 16:30:08,820 - root - WARNING - Too few points to create valid contours



    
![png](tutorial_5_results_and_samples_files/tutorial_5_results_and_samples_33_15.png)
    


__Other Results__

The samples contain many useful vectors, including the samples with the highest posterior values.


```python
max_log_posterior_instance = samples.max_log_posterior()

print("Maximum Log Posterior Vector:")
print("Centre = ", max_log_posterior_instance.gaussian.centre)
print("Normalization = ", max_log_posterior_instance.gaussian.normalization)
print("Sigma = ", max_log_posterior_instance.gaussian.sigma, "\n")

```

    Maximum Log Posterior Vector:
    Centre =  -24011277044876.77
    Normalization =  3125270.2307336275
    Sigma =  -2387925142098.1606 
    


All methods above are available as a vector:


```python
median_pdf_instance = samples.median_pdf(as_instance=False)
values_at_upper_sigma = samples.values_at_upper_sigma(sigma=3.0, as_instance=False)
values_at_lower_sigma = samples.values_at_lower_sigma(sigma=3.0, as_instance=False)
errors_at_upper_sigma = samples.errors_at_upper_sigma(sigma=3.0, as_instance=False)
errors_at_lower_sigma = samples.errors_at_lower_sigma(sigma=3.0, as_instance=False)
```

__Sample Instance__

A non-linear search retains every model that is accepted during the model-fit.

We can create an instance of any lens model -- below we create an instance of the last accepted model.


```python
instance = samples.from_sample_index(sample_index=-1)

print("Gaussian Instance of last sample")
print("Centre = ", instance.gaussian.centre)
print("Normalization = ", instance.gaussian.normalization)
print("Sigma = ", instance.gaussian.sigma, "\n")
```

    Gaussian Instance of last sample
    Centre =  -2.5614444939533196e+16
    Normalization =  798.0401817263084
    Sigma =  -2627054060301375.5 
    


__Bayesian Evidence__

If a nested sampling `NonLinearSearch` is used, the evidence of the model is also available which enables Bayesian
model comparison to be performed (given we are using Emcee, which is not a nested sampling algorithm, the log evidence 
is None).:


```python
log_evidence = samples.log_evidence
```

__Derived Errors (PDF from samples)__

Computing the errors of a quantity like the `sigma` of the Gaussian is simple, because it is sampled by the non-linear 
search. Thus, to get their errors above we used the `Samples` object to simply marginalize over all over parameters 
via the 1D Probability Density Function (PDF).

Computing errors on derived quantities is more tricky, because they are not sampled directly by the non-linear search. 
For example, what if we want the error on the full width half maximum (FWHM) of the Gaussian? In order to do this
we need to create the PDF of that derived quantity, which we can then marginalize over using the same function we
use to marginalize model parameters.

Below, we compute the FWHM of every accepted model sampled by the non-linear search and use this determine the PDF 
of the FWHM. When combining the FWHM's we weight each value by its `weight`. For Emcee, an MCMC algorithm, the
weight of every sample is 1, but weights may take different values for other non-linear searches.

In order to pass these samples to the function `marginalize`, which marginalizes over the PDF of the FWHM to compute 
its error, we also pass the weight list of the samples.

(Computing the error on the FWHM could be done in much simpler ways than creating its PDF from the list of every
sample. We chose this example for simplicity, in order to show this functionality, which can easily be extended to more
complicated derived quantities.)


```python
fwhm_list = []

for sample in samples.sample_list:
    instance = sample.instance_for_model(model=samples.model)

    sigma = instance.gaussian.sigma

    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma

    fwhm_list.append(fwhm)

median_fwhm, lower_fwhm, upper_fwhm = af.marginalize(
    parameter_list=fwhm_list, sigma=3.0, weight_list=samples.weight_list
)

print(f"FWHM = {median_fwhm} ({upper_fwhm} {lower_fwhm}")
```

    FWHM = 25.4265904087898 (6898791711474904.0 -6.302666628080957e+17


__Samples Filtering__

Our samples object has the results for all three parameters in our model. However, we might only be interested in the
results of a specific parameter.

The basic form of filtering specifies parameters via their path, which was printed above via the model and is printed 
again below.


```python
samples = result.samples

print("Parameter paths in the model which are used for filtering:")
print(samples.model.paths)

print("All parameters of the very first sample")
print(samples.parameter_lists[0])

samples = samples.with_paths([("gaussian", "centre")])

print("All parameters of the very first sample (containing only the Gaussian centre.")
print(samples.parameter_lists[0])

print("Maximum Log Likelihood Model Instances (containing only the Gaussian centre):\n")
print(samples.max_log_likelihood(as_instance=False))
```

    Parameter paths in the model which are used for filtering:
    [('gaussian', 'centre'), ('gaussian', 'normalization'), ('gaussian', 'sigma'), ('exponential', 'centre'), ('exponential', 'normalization'), ('exponential', 'rate')]
    All parameters of the very first sample
    [31513.97299772367, 0.11335924010781939, 3099.028904610647, 49.92329668729731, 52.104725511588285, 0.06232091297101713]
    All parameters of the very first sample (containing only the Gaussian centre.
    [31513.97299772367]
    Maximum Log Likelihood Model Instances (containing only the Gaussian centre):
    
    [-2.9410156136578074e+17]


Above, we specified each path as a list of tuples of strings. 

This is how the source code internally stores the path to different components of the model, but it is not 
in-profile_1d with the PyAutoFIT API used to compose a model.

We can alternatively use the following API:


```python
samples = result.samples

samples = samples.with_paths(["gaussian.centre"])

print("All parameters of the very first sample (containing only the Gaussian centre).")
print(samples.parameter_lists[0])
```

    All parameters of the very first sample (containing only the Gaussian centre).
    [31513.97299772367]


Above, we filtered the `Samples` but asking for all parameters which included the path ("gaussian", "centre").

We can alternatively filter the `Samples` object by removing all parameters with a certain path. Below, we remove
the Gaussian's `centre` to be left with 2 parameters; the `normalization` and `sigma`.


```python
samples = result.samples

print("Parameter paths in the model which are used for filtering:")
print(samples.model.paths)

print("All parameters of the very first sample")
print(samples.parameter_lists[0])

samples = samples.without_paths(["gaussian.centre"])

print(
    "All parameters of the very first sample (containing only the Gaussian normalization and sigma)."
)
print(samples.parameter_lists[0])
```

    Parameter paths in the model which are used for filtering:
    [('gaussian', 'centre'), ('gaussian', 'normalization'), ('gaussian', 'sigma'), ('exponential', 'centre'), ('exponential', 'normalization'), ('exponential', 'rate')]
    All parameters of the very first sample
    [31513.97299772367, 0.11335924010781939, 3099.028904610647, 49.92329668729731, 52.104725511588285, 0.06232091297101713]
    All parameters of the very first sample (containing only the Gaussian normalization and sigma).
    [0.11335924010781939, 3099.028904610647, 49.92329668729731, 52.104725511588285, 0.06232091297101713]


__Latex__

If you are writing modeling results up in a paper, you can use inbuilt latex tools to create latex table 
code which you can copy to your .tex document.

By combining this with the filtering tools below, specific parameters can be included or removed from the latex.

Remember that the superscripts of a parameter are loaded from the config file `notation/label.yaml`, providing high
levels of customization for how the parameter names appear in the latex table. This is especially useful if your model
uses the same model components with the same parameter, which therefore need to be distinguished via superscripts.


```python
latex = af.text.Samples.latex(
    samples=result.samples,
    median_pdf_model=True,
    sigma=3.0,
    name_to_label=True,
    include_name=True,
    include_quickmath=True,
    prefix="Example Prefix ",
    suffix=" \\[-2pt]",
)

print(latex)
```

    Example Prefix $x^{\rm{g}} = 50.14^{+19035565111885748.00}_{-646866592471125888.00}$ & $norm^{\rm{g}} = 8.73^{+876330725.60}_{-8.73}$ & $\sigma^{\rm{g}} = 10.80^{+1954631025976492.25}_{-66506714496663272.00}$ & $x^{\rm{e}} = 50.10^{+106.98}_{-0.50}$ & $norm^{\rm{e}} = 51.74^{+2.33}_{-51.37}$ & $\lambda^{\rm{e}} = 0.06^{+1.09}_{-0.01}$ \[-2pt]


Finish.


```python

```
