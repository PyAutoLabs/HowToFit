"""
Tutorial 7: The Details
=======================

In tutorial 6 we opened up the non-linear search and looked at how it moves. We saw that some searches only ever ask
the likelihood function "what is the value here?", whilst gradient based searches also ask "and which way is uphill?",
and that **PyAutoFit** can answer the second question exactly by using JAX to differentiate the likelihood function.

This tutorial is about everything else: the details sitting underneath every model-fit. How your model is
parameterized, what happens when two parameter combinations describe the same data, what a search does when a
parameter stops mattering, and what parameter space looks like from the search's point of view.

I want to be upfront. Most of the time none of this matters, and a sensible model fitted with a sensible search
gives a sensible answer. The reason to learn these details is that when they do matter they are hard to spot,
because the failure mode is not an error message but a wrong answer that looks completely reasonable.

__Overview__

In this tutorial, we will look at the details that make a model-fit succeed or fail, specifically:

- Composing a model whose components can swap roles, producing mirror solutions, and removing them with an
  "assertion".

- Creating a "plateau", a flat region of parameter space where a parameter has no effect, and removing it.

- Running the same problem with three searches, and using the disagreement between them as a diagnostic.

- Making the likelihood return `NaN` on purpose, and reading the `NaN` diagnostics a gradient search writes out.

- Looking at parameter space the way a search does, through the "unit cube" defined by the priors.

__Contents__

This tutorial is split into the following sections:

- **Data**: Load the two-Gaussian dataset used to study degenerate parameterizations.
- **Model**: The JAX-capable `Gaussian` model component from tutorial 6.
- **Analysis**: The `Analysis` class, fitting a `Collection` of profiles on NumPy or JAX.
- **Parameterization**: How two model components that can swap roles produce mirror solutions, and why that hurts a search.
- **Assertions**: Removing mirror solutions with `add_assertion` and what happens to a sample that violates one.
- **Plateaus**: How a parameter whose effect vanishes (a normalization of zero) creates a flat region of parameter space, and how to remove it.
- **Comparing Searches**: Running the same problems with MCMC, nested sampling and gradient descent to see how each is affected, and why comparing searches is a diagnostic in itself.
- **Clipping**: Parameter combinations that are unphysical even when every individual prior is sensible, the NaN likelihoods they produce, and the resample figure of merit each search substitutes.
- **NaN Diagnostics**: Reading the value-NaN and gradient-NaN counters in `search.summary`, and why a finite likelihood does not guarantee a finite gradient.
- **Unit Cube Vs Physical**: How a search actually sees parameter space through the priors, and why that changes how it explores.
- **Summary**: When these details matter, and what attending to them buys you.
"""

# from autofit import setup_notebook; setup_notebook()

from os import path
import time
import numpy as np
import matplotlib.pyplot as plt

import autofit as af
import autofit.plot as aplt

"""
__Data__

Load and plot the dataset from the `HowToFit/dataset` folder.

We use the `gaussian_x2` dataset, simulated from two Gaussians sharing a centre of 50.0. The first has a
normalization of 20.0 and a sigma of 1.0, a narrow spike; the second a normalization of 40.0 and a sigma of 5.0, a
broader bump underneath. The two are summed and noise added, as in tutorials 2 and 4. Two identical model components
fitted to two component data is the simplest example of the parameterization problem we study first.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x2")

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
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

xvalues = np.arange(data.shape[0])

plt.errorbar(
    xvalues,
    data,
    yerr=noise_map,
    linestyle="",
    color="k",
    ecolor="k",
    elinewidth=1,
    capsize=2,
)
plt.title("1D dataset containing two blended Gaussians.")
plt.xlabel("x values of profile")
plt.ylabel("Profile Normalization")
plt.show()
plt.clf()

"""
__Model__

The narrow spike sits on top of the broader bump, but only just. The two components are blended, and by eye it is
hard to say where one ends and the other begins. That blending is what makes the parameter space interesting.

We now re-define the `Gaussian` class, using the version introduced in tutorial 6, which is identical to the class
used since tutorial 1 except that `model_data_from` takes an extra argument `xp` and is written in `xp.` functions.
As tutorial 6 showed, `xp` is whichever array library the fit is running with: `numpy` by default, `jax.numpy` when
the `Analysis` has been told to use JAX.
"""


class Gaussian:
    def __init__(
        self,
        centre: float = 30.0,  # <- **PyAutoFit** recognises these constructor arguments
        normalization: float = 1.0,  # <- are the Gaussian`s model parameters.
        sigma: float = 5.0,
    ):
        """
        Represents a 1D Gaussian profile.

        This is a model-component of example models in the **HowToFit** lectures and is used to perform model-fitting
        of example datasets.

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

    def model_data_from(self, xvalues: np.ndarray, xp=np) -> np.ndarray:
        """
        Returns a 1D Gaussian on an input list of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Gaussian, via its `centre`.

        The output is referred to as the `model_data` to signify that it is a representation of the data from the
        model.

        Parameters
        ----------
        xvalues
            The x coordinates in the original reference frame of the data.
        xp
            The array library the calculation is performed with, `numpy` by default and `jax.numpy` when the fit
            is being performed with JAX.

        Returns
        -------
        np.array
            The Gaussian values at the input x coordinates.
        """
        transformed_xvalues = xp.subtract(xvalues, self.centre)
        return xp.multiply(
            xp.divide(self.normalization, self.sigma * xp.sqrt(2.0 * np.pi)),
            xp.exp(-0.5 * xp.square(xp.divide(transformed_xvalues, self.sigma))),
        )


"""
__Analysis__

The `Analysis` class below is the tutorial 4 version, which sums the model data of every profile in the instance,
combined with the tutorial 6 version, which accepts `use_jax` and passes it to `af.Analysis` via `super().__init__`.

Two details matter. The first is `self._xp`, an attribute created for us by `af.Analysis` which is `numpy` when
`use_jax=False` and `jax.numpy` when `use_jax=True`. Every array operation in the likelihood goes through it, which
is why we write `self._xp.sum` rather than the built-in `sum`. The second is `model_data_from_instance`: the
`instance` is an iterable `Collection`, so we sum the model data of every profile it contains without knowing how
many there are.

We create the `Analysis` at the end of the cell below, leaving `use_jax` at its default of `False` until we need
gradients.
"""


class Analysis(af.Analysis):
    def __init__(self, data: np.ndarray, noise_map: np.ndarray, use_jax: bool = False):
        """
        The `Analysis` class acts as an interface between the data and model in **PyAutoFit**.

        Parameters
        ----------
        data
            A 1D numpy array containing the data (e.g. a noisy 1D signal) fitted in the workspace examples.
        noise_map
            A 1D numpy array containing the noise values of the data, used for computing the goodness of fit
            metric, the log likelihood.
        use_jax
            If `True` the log likelihood function is computed with `jax.numpy`, making it differentiable and
            therefore usable by the gradient based searches introduced in tutorial 6.
        """
        super().__init__(use_jax=use_jax)

        self.data = data
        self.noise_map = noise_map

    def model_data_from_instance(self, instance):
        """
        Returns the summed model data of every profile in an instance.

        The `instance` is a `Collection`, which is iterable, so a list comprehension over it returns the model data
        of every profile it contains.
        """
        xvalues = self._xp.arange(self.data.shape[0])

        return sum(
            [
                profile.model_data_from(xvalues=xvalues, xp=self._xp)
                for profile in instance
            ]
        )

    def log_likelihood_function(self, instance) -> float:
        """
        Returns the log likelihood of a fit of a collection of 1D profiles to the dataset.

        Every array operation uses `self._xp`, which is `numpy` for an ordinary fit and `jax.numpy` for a fit which
        uses JAX. The log likelihood is therefore differentiable whenever `use_jax=True` was passed to `__init__`.
        """
        model_data = self.model_data_from_instance(instance=instance)

        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        chi_squared = self._xp.sum(chi_squared_map)
        noise_normalization = self._xp.sum(
            self._xp.log(2 * np.pi * self.noise_map**2.0)
        )
        log_likelihood = -0.5 * (chi_squared + noise_normalization)

        return log_likelihood


analysis = Analysis(data=data, noise_map=noise_map)

"""
__Parameterization__

"Parameterization" means the specific choice of parameters we use to describe our model. It is easy to think of a
model as a fixed thing, but it is not: the same physical model can be written down in many different ways, and the
choice we make changes the shape of parameter space that the search has to explore.

Our dataset contains two Gaussians, so the obvious model is a `Collection` of two `Gaussian` components, with
identical priors on both. This is the model composition API from tutorial 1, and it is the same six free parameter
model that tutorial 2 fitted by hand.
"""
model = af.Collection(gaussian_0=af.Model(Gaussian), gaussian_1=af.Model(Gaussian))

for gaussian in [model.gaussian_0, model.gaussian_1]:
    gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    gaussian.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=25.0)

print(model.info)

"""
There is a problem hiding in that model. The two components are interchangeable: nothing says `gaussian_0` is the
narrow one and `gaussian_1` the broad one. Swap all six values over and we get exactly the same summed model data,
because addition does not care about the order of its arguments, which means the same residuals, the same
chi-squared and the same log likelihood.

Let's confirm it, using the parameter ordering of the model vector which, as tutorial 1 showed, is given by
`model.model_component_and_parameter_names`.
"""
print(model.model_component_and_parameter_names)

vector_ordered = [50.0, 20.0, 1.0, 50.0, 40.0, 5.0]
vector_swapped = [50.0, 40.0, 5.0, 50.0, 20.0, 1.0]

instance_ordered = model.instance_from_vector(vector=vector_ordered)
instance_swapped = model.instance_from_vector(vector=vector_swapped)

print("Log likelihood of the ordered model: ")
print(analysis.log_likelihood_function(instance=instance_ordered))
print("Log likelihood of the swapped model: ")
print(analysis.log_likelihood_function(instance=instance_swapped))

"""
The two log likelihoods are identical to the last decimal place. They are not merely similar, they are the same
number, because the two vectors produce the same model data.

This is a "degeneracy", and this one is a "label switching" degeneracy: the labels `gaussian_0` and `gaussian_1` are
arbitrary, so every solution has a mirror image partner elsewhere. Our six dimensional parameter space does not
contain one peak, it contains two identical peaks related by a reflection.

Why does this hurt? Recall the searches of tutorial 3. Every search has to decide which of the two peaks to spend
its effort on, and nothing in the data prefers one over the other. A search that commits to one peak gives an
answer whose component labels are a coin toss; a search that splits its effort between both returns a posterior
that is a smeared average of two solutions rather than an estimate of either, whose "errors" are not errors at all
but a measure of the distance between the mirrors. Either way we have paid for a second copy of the answer.

Let's fit with `DynestyStatic`, the nested sampling search from tutorial 3.
"""
search = af.DynestyStatic(
    sample="rwalk",  # This makes dynesty run faster, dont worry about what it means for now!
)

print(
    """
    The non-linear search has begun running.
    This Jupyter notebook cell with progress once the search has completed - this could take a few minutes!
    """
)

start = time.time()

result_flip = search.fit(model=model, analysis=analysis)

print(f"Dynesty run time: {time.time() - start} seconds")
print("The search has finished run - you may now continue the notebook.")

"""
The corner plot below shows the Probability Density Functions (PDF's) of the result, via the `corner.py` wrapper
`aplt.corner_cornerpy` we met in tutorial 5. The run is short, so there may be too few samples for `corner.py` to
draw contours; the scatter plot afterwards makes the point without them, putting every sample's `gaussian_0.sigma`
against its `gaussian_1.sigma`.

Its diagonal is the line where the two sigmas are equal, and the two mirror solutions sit on opposite sides of it.
What you will almost certainly see is the great majority of samples piled on one side, with a thin scatter of early
exploration samples on the other: the search committed to one mirror. Which one is a coin toss, and I got each of
them on different runs while writing this tutorial.
"""
aplt.corner_cornerpy(samples=result_flip.samples)

sigma_0_list = [parameters[2] for parameters in result_flip.samples.parameter_lists]
sigma_1_list = [parameters[5] for parameters in result_flip.samples.parameter_lists]

plt.scatter(sigma_0_list, sigma_1_list, s=1, color="k")
plt.title("Samples of the two sigmas, split by the two mirror solutions.")
plt.xlabel("gaussian_0 sigma")
plt.ylabel("gaussian_1 sigma")
plt.show()
plt.clf()

ordered_count = sum(
    sigma_0 < sigma_1 for sigma_0, sigma_1 in zip(sigma_0_list, sigma_1_list)
)

print(f"Samples with gaussian_0 narrower than gaussian_1: {ordered_count}")
print(
    f"Samples with gaussian_0 broader than gaussian_1: {len(sigma_0_list) - ordered_count}"
)

"""
The maximum log likelihood model printed below is one of the two mirror solutions, and it fits the data beautifully.
Nothing has failed.

What has gone wrong is that the answer's labels are arbitrary. Read off "the narrow component has a normalization
of about 20" and you may be reading the wrong component, because the next run may put that solution in the other
slot. A degeneracy does not make the fit worse, it makes the *answer ambiguous*, and it costs the search effort
exploring a region it did not need to. The fit does not crash, it quietly costs more and tells us less than it
should.

Be clear about what this short run does and does not show. It shows that the two mirrors exist, which we proved
exactly with the two identical log likelihoods, and that the search commits to one. It does not show a posterior
with two clean modes: a nested sampler with far more live points, run far longer, is designed to find and report
both peaks, and we have deliberately traded that away for a fit that runs in seconds.
"""
print(result_flip.info)

"""
__Assertions__

The fix is to tell the model something we already know: `gaussian_0` is the narrow component and `gaussian_1` is
the broad one. If we forbid every model where that is not true, we delete one of the two mirror peaks and leave a
parameter space with a single solution.

**PyAutoFit** expresses this with an "assertion", added to the model with `add_assertion`. An assertion is a
statement about the model's parameters that every accepted model must satisfy. Here we assert that the sigma of
`gaussian_0` is less than the sigma of `gaussian_1`, which makes `gaussian_0` the narrow one by definition.
"""
model = af.Collection(gaussian_0=af.Model(Gaussian), gaussian_1=af.Model(Gaussian))

for gaussian in [model.gaussian_0, model.gaussian_1]:
    gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    gaussian.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=25.0)

model.add_assertion(model.gaussian_0.sigma < model.gaussian_1.sigma)

"""
What actually happens when the search proposes a model that violates the assertion?

On the NumPy path, building the instance from the parameter vector raises an exception internally, which
**PyAutoFit** catches and converts into a special value called the "resample figure of merit". The search never sees
a likelihood for that model, it sees a number so bad it can never be selected.

On the JAX path, introduced in tutorial 6 via `use_jax=True`, an exception cannot be raised at all, because the
likelihood is traced and compiled rather than executed line by line. **PyAutoFit** instead evaluates the assertion
as a traced boolean and applies it to the final figure of merit with a `where`, mapping a violating model to exactly
the same resample figure of merit. Different mechanism, same outcome, which is what lets a NumPy fit and a JAX fit
agree exactly.

We can check the assertion ourselves with `assertions_satisfied_from_vector`, using the two mirror vectors from
before. One passes and one fails, which is the whole point.
"""
print("Ordered vector satisfies the assertion: ")
print(model.assertions_satisfied_from_vector(vector_ordered))
print("Swapped vector satisfies the assertion: ")
print(model.assertions_satisfied_from_vector(vector_swapped))

"""
Two practical notes. An assertion becomes part of the model's unique identifier, so a model with an assertion
writes to a different output folder than the same model without one; they are different models, and results from
one should never be resumed as the other.

The degeneracy can also be broken without an assertion, by giving the two components priors that cannot overlap.
Give `gaussian_0.sigma` a `UniformPrior` between 0.0 and 3.0 and `gaussian_1.sigma` one between 3.0 and 25.0, and
the mirror solution is not in parameter space at all. Use an assertion when you know the ordering but not the
values.

Let's refit with the assertion in place.
"""
print(
    """
    The non-linear search has begun running.
    This Jupyter notebook cell with progress once the search has completed - this could take a few minutes!
    """
)

search = af.DynestyStatic(
    sample="rwalk",
)

start = time.time()

result_assert = search.fit(model=model, analysis=analysis)

print(f"Dynesty run time: {time.time() - start} seconds")
print("The search has finished run - you may now continue the notebook.")

"""
The counts printed below are the crisp demonstration. Before the assertion, samples appeared on both sides of the
diagonal; with it in place the violating side is empty, and empty by construction rather than luck, because every
model proposed there was handed the resample figure of merit instead of a likelihood.

The answer is now unambiguous: `gaussian_0` is the narrow component because we said so, and the maximum likelihood
model comes back in that order every run rather than on a coin toss. Half our parameter space has gone, and with it
half the work the search had to do.

Removing a degeneracy is the single highest value thing you can do to a slow model-fit!
"""
aplt.corner_cornerpy(samples=result_assert.samples)

sigma_0_list = [parameters[2] for parameters in result_assert.samples.parameter_lists]
sigma_1_list = [parameters[5] for parameters in result_assert.samples.parameter_lists]

plt.scatter(sigma_0_list, sigma_1_list, s=1, color="k")
plt.title("Samples of the two sigmas with the assertion applied.")
plt.xlabel("gaussian_0 sigma")
plt.ylabel("gaussian_1 sigma")
plt.show()
plt.clf()

ordered_count = sum(
    sigma_0 < sigma_1 for sigma_0, sigma_1 in zip(sigma_0_list, sigma_1_list)
)

print(f"Samples with gaussian_0 narrower than gaussian_1: {ordered_count}")
print(f"Samples that violate the assertion: {len(sigma_0_list) - ordered_count}")

print(result_assert.info)

"""
__Plateaus__

The second detail is what happens when a parameter stops mattering.

To create the problem we now load the single Gaussian dataset, `gaussian_x1`, which was simulated from one Gaussian
with a centre of 50.0, a normalization of 25.0 and a sigma of 10.0. It is the dataset we fitted in tutorials 2 and
3. We will then deliberately fit it with the two component model we have been using, which is one component more
than the data needs.
"""
dataset_x1_path = path.join("dataset", "example_1d", "gaussian_x1")

if not path.exists(dataset_x1_path):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/simulators/simulators.py"],
        check=True,
    )

data_x1 = af.util.numpy_array_from_json(
    file_path=path.join(dataset_x1_path, "data.json")
)
noise_map_x1 = af.util.numpy_array_from_json(
    file_path=path.join(dataset_x1_path, "noise_map.json")
)

analysis_x1 = Analysis(data=data_x1, noise_map=noise_map_x1)

plt.errorbar(
    xvalues,
    data_x1,
    yerr=noise_map_x1,
    linestyle="",
    color="k",
    ecolor="k",
    elinewidth=1,
    capsize=2,
)
plt.title("1D dataset containing a single Gaussian.")
plt.xlabel("x values of profile")
plt.ylabel("Profile Normalization")
plt.show()
plt.clf()

"""
The data needs only one Gaussian, so the best thing the second component can do is get out of the way, which it
does by taking a normalization of zero and contributing nothing to the model data.

Here is the consequence. Once `gaussian_1.normalization` is zero, its `centre` and `sigma` have no effect on the
model data at all. Every value gives the same model data, chi-squared and log likelihood. Instead of a peak in those
two dimensions, parameter space contains a perfectly flat region: a "plateau".

Let's demonstrate it, evaluating the likelihood with `gaussian_1.normalization` fixed at zero and its `centre` and
`sigma` set to three wildly different pairs of values.
"""
model_x1 = af.Collection(gaussian_0=af.Model(Gaussian), gaussian_1=af.Model(Gaussian))

for gaussian in [model_x1.gaussian_0, model_x1.gaussian_1]:
    gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    gaussian.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=25.0)

for centre_1, sigma_1 in [(20.0, 2.0), (80.0, 15.0), (50.0, 0.5)]:
    vector = [50.0, 25.0, 10.0, centre_1, 0.0, sigma_1]

    instance = model_x1.instance_from_vector(vector=vector)

    print(
        f"gaussian_1 centre = {centre_1}, sigma = {sigma_1}, log likelihood = "
        f"{analysis_x1.log_likelihood_function(instance=instance)}"
    )

"""
Three completely different models, one identical log likelihood. That is a plateau, two dimensions wide.

Every search in tutorial 3 struggles here, because nothing in the likelihood tells it which way to go. An MLE search
estimates the gradient, finds it is exactly zero and concludes it has converged. MCMC walkers wander at random,
since every proposal is equally good. A nested sampler's live points scatter uniformly across the flat region.

Worse, the "errors" reported for the plateau parameters are not errors. If a parameter has no effect on the
likelihood its posterior is just its prior, so the error bar is the width of the prior you happened to choose. It
looks like a measurement and is nothing of the sort.

There are three ways to deal with a plateau, in order of increasing honesty. The first is to stop the normalization
reaching zero: a `LogUniformPrior`, used in tutorial 4 for prior tuning, spreads its probability evenly across
orders of magnitude and has a strictly positive lower limit, so a normalization of exactly zero is not in parameter
space.
"""
model_x1.gaussian_1.normalization = af.LogUniformPrior(
    lower_limit=1e-2, upper_limit=1e2
)

print(model_x1.gaussian_1.normalization)

"""
The second is to remove the offending parameters altogether, by fixing them. As tutorial 1 showed, assigning a
float to a model parameter turns it into a constant rather than a free parameter, which we confirm by checking
`prior_count` before and after. If a parameter genuinely has no effect on the data, any value will do.
"""
model_fixed = af.Collection(
    gaussian_0=af.Model(Gaussian), gaussian_1=af.Model(Gaussian)
)

print(f"Number of free parameters before fixing sigma: {model_fixed.prior_count}")

model_fixed.gaussian_1.sigma = 5.0

print(f"Number of free parameters after fixing sigma: {model_fixed.prior_count}")

"""
The third is the honest answer, and the one I would use in a real analysis: reduce the model to what the data
supports, the "reducing complexity" strategy of tutorial 4. The data contains one Gaussian, so the model should
contain one Gaussian, and then there is no second normalization to go to zero and no plateau at all.

The reason to know the other two is that you cannot always do this. Sometimes the extra component is physically
required even when this dataset cannot constrain it, and a sensible prior or a fixed parameter is how you stop it
poisoning the fit.
"""
model_simple = af.Collection(gaussian=af.Model(Gaussian))

print(f"Number of free parameters in the reduced model: {model_simple.prior_count}")

"""
__Comparing Searches__

We now have two badly parameterized problems: the flip degeneracy on the two Gaussian data, and the plateau on the
single Gaussian data. We have already run nested sampling on the flip problem. Let's run MCMC on it too, then run
all three families on the plateau problem, and compare what comes back.

We start with MCMC, using `Emcee` on the flip problem exactly as tutorial 3 did. The walkers begin scattered
through parameter space and climb towards whichever mirror peak is nearest.
"""
model_emcee = af.Collection(
    gaussian_0=af.Model(Gaussian), gaussian_1=af.Model(Gaussian)
)

for gaussian in [model_emcee.gaussian_0, model_emcee.gaussian_1]:
    gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    gaussian.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=25.0)

search = af.Emcee(
    nwalkers=20,  # The number of walkers we'll use to sample parameter space.
    nsteps=500,  # The number of steps each walker takes.
)

print(
    """
    The non-linear search has begun running.
    This Jupyter notebook cell with progress once the search has completed - this could take a few minutes!
    """
)

start = time.time()

result_emcee = search.fit(model=model_emcee, analysis=analysis)

print(f"Emcee run time: {time.time() - start} seconds")
print("The search has finished run - you may now continue the notebook.")

print(result_emcee.info)

"""
Compare that maximum likelihood model to the one the nested sampler found. With only 20 walkers taking 500 steps
they have not had long enough to climb either mirror properly, and they usually finish somewhere that is neither
mirror: the two normalizations and sigmas come back at values the simulator never used, sharing the flux out
between the components in whatever way the walkers happened to drift into.

This is exactly the stochasticity of tutorial 4, and it deserves to be taken seriously rather than treated as a
nuisance. Run the cell above again and you will get a different answer, because the walkers start in different
random places. **That variability is itself the diagnostic.** A fit whose answer changes every run is telling you
the search has not converged on this parameter space, and the honest response is more steps, a better starting
point, or an easier parameter space.

Now the plateau problem, with the same two searches. Nested sampling first, with more live points than we used
above so that it maps the flat region properly rather than skating over it.

Note that the two component model has the flip degeneracy in it as well as the plateau, because either component
could be the one that vanishes. We therefore use the tool we just learned and assert that `gaussian_0` carries more
normalization than `gaussian_1`, which makes `gaussian_1` the vanishing component by definition and lets us talk
about its parameters without having to check which slot the search happened to use.
"""
model_plateau_nest = af.Collection(
    gaussian_0=af.Model(Gaussian), gaussian_1=af.Model(Gaussian)
)

for gaussian in [model_plateau_nest.gaussian_0, model_plateau_nest.gaussian_1]:
    gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    gaussian.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=25.0)

model_plateau_nest.add_assertion(
    model_plateau_nest.gaussian_0.normalization
    > model_plateau_nest.gaussian_1.normalization
)

search = af.DynestyStatic(
    nlive=200,  # More live points than before, so the flat region is properly sampled.
    sample="rwalk",
)

print(
    """
    The non-linear search has begun running.
    This Jupyter notebook cell with progress once the search has completed - this could take a few minutes!
    """
)

start = time.time()

result_plateau_nest = search.fit(model=model_plateau_nest, analysis=analysis_x1)

print(f"Dynesty run time: {time.time() - start} seconds")
print("The search has finished run - you may now continue the notebook.")

print(result_plateau_nest.info)

"""
The result above is the plateau made visible in numbers, so it is worth reading carefully.

`gaussian_0` has recovered the true Gaussian, with a centre near 50.0, a normalization near 25.0 and a sigma near
10.0. `gaussian_1`, the component our assertion made the smaller one, has been pushed down to a small remnant whose
normalization interval runs down towards zero. The data needed one Gaussian and the search gave it one.

Now compare the parameter uncertainties of the two components. The cell below prints the one sigma interval of
every parameter, which as tutorial 5 showed comes from `values_at_sigma`.
"""
names = result_plateau_nest.samples.model.model_component_and_parameter_names
values = result_plateau_nest.samples.values_at_sigma(sigma=1.0, as_instance=False)

for name, (lower, upper) in zip(names, values):
    print(f"{name}: {lower} -> {upper}  (width {upper - lower})")

"""
Compare the two components line by line, and compare the widths rather than the values, because the values move
from run to run. `gaussian_0`, which recovered the real Gaussian, has a tight interval on its centre and a tighter
one still on its sigma. `gaussian_1`, which contributes almost nothing, has intervals on the same two parameters
that are many times wider, with its sigma spanning a large part of the 0.0 to 25.0 prior we gave it.

That is the plateau statement made quantitative. The less a component contributes, the less its shape parameters
matter, so their posterior drifts back towards the prior and the "error bar" you would quote drifts towards the
prior width. It looks like a measurement and it is not one!

Now MCMC on the same plateau problem.
"""
model_plateau_mcmc = af.Collection(
    gaussian_0=af.Model(Gaussian), gaussian_1=af.Model(Gaussian)
)

for gaussian in [model_plateau_mcmc.gaussian_0, model_plateau_mcmc.gaussian_1]:
    gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    gaussian.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=25.0)

search = af.Emcee(
    nwalkers=20,
    nsteps=500,
)

print(
    """
    The non-linear search has begun running.
    This Jupyter notebook cell with progress once the search has completed - this could take a few minutes!
    """
)

start = time.time()

result_plateau_mcmc = search.fit(model=model_plateau_mcmc, analysis=analysis_x1)

print(f"Emcee run time: {time.time() - start} seconds")
print("The search has finished run - you may now continue the notebook.")

print(result_plateau_mcmc.info)

"""
Finally the gradient search. As tutorial 6 explained, `MultiStartAdam` differentiates the likelihood with JAX, so
its `Analysis` must be created with `use_jax=True`, and it walks many independent "lanes" uphill from different
starting points.
"""
model_plateau = af.Collection(
    gaussian_0=af.Model(Gaussian), gaussian_1=af.Model(Gaussian)
)

for gaussian in [model_plateau.gaussian_0, model_plateau.gaussian_1]:
    gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    gaussian.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=25.0)

analysis_x1_jax = Analysis(data=data_x1, noise_map=noise_map_x1, use_jax=True)

search = af.MultiStartAdam(
    n_starts=8,  # The number of independent lanes launched through parameter space.
    n_steps=150,  # The number of gradient steps each lane takes.
    learning_rate=0.5,
)

print(
    """
    The non-linear search has begun running.
    This Jupyter notebook cell with progress once the search has completed - this could take a few minutes!
    """
)

start = time.time()

result_adam = search.fit(model=model_plateau, analysis=analysis_x1_jax)

print(f"MultiStartAdam run time: {time.time() - start} seconds")
print("The search has finished run - you may now continue the notebook.")

print(result_adam.info)

"""
Put the results side by side. I want to separate two things carefully here: what these particular short runs
actually showed, and what we know about the mechanisms independently of them.

**What the runs above showed.** On the flip problem, nested sampling committed to one mirror and fitted it
excellently, while MCMC, given only 20 walkers and 500 steps, typically finished at a model worse than either
mirror and gave a different answer each time it was run.

On the plateau problem, nested sampling recovered the true Gaussian, pushed the spare component down towards zero
normalization, and reported intervals on that component's shape parameters many times wider than on the real one's.
MCMC also found roughly the right Gaussian, and left its spare component wandering, often finishing with its sigma
against the edge of the prior, which is what a walker with nothing to climb does. The gradient search converged quickly, and where
it converged depended on where its lanes started: some runs recover the true Gaussian, and some settle on a poor
local maximum with a broad component parked near the edge of its prior. Compare the result above with the true
values to see which you got.

Notice that none of the four ever announced a problem. Every one ran to completion and returned a result with error
bars.

**What we know from the mechanisms.** A plateau is a region of exactly zero gradient, and a zero gradient point is
a fixed point a gradient search cannot leave: a lane stepping onto one stops moving and its figure of merit stops
changing, which from the outside is indistinguishable from convergence. A local maximum is a zero gradient point
too, which is why the run above looks exactly as converged as a correct one would. Nested sampling has no such
trap, because its live points are replaced from the whole prior volume rather than moved uphill from where they
are. MCMC sits in between: a walker on a flat region accepts every proposal, so it wanders rather than freezing,
but it may wander for a very long time.

None of this says which search is best, and that is not the point. If searches with completely different mechanisms
agree, the answer is probably a property of your data. If they disagree, or if one gives a different answer every
run, that is almost never a statement about which search is better. It is a statement about your parameter space,
and the degeneracies and flat regions you have accidentally put in it.

Running more than one search tells you about your model, not just your searches!
"""

r"""
__Clipping__

So far every model has been computable everywhere: any combination of values inside the priors gave a number.

That is not always true. Consider a different model of our two Gaussian data, in which the components are not
independent. Suppose the narrow component is an instrumental effect blurred into the broad one, so the width we
want for the broad component is what is left after the narrow one is removed in quadrature:

\[ \sigma_{\rm halo} = \sqrt{\sigma_{1}^2 - \sigma_{0}^2} \]

Where:

- $\sigma_{0}$: the sigma of the narrow component, `gaussian_0`.
- $\sigma_{1}$: the sigma of the broad component, `gaussian_1`.
- $\sigma_{\rm halo}$: the width of the broad component after the narrow one is removed.

In words: the broad component's true width is the broad width with the narrow width subtracted in quadrature.

Now look at the priors. `gaussian_0.sigma` runs from 0.0 to 25.0 and so does `gaussian_1.sigma`, both perfectly
sensible ranges. But the *combination* $\sigma_{1} < \sigma_{0}$ asks for the square root of a negative number,
which is not a number at all. The individual priors are fine, the combination is unphysical.
"""


class DeconvolvedAnalysis(Analysis):
    def model_data_from_instance(self, instance):
        """
        Returns the summed model data of a narrow component plus a broad component whose width is the quadrature
        difference of the two sigmas.

        This is unphysical, and undefined, whenever the sigma of `gaussian_1` is smaller than the sigma of
        `gaussian_0`.
        """
        xp = self._xp

        xvalues = xp.arange(self.data.shape[0])

        sigma_halo = xp.sqrt(
            instance.gaussian_1.sigma**2.0 - instance.gaussian_0.sigma**2.0
        )

        halo = Gaussian(
            centre=instance.gaussian_1.centre,
            normalization=instance.gaussian_1.normalization,
            sigma=sigma_halo,
        )

        return instance.gaussian_0.model_data_from(
            xvalues=xvalues, xp=xp
        ) + halo.model_data_from(xvalues=xvalues, xp=xp)


"""
Let's evaluate this likelihood at a physical set of parameters and at an unphysical one, where the only difference
is that the two sigmas have been swapped so the "narrow" component is wider than the "broad" one.
"""
analysis_nan = DeconvolvedAnalysis(data=data, noise_map=noise_map)

model_nan = af.Collection(gaussian_0=af.Model(Gaussian), gaussian_1=af.Model(Gaussian))

for gaussian in [model_nan.gaussian_0, model_nan.gaussian_1]:
    gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    gaussian.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=25.0)

vector_physical = [50.0, 20.0, 1.0, 50.0, 40.0, 5.0]
vector_unphysical = [50.0, 20.0, 5.0, 50.0, 40.0, 1.0]

print("Log likelihood of the physical model: ")
print(
    analysis_nan.log_likelihood_function(
        instance=model_nan.instance_from_vector(vector=vector_physical)
    )
)
print("Log likelihood of the unphysical model: ")
print(
    analysis_nan.log_likelihood_function(
        instance=model_nan.instance_from_vector(vector=vector_unphysical)
    )
)

"""
The second log likelihood is `nan`, short for "not a number": what floating point arithmetic produces when asked to
do something impossible. NumPy also prints a `RuntimeWarning: invalid value encountered in sqrt`, which is expected
here and not a bug. A `nan` is contagious, in that any sum or product involving one gives another `nan`.

A search receiving a `nan` could not rank it against anything else, so **PyAutoFit** never lets one through. Before
the figure of merit is returned it is checked for `nan` and for infinity, and either is replaced by the "resample
figure of merit" we met in the assertions section: this model is unusable, draw another.

The detail worth knowing is that the resample figure of merit is not the same number for every search. For the MLE
and MCMC searches it is `-inf`, negative infinity, the natural choice because it is the worst possible log
likelihood: an MLE search never steps towards it and an MCMC walker never accepts it.

For the nested samplers and for SMC it is `-1e99` instead, a huge negative number that is nevertheless finite. Their
arithmetic with log weights and shell evidences has to stay finite, so a genuine `-inf` would poison the whole
calculation rather than reject one sample. You do not choose these values; each search sets its own internally, and
there is no user facing argument for them.

There is one family of search for which rejection is not enough. As tutorial 6 showed, the gradient searches step in
*physical* parameter units, computing a gradient and moving along it. A large enough step carries a lane out of a
prior's range, where the log prior of a `UniformPrior` is `-inf`. An MCMC search rejects such a proposal and the
walker stays put; a nested sampler never leaves the unit cube. A gradient step has neither protection: nothing pulls
the lane back, because outside the box the log prior is a *constant* `-inf` and the derivative of a constant is
zero, so the lane keeps walking away.

That is why the gradient searches take a `clipper` argument. A "clipper" projects each lane back inside the prior
box after every step, supplying the restoring force that rejection gives the other searches.
"""
clipper = af.ClipperPriorBox()

print(f"Clipper: {type(clipper).__name__}")

"""
__NaN Diagnostics__

When you run a gradient search, **PyAutoFit** counts how often the likelihood and the gradient went non finite and
writes the counts to a file called `search.summary` in the output folder. To read that file we need the search to
write to disk, which as tutorial 5 showed means passing a `name` and a `path_prefix`.

Let's fit our unphysical model with `MultiStartAdam`, using the clipper above, and then read the summary back.
"""
analysis_nan_jax = DeconvolvedAnalysis(data=data, noise_map=noise_map, use_jax=True)

model_nan_fit = af.Collection(
    gaussian_0=af.Model(Gaussian), gaussian_1=af.Model(Gaussian)
)

for gaussian in [model_nan_fit.gaussian_0, model_nan_fit.gaussian_1]:
    gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    gaussian.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=25.0)

search = af.MultiStartAdam(
    name="tutorial_7_the_details_nan",
    path_prefix="chapter_1_introduction",
    n_starts=8,
    n_steps=150,
    learning_rate=0.5,
    clipper=clipper,
)

print(
    """
    The non-linear search has begun running.
    Checkout the HowToFit/output/chapter_1_introduction/tutorial_7_the_details_nan
    folder for live output of the results.
    This Jupyter notebook cell with progress once the search has completed - this could take a few minutes!
    """
)

start = time.time()

result_nan = search.fit(model=model_nan_fit, analysis=analysis_nan_jax)

print(f"MultiStartAdam run time: {time.time() - start} seconds")
print("The search has finished run - you may now continue the notebook.")

summary_path = path.join(str(search.paths.output_path), "search.summary")

if path.exists(summary_path):
    with open(summary_path) as summary_file:
        print(summary_file.read())

"""
The block headed `Resampling Info` is the one we care about, and its two counters mean different things.

`Value-NaN Lane-Steps` counts steps on which the likelihood itself was undefined, which happens whenever a lane
wanders into the region where `gaussian_1.sigma` is smaller than `gaussian_0.sigma`: the square root of a negative
number, a `nan` value, replaced by the resample figure of merit.

`Gradient-NaN Lane-Steps` counts something subtler: steps on which the likelihood *value* was fine but its
*gradient* was not. Such a lane does not die and reports no error. The optimizer sees a non finite update, refuses
to apply it, and the lane stays exactly where it is, then does the same on the next step. A lane whose gradient is
`nan` is frozen, and a frozen lane is indistinguishable from a converged one in the figure of merit trace!

`Resurrections` counts lanes restarted from a fresh point after dying, and the rate lines express the counters as a
fraction of the total lane-steps taken.

The two counters are deliberately disjoint: a step is counted as a gradient-NaN only if its value was finite, so in
the run above, where every failure was an undefined square root, the gradient-NaN count is zero. Do not read that
as "gradient NaNs cannot happen here". How can a value be finite while its gradient is not? Here is the smallest
example, and it matters because the same pattern occurs constantly in real likelihood functions.
"""
import jax
import jax.numpy as jnp

unsafe = lambda x: jnp.where(x > 0.0, jnp.sqrt(x), 0.0)

print("Unsafe function at x = -1.0:")
print(f"value    = {unsafe(-1.0)}")
print(f"gradient = {jax.grad(unsafe)(-1.0)}")

"""
The value is `0.0`, exactly as intended, because the `where` selected the safe branch. The gradient is `nan`.

Reverse mode automatic differentiation, which is how JAX computes gradients, differentiates *both* branches of a
`where` and multiplies the unselected one by zero. The derivative of `sqrt(x)` at `x = -1` is `nan`, and zero times
`nan` is `nan`, not zero. The mask protected the value and did nothing for the gradient.

The fix is never to evaluate the offending operation at the invalid input, which takes a second `where` inside the
first. The inner `where` replaces the invalid input with a harmless one, so `sqrt` is only called on a positive
number, and the outer `where` still selects the right answer.
"""
safe = lambda x: jnp.where(x > 0.0, jnp.sqrt(jnp.where(x > 0.0, x, 1.0)), 0.0)

print("Safe function at x = -1.0:")
print(f"value    = {safe(-1.0)}")
print(f"gradient = {jax.grad(safe)(-1.0)}")

"""
Same value, and now a finite gradient.

The lesson generalizes far beyond this function. A guard that checks the result of a calculation can only protect
the value, because by the time it runs the damage to the derivative is already recorded. Safety has to live where
the `nan` is created, not where it is detected.

A good likelihood value does not mean a good gradient!

__Unit Cube Vs Physical__

The last detail is the most fundamental, and it changes how you think about every prior you have written.

In tutorial 1 we mapped a vector of physical parameter values onto a model instance with `instance_from_vector`, and
in tutorial 3 we gave each parameter a prior. Those ideas are closer than they look, because a non-linear search
does not work in physical units at all.

What a search explores is the "unit cube": a space where every parameter runs from 0.0 to 1.0, whatever it means and
whatever units it has, and the prior is the function converting a unit value into a physical one. **PyAutoFit**
exposes this mapping directly, so let's build a model with three deliberately different priors and ask what the
exact centre of the unit cube corresponds to.
"""
model_cube = af.Model(Gaussian)

model_cube.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model_cube.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
model_cube.sigma = af.GaussianPrior(mean=10.0, sigma=5.0)

print(model_cube.model_component_and_parameter_names)

print("Physical vector at the centre of the unit cube:")
print(model_cube.vector_from_unit_vector(unit_vector=[0.5, 0.5, 0.5]))

instance_cube = model_cube.instance_from_unit_vector(unit_vector=[0.5, 0.5, 0.5])

print(f"Centre = {instance_cube.centre}")
print(f"Normalization = {instance_cube.normalization}")
print(f"Sigma = {instance_cube.sigma}")

"""
Look carefully at the second number. A unit value of 0.5 mapped `centre` to 50.0, which is what you would expect
from a `UniformPrior` between 0.0 and 100.0. But the same unit value mapped `normalization` to 1.0, not 50.005.

That is because a `LogUniformPrior` between 0.01 and 100.0 puts the *geometric* midpoint at the centre of the cube,
and the geometric midpoint of 0.01 and 100.0 is 1.0. Let's see it directly, comparing the two priors a quarter of
the way through the cube as well.
"""
print(
    f"UniformPrior at unit 0.25:    {af.UniformPrior(lower_limit=0.0, upper_limit=100.0).value_for(0.25)}"
)
print(
    f"UniformPrior at unit 0.50:    {af.UniformPrior(lower_limit=0.0, upper_limit=100.0).value_for(0.5)}"
)
print(
    f"LogUniformPrior at unit 0.25: {af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2).value_for(0.25)}"
)
print(
    f"LogUniformPrior at unit 0.50: {af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2).value_for(0.5)}"
)

"""
The uniform prior moves in equal steps of 25.0 across the cube. The log uniform prior moves in equal steps of one
order of magnitude, from 0.1 a quarter of the way through to 1.0 at halfway.

This is the point of the section. The prior is not a filter applied to a search that would otherwise explore
physical space evenly, the prior *is* the geometry of the space the search explores. A log uniform prior makes the
search spend equal effort on each decade, which is what you want for a parameter that could be 0.1 or 100. A
uniform prior over the same range spends most of its effort above 10.0 and would essentially never propose a value
below 1.0.

The searches use this differently. Nested samplers work in the unit cube directly, drawing their live points from
it, which is why they cannot take a starting point and never step outside a prior's range. MLE and MCMC searches
step in physical units instead, which is why they can walk out of a prior box, and why clipping matters for them
and not for nested sampling.

Choosing a prior is not just a statement about what you believe. It is a decision about what the search's parameter
space looks like!

__Summary__

There are many details underneath a model-fit. We have looked at parameterization and mirror solutions, assertions,
plateaus, the resample figure of merit, `nan` values and their gradients, and the unit cube. There are more, and no
tutorial could cover them all.

Most of the time, none of them matter. A sensible model, sensible priors and a sensible search give a sensible
answer, and you could go a whole career without thinking about a single thing in this tutorial. That is a property
of **PyAutoFit**: the defaults are chosen so the ordinary case works.

When they do matter, attending to them buys three things. Inference runs faster, sometimes by a large factor,
because the search wastes no effort on redundant regions of parameter space. It converges on the right solution more
often, because there are fewer wrong places to get stuck. And it protects you from the worst case in model-fitting,
which is not a fit that crashes but a fit that returns the wrong answer and gives you no reason to doubt it.

The tools in this tutorial are how you get a fit to complain out loud instead of failing quietly!

Tutorial 8 moves on from a single fit to the scientific workflow: how to organise, compare and interpret the many
fits a real analysis is made of. There is also an optional tutorial on the Bayesian formalism underlying everything
we have done, which puts the priors, likelihoods and evidences used throughout this chapter on a formal footing.
"""
