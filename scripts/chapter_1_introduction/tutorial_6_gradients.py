"""
Tutorial 6: Gradients
=====================

In tutorial 3, describing how a maximum likelihood estimator works, we said it "evaluates the likelihood at nearby
points to estimate the gradient, determining the direction in which to move up in parameter space". That sentence
slipped by quickly, but it hides one of the most important ideas in model-fitting: the search does not have to be
blind, because the likelihood surface has a slope, and the slope tells us which way is up.

In tutorial 5 we saw the other half of the story: errors, computed by marginalizing over the Probability Density
Function of the samples. Errors came from samples, and samples are expensive. Here we make the gradient explicit,
hand it to three types of non-linear search, and end by computing errors without drawing a single sample.

__Overview__

In this tutorial, we will introduce gradients and use them to fit a 1D Gaussian profile to noisy data. Specifically,
we will:

- Introduce the "gradient" of the log likelihood and plot it on a slice through parameter space.

- Compute a gradient by "finite differencing", and see why the answer depends on the step size.

- Use JAX to differentiate the `log_likelihood_function` exactly, via "automatic differentiation".

- Compare searches which use gradients with those which do not.

- Estimate errors from the second derivatives of the log likelihood.

__Contents__

This tutorial is split into the following sections:

- **Data**: Load and plot the 1D Gaussian dataset fitted throughout the chapter.
- **Model**: The `Gaussian` model component, now written so it can run on NumPy or JAX.
- **Analysis**: The `Analysis` class with a log likelihood function that can be differentiated.
- **Gradients**: Extending the parameter-space picture of tutorial 3 with the idea that a search can know which way "up" is at every point.
- **Finite Differencing**: Computing a gradient numerically by nudging each parameter and re-evaluating the likelihood.
- **JAX and Autodiff**: Why analytic gradients are impractical, and how JAX differentiates the likelihood function under the hood.
- **Maximum Likelihood Estimation (MLE)**: Comparing a finite-difference optimizer (`LBFGS`) with a multi-start autodiff optimizer (`MultiStartAdam`).
- **Markov Chain Monte Carlo (MCMC)**: Comparing walkers that ignore gradients (`Emcee`) with Hamiltonian sampling that follows them (`BlackJAXNUTS`).
- **Nested Sampling**: Why nested sampling does not use gradients, and how JAX still speeds it up (`Nautilus`).
- **Errors From Curvature**: Using second derivatives at the maximum likelihood solution to estimate parameter errors and comparing them with the errors from samples in tutorial 5.
- **Wrap Up**: Concluding the tutorial with a recap of how gradients change each search.
"""

# from autofit import setup_notebook; setup_notebook()

from os import path
import matplotlib.pyplot as plt
import numpy as np

import autofit as af
import autofit.plot as aplt

"""
__Data__

Load and plot the dataset from the `HowToFit/dataset` folder.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")

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
plt.title("1D Gaussian dataset.")
plt.xlabel("x values of profile")
plt.ylabel("Profile Normalization")
plt.show()
plt.clf()

"""
__Model__

This is the same dataset we fitted in tutorials 2 and 3, whose true parameters are `centre=50.0`,
`normalization=25.0` and `sigma=10.0`. We lean on those values here, because a gradient is always a gradient *at a
point*, and the truth is the most interesting point to stand at.

We re-paste the `Gaussian` class in full, as the chapter always does. There is exactly one change from tutorials 1
to 5: `model_data_from` takes an extra argument `xp`, which defaults to `np`, and every operation inside it is
written as `xp.subtract`, `xp.multiply`, `xp.divide`, `xp.sqrt`, `xp.exp` and `xp.square`. The `xp` argument asks
"which array library should I compute with?", and later we pass it a library which gives us gradients for free.
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
            The array library the calculation is performed with, which is NumPy by default and JAX when the
            model data is being differentiated.

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
We compose the model exactly as in tutorial 3, with uniform priors. They are wider than tutorial 3's, where
`normalization` and `sigma` ran from 0.0 to 10.0, so the true values sit inside parameter space rather than on its
edge.

We also print `model.model_component_and_parameter_names`, the order of the parameters when the model is expressed
as a plain vector. This matters throughout, because gradients are vectors too and their entries share that order.
"""
model = af.Model(Gaussian)

model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=25.0)

print("Order of the parameters in a model vector:")
print(model.model_component_and_parameter_names)

"""
__Analysis__

The `Analysis` class plays the same role it did in tutorial 3, receiving the data and defining the
`log_likelihood_function`. There are two small changes, and they are the reason this tutorial exists.

First, `__init__` takes a `use_jax` input which it passes to `super().__init__(use_jax=use_jax)`, the switch saying
which array library the analysis computes with: NumPy when `use_jax=False`, the default, and JAX when `use_jax=True`.

Second, the `log_likelihood_function` uses `self._xp` rather than `np`. `self._xp` is the library that switch
selected, passed into the `Gaussian` via its `xp` argument so the whole calculation uses one library. Note we write
`self._xp.sum(...)` and not the builtin `sum(...)`, which iterates the array one element at a time. The environment
variable `PYAUTO_DISABLE_JAX=1` forces every analysis back to NumPy.
"""


class Analysis(af.Analysis):
    def __init__(self, data: np.ndarray, noise_map: np.ndarray, use_jax: bool = False):
        """
        The `Analysis` class acts as an interface between the data and model in **PyAutoFit**.

        Its `log_likelihood_function` defines how the model is fitted to the data and it is called many times by
        the non-linear search fitting algorithm.

        Parameters
        ----------
        data
            A 1D numpy array containing the data (e.g. a noisy 1D signal) fitted in the workspace examples.
        noise_map
            A 1D numpy array containing the noise values of the data, used for computing the goodness of fit
            metric, the log likelihood.
        use_jax
            If `True` the log likelihood is computed with JAX, meaning it can be differentiated, and if `False`
            it is computed with NumPy.
        """
        super().__init__(use_jax=use_jax)

        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance) -> float:
        """
        Returns the log likelihood of a fit of a 1D Gaussian to the dataset.

        The calculation is identical to the one performed in tutorial 3, except every operation is performed with
        `self._xp`, the array library chosen by the `use_jax` input, instead of NumPy.
        """
        xp = self._xp

        xvalues = xp.arange(self.data.shape[0])

        model_data = instance.model_data_from(xvalues=xvalues, xp=xp)
        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        chi_squared = xp.sum(chi_squared_map)
        noise_normalization = xp.sum(xp.log(2 * np.pi * self.noise_map**2.0))
        log_likelihood = -0.5 * (chi_squared + noise_normalization)

        return log_likelihood


"""
We create two analysis objects from the same data, one computing with NumPy and one with JAX. They compute the same
number. We use the NumPy one for the next two sections, and the JAX one when we start differentiating.
"""
analysis = Analysis(data=data, noise_map=noise_map)
analysis_jax = Analysis(data=data, noise_map=noise_map, use_jax=True)

instance = model.instance_from_vector(vector=[50.0, 25.0, 10.0])

print("Log likelihood of the true model, computed with NumPy:")
print(analysis.log_likelihood_function(instance=instance))

r"""
__Gradients__

In tutorial 3 we pictured model-fitting as an exploration of a "parameter space": a surface, one axis per free
parameter, whose height is the log likelihood. Our job was to find its peak, and the searches we used explored it
essentially blind, evaluating points and remembering which came back high.

But a surface has more structure than that. At any point there is a slope: walk a tiny distance in the `centre`
direction and the log likelihood changes by some amount, walk the same distance in `sigma` and it changes by a
different amount. One such number per parameter gives the "gradient", a vector pointing in the direction the log
likelihood increases fastest, whose length says how steeply.

The gradient of a function \( f \) with respect to parameters \( x, y, z \) is written:

\[ \nabla f = \left( \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z} \right) \]

Where:

- \( \nabla f \): the gradient, a vector with one entry per parameter.
- \( \partial f / \partial x \): the "partial derivative" of \( f \) with respect to \( x \), how much \( f \)
  changes when \( x \) changes and everything else is held fixed.

The gradient is a list of slopes, one per parameter, measured at the same point. A search which knows it does not
have to guess which way to move, it can simply walk uphill!

Below we fix `normalization` and `sigma` at their true values and vary only `centre`, a 1D "slice" through the three
dimensional parameter space.
"""
centre_list = np.linspace(40.0, 60.0, 100)

log_likelihood_list = []

for centre in centre_list:
    instance = model.instance_from_vector(vector=[centre, 25.0, 10.0])
    log_likelihood_list.append(analysis.log_likelihood_function(instance=instance))

plt.plot(centre_list, log_likelihood_list, color="k")
plt.title("Log likelihood as a function of the Gaussian centre.")
plt.xlabel("centre")
plt.ylabel("Log Likelihood")
plt.show()
plt.close()

"""
The slice is a smooth hill peaking almost exactly at the true `centre` of 50.0, falling away steeply because an
offset Gaussian produces enormous residuals.

Now the important part. Pick a point on that hill, say `centre=44.0`, and ask "if I step a little to the right, does
the log likelihood go up or down, and by how much?". The answer is the slope of the curve there, drawn below as the
straight line touching the curve at that point, the "tangent line".
"""
centre_point = 44.0
step = 0.01

instance_up = model.instance_from_vector(vector=[centre_point + step, 25.0, 10.0])
instance_down = model.instance_from_vector(vector=[centre_point - step, 25.0, 10.0])

slope = (
    analysis.log_likelihood_function(instance=instance_up)
    - analysis.log_likelihood_function(instance=instance_down)
) / (2.0 * step)

instance_point = model.instance_from_vector(vector=[centre_point, 25.0, 10.0])
log_likelihood_point = analysis.log_likelihood_function(instance=instance_point)

print(f"Slope of the log likelihood at centre = {centre_point}: {slope}")

tangent_centre_list = np.linspace(centre_point - 3.0, centre_point + 3.0, 10)
tangent_list = log_likelihood_point + slope * (tangent_centre_list - centre_point)

plt.plot(centre_list, log_likelihood_list, color="k")
plt.plot(tangent_centre_list, tangent_list, color="r")
plt.scatter(centre_point, log_likelihood_point, color="r")
plt.title("The tangent line shows which way is up at a point.")
plt.xlabel("centre")
plt.ylabel("Log Likelihood")
plt.show()
plt.close()

"""
The red line is the gradient made visible. It slopes upwards to the right and its value is positive, which together
say that increasing `centre` from 44.0 increases the log likelihood. A search standing here does not need to try
both directions to learn that, and the size of the number tells it the surface is steep enough for a decent step.

The same holds for every parameter. Below we slice in `sigma` instead, giving a hill of a different shape and
therefore different slopes.
"""
sigma_list = np.linspace(5.0, 15.0, 100)

log_likelihood_list = []

for sigma in sigma_list:
    instance = model.instance_from_vector(vector=[50.0, 25.0, sigma])
    log_likelihood_list.append(analysis.log_likelihood_function(instance=instance))

plt.plot(sigma_list, log_likelihood_list, color="k")
plt.title("Log likelihood as a function of the Gaussian sigma.")
plt.xlabel("sigma")
plt.ylabel("Log Likelihood")
plt.show()
plt.close()

r"""
Our model has three parameters, so its gradient has three entries, one per slice of the kind we just drew, ordered
as the parameter names we printed earlier:

\[ \left( \frac{\partial \log L}{\partial centre}, \frac{\partial \log L}{\partial normalization}, \frac{\partial \log L}{\partial \sigma} \right) \]

The five-Gaussian model of tutorial 4 has a gradient with 15 entries; a lens model in astronomy might have 50. It
always has as many entries as the model has free parameters, and always belongs to one specific point. Move to a
new point and you get a new gradient!

__Finite Differencing__

So how do we compute a gradient? The most direct way is the one we just used for the tangent line: nudge a parameter
by a small amount \( h \), evaluate the log likelihood, nudge it the other way, evaluate again, and take the
difference. This is "finite differencing", in the "central difference" form:

\[ \frac{\partial f}{\partial x} \approx \frac{f(x + h) - f(x - h)}{2h} \]

Where:

- \( h \): the step size, a small number chosen by us.
- \( f(x + h) \): the log likelihood with the parameter increased by \( h \), everything else held fixed.
- \( f(x - h) \): the log likelihood with the parameter decreased by \( h \).

In words: measure how much the function changed over a small interval and divide by the width of that interval.
Below we write this out for all three parameters, returning a list of three slopes: a gradient.
"""


def log_likelihood_from_vector_numpy(vector):
    """
    Returns the log likelihood of a model defined by an input vector of parameter values, computed with NumPy.
    """
    instance = model.instance_from_vector(vector=list(vector))

    return analysis.log_likelihood_function(instance=instance)


def gradient_via_finite_difference_from(vector, h=1.0e-3):
    """
    Returns the gradient of the log likelihood at an input vector of parameter values, computed via central finite
    differences with a step size `h`.
    """
    gradient = []

    for index in range(len(vector)):
        vector_up = list(vector)
        vector_up[index] += h

        vector_down = list(vector)
        vector_down[index] -= h

        log_likelihood_up = log_likelihood_from_vector_numpy(vector=vector_up)
        log_likelihood_down = log_likelihood_from_vector_numpy(vector=vector_down)

        gradient.append((log_likelihood_up - log_likelihood_down) / (2.0 * h))

    return gradient


"""
We evaluate the gradient at a point some way from the true solution, `centre=44.0`, `normalization=20.0`
and `sigma=8.0`, roughly where a search might find itself part way through a fit.
"""
vector = [44.0, 20.0, 8.0]

gradient = gradient_via_finite_difference_from(vector=vector)

print("Gradient of the log likelihood at [44.0, 20.0, 8.0]:\n")
print(f"d(log likelihood) / d(centre)        = {gradient[0]}")
print(f"d(log likelihood) / d(normalization) = {gradient[1]}")
print(f"d(log likelihood) / d(sigma)         = {gradient[2]}")

r"""
The signs tell a search to increase `centre`, decrease `normalization` and increase `sigma`. Compared to the true
values of 50.0, 25.0 and 10.0 that is sensible in two of the three parameters: the gradient is a local instruction,
under no obligation to point straight at the peak.

There is a catch. We had to choose \( h \), and the answer depends on that choice. Below we compute the gradient
with respect to `centre` for step sizes spanning ten orders of magnitude.
"""
print("Step size h        d(log likelihood) / d(centre)\n")

for h in [10.0, 1.0, 0.1, 1.0e-3, 1.0e-6, 1.0e-9]:
    gradient = gradient_via_finite_difference_from(vector=vector, h=h)

    print(f"{h:<18.0e} {gradient[0]}")

r"""
The table shows the two failure modes of finite differencing, one at each end.

When \( h \) is too large, the two evaluations are so far apart that the line between them is not the slope at our
point at all, it is the average slope over a wide interval. The answer is "biased".

When \( h \) is too small, the two log likelihoods subtracted are almost identical numbers. A computer stores them
to about sixteen significant figures, so subtracting them keeps mostly round-off noise, which dividing by a tiny
number then amplifies. The answer becomes erratic.

Somewhere in the middle is a sweet spot, visible as the region where the answer stops changing, but its location
depends on the units of the parameter, the scale of the data and the model, so it must be found by trial and error
for every new problem. That is not a comfortable position to be in!

There is a second cost. Each parameter needs two evaluations, so a gradient for a model with \( N \) parameters
costs \( 2N \) evaluations, which for a 50-parameter model whose likelihood takes a second is over a minute of
computation to work out which way to take one step.

This is exactly what the `LBFGS` search of tutorial 3 was doing, silently, on your behalf. It "evaluates the
likelihood at nearby points to estimate the gradient", and now we know what that means: finite differencing, with a
step size it chose for us.

__JAX and Autodiff__

If finite differencing is unreliable and expensive, why not write the derivative down? For our 1D Gaussian we could.
But no interesting log likelihood is a short equation. A realistic one is a long chain of operations: in astronomy,
ray tracing light through a gravitational lens, convolving with the telescope's point spread function, solving a
large linear system to reconstruct a source, and only then comparing to data. Differentiating that by hand means
applying the chain rule hundreds of times and maintaining the result every time the model changes. It is
impractical, and error prone in a way which is hard to detect, because a subtly wrong gradient still points roughly
uphill and still produces a plausible looking fit.

"Automatic differentiation", or "autodiff", solves this. Every calculation is built from a handful of elementary
operations whose derivatives we know, so a library which records those operations as they run can apply the chain
rule to that record and produce the exact derivative of whatever was computed.

This is what JAX does, and it is why our `Gaussian` was written in terms of `xp` and our `Analysis` in terms
of `self._xp`. When `use_jax=True`, `xp` is `jax.numpy`, whose operations record themselves as they execute, and the
derivative that comes out is exact and costs a small fixed multiple of one likelihood evaluation, however many
parameters the model has.

Below we build a function mapping a vector of parameters to a log likelihood and ask JAX to differentiate it. That
is the whole API: `jax.grad(f)` returns a function which, called at a point, returns the gradient of `f` there.
"""
import jax
import jax.numpy as jnp


def log_likelihood_from_vector(vector):
    """
    Returns the log likelihood of a model defined by an input vector of parameter values, computed with JAX.
    """
    instance = model.instance_from_vector(vector=vector, xp=jnp)

    return analysis_jax.log_likelihood_function(instance=instance)


gradient_jax = jax.grad(log_likelihood_from_vector)(jnp.array(vector))

gradient_finite_difference = gradient_via_finite_difference_from(vector=vector)

print("Gradient at [44.0, 20.0, 8.0] via automatic differentiation:")
print(gradient_jax)

print("\nGradient at [44.0, 20.0, 8.0] via finite differencing:")
print(gradient_finite_difference)

r"""
The two gradients agree to several decimal places, the sanity check that matters: autodiff computes the same
quantity we computed by hand, exactly and in one pass rather than approximately and in six. Where they disagree it
is the finite-difference answer that is wrong, because it carries the error we chose when we picked \( h \).

JAX has a second trick. `jax.jit` compiles a function into optimised machine code the first time it is called and
reuses that code on every subsequent call, so for a search calling the likelihood thousands of times a one-off
compilation cost is an excellent trade. Below we time a hundred compiled calls against a hundred uncompiled ones.
"""
import time

log_likelihood_jit = jax.jit(log_likelihood_from_vector)

start = time.time()
log_likelihood_jit(jnp.array(vector))
print(f"Time of the first (compiling) call: {time.time() - start} seconds")

start = time.time()
for i in range(100):
    log_likelihood_jit(jnp.array(vector))
print(f"Time of 100 compiled calls: {time.time() - start} seconds")

start = time.time()
for i in range(100):
    log_likelihood_from_vector(jnp.array(vector))
print(f"Time of 100 uncompiled calls: {time.time() - start} seconds")

r"""
__Maximum Likelihood Estimation (MLE)__

We now have gradients. Let us give them to the three families of search we met in tutorial 3.

We start with `LBFGS`, passing it the NumPy analysis, because `LBFGS` cannot use JAX gradients: it estimates the
gradient itself by finite differencing, exactly as we did by hand, so every iteration costs \( 2N + 1 \)
evaluations. It also starts from a single point, the centre of the priors. In tutorial 3 that was enough to trap it
in a local maximum; with the wider priors here it does better, but a single walker walking uphill can still only
find the peak it happens to be standing on.
"""
search = af.LBFGS()

print(
    """
    The non-linear search has begun running.
    This Jupyter notebook cell with progress once the search has completed - this could take a few minutes!
    """
)

start = time.time()

result = search.fit(model=model, analysis=analysis)

print("The search has finished run - you may now continue the notebook.")
print(f"LBFGS run time: {time.time() - start} seconds")

print(result.info)

"""
Now the gradient-aware alternative, `MultiStartAdam`. Two words in that name describe what it does differently.

"Multi start" means it starts not from one point but from many, scattered across the priors. Each start is a "lane"
walking uphill independently, so a lane which walks into a local maximum loses to the lanes which found something
better, addressing the failure mode that trapped `LBFGS` in tutorial 3.

"Adam" is the algorithm each lane steps with, a gradient method used to train neural networks, which follows the
exact autodiff gradient and adapts its step size per parameter. Because the gradients come from JAX, every lane is
advanced together in one compiled call using `vmap`, so twelve lanes cost far less than twelve times one lane.

This search is given a `name` and `path_prefix`, so its results are written to the `output` folder as tutorial 5
did, because there is a file in there we are about to read.
"""
search = af.MultiStartAdam(
    name="tutorial_6_gradients_adam",
    path_prefix="chapter_1_introduction",
    n_starts=12,  # The number of independent lanes which walk uphill from different starting points.
    n_steps=200,  # The maximum number of gradient steps each lane takes.
    learning_rate=0.5,  # How far each lane moves per step, in units of the adapted gradient.
    batch_size=None,  # None evaluates every lane in one compiled call; pass an integer to cap memory on large models.
)

print(
    """
    The non-linear search has begun running.
    Checkout the HowToFit/output/chapter_1_introduction/tutorial_6_gradients_adam
    folder for live output of the results.
    This Jupyter notebook cell with progress once the search has completed - this could take a few minutes!
    """
)

start = time.time()

result = search.fit(model=model, analysis=analysis_jax)

print("The search has finished run - you may now continue the notebook.")
print(f"MultiStartAdam run time: {time.time() - start} seconds")

print(result.info)

"""
Let us plot the maximum likelihood fit, which should look familiar from tutorial 3.
"""
model_data = result.max_log_likelihood_instance.model_data_from(xvalues=xvalues)

plt.errorbar(
    x=xvalues,
    y=data,
    yerr=noise_map,
    linestyle="",
    color="k",
    ecolor="k",
    elinewidth=1,
    capsize=2,
)
plt.plot(xvalues, model_data, color="r")
plt.title("MultiStartAdam model fit to 1D Gaussian dataset.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
Because this search wrote its results to hard disk, its output folder contains a file called `search.summary`: how
long the search took, how long one log likelihood evaluation took and, for gradient searches, diagnostics on how
often things went wrong. We read it back below.
"""
search_summary_path = path.join(str(search.paths.output_path), "search.summary")

if path.exists(search_summary_path):
    with open(search_summary_path) as f:
        print(f.read())

"""
The block at the bottom, headed `Resampling Info`, contains two entries only a gradient search can report:

- `Value-NaN Lane-Steps`: the number of times a lane stepped somewhere the log likelihood could not be computed at
  all, most often because it stepped outside the priors.

- `Gradient-NaN Lane-Steps`: the number of times the log likelihood *was* computable but its gradient was not. This
  is the sneakier of the two, because such a lane does not crash or die, it simply stops moving while continuing to
  look perfectly healthy.

Both counters are usually small and harmless, but they are the vocabulary you need to diagnose a gradient fit that
has gone quietly wrong. Tutorial 7 explains where they come from and what to do about them.

__Markov Chain Monte Carlo (MCMC)__

In tutorial 3 we used `Emcee`, whose walkers propose a step, compute the likelihood there and accept or reject the
move by comparing it to where they are. Notice what is missing: the walkers never ask which way is up. That is a
strength, because `Emcee` works on any likelihood you can evaluate, and a weakness, because in many dimensions a
random step is overwhelmingly likely to point somewhere worse, so most proposals are rejected.
"""
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

result = search.fit(model=model, analysis=analysis)

print("The search has finished run - you may now continue the notebook.")
print(f"Emcee run time: {time.time() - start} seconds")

print(result.info)

"""
Now the gradient-aware alternative, `BlackJAXNUTS`, which is Hamiltonian Monte Carlo. The physical picture behind it
is genuinely helpful.

Imagine the likelihood surface turned upside down, so its peak becomes a valley, and place a ball on the resulting
landscape. Give it a random flick and let it roll: it accelerates down slopes, coasts up the other side and travels
a long way while staying in regions the landscape favours. That trajectory is computed from the gradient at each
moment, which is what autodiff hands us for free. Where the ball stops becomes the next sample, and because it
travelled a long, informed distance rather than a small random hop, consecutive samples are far less similar.

"NUTS" stands for the No U-Turn Sampler, which solves the awkward choice here: how long to let the ball roll. Roll
too briefly and you wasted the gradient; roll too long and the ball curves back on itself. NUTS stops the trajectory
when it starts doubling back. Being a gradient method, it needs the JAX analysis.
"""
search = af.BlackJAXNUTS(
    num_warmup=200,  # Steps used to tune the sampler, which are then discarded.
    num_samples=300,  # Steps kept as samples of the posterior.
)

print(
    """
    The non-linear search has begun running.
    This Jupyter notebook cell with progress once the search has completed - this could take a few minutes!
    """
)

start = time.time()

result = search.fit(model=model, analysis=analysis_jax)

print("The search has finished run - you may now continue the notebook.")
print(f"BlackJAXNUTS run time: {time.time() - start} seconds")

print(result.info)

"""
Hamiltonian sampling comes with its own diagnostics, stored in the `samples_info` dictionary. Three are worth
knowing:

- `n_divergent`: the number of trajectories which "diverged", meaning the ball flew off to infinity instead of
  following the landscape. A handful is tolerable; many means the steps are too large and the samples cannot be
  trusted.

- `ess_min`: the "effective sample size" of the worst constrained parameter. Consecutive samples are correlated, so
  300 samples are worth fewer than 300 independent draws, and this says how many they are worth.

- `mean_acceptance`: the fraction of proposed trajectories accepted, which for NUTS should sit high, around the 0.8
  the warm up phase tunes towards. A low value means the sampler is struggling.
"""
samples = result.samples

print("Diagnostics of the Hamiltonian Monte Carlo fit:\n")
print(f"Number of divergent trajectories = {samples.samples_info.get('n_divergent')}")
print(f"Minimum effective sample size    = {samples.samples_info.get('ess_min')}")
print(
    f"Mean acceptance rate             = {samples.samples_info.get('mean_acceptance')}"
)

"""
Because NUTS maps out the posterior, we can plot the Probability Density Functions of its samples with `corner.py`,
wrapped via the `aplt.corner_cornerpy` function, exactly as we did for `Emcee` in tutorial 5.
"""
aplt.corner_cornerpy(samples=result.samples)

"""
__Nested Sampling__

Finally, nested sampling, the interesting case, because gradients do not help it at all.

Recall from tutorial 3 how it works: a set of "live points" is drawn from the priors, the lowest likelihood one is
discarded, and a replacement is drawn from the priors subject to having a higher likelihood. Repeat, and the live
points contract onto the peak. There is no walker in that description and therefore nothing to steer. The algorithm
never asks "which way is up from here", it asks "give me any point better than this one".

JAX still helps, for a different reason. `Nautilus` proposes points in batches, and with a JAX analysis
**PyAutoFit** evaluates a whole batch in one compiled call rather than looping in Python, which is the
`use_jax_vmap=True` input, on by default. Gradients change *how a search explores*; JAX compilation changes *how
fast the likelihood is evaluated*, and nested sampling benefits only from the second.
"""
search = af.Nautilus(
    n_live=150,  # The number of live points used to explore parameter space.
    n_eff=200,  # The effective sample size the search runs until it reaches.
)

print(
    """
    The non-linear search has begun running.
    This Jupyter notebook cell with progress once the search has completed - this could take a few minutes!
    """
)

start = time.time()

result = search.fit(model=model, analysis=analysis_jax)

print("The search has finished run - you may now continue the notebook.")
print(f"Nautilus run time: {time.time() - start} seconds")

print(result.info)

r"""
__Errors From Curvature__

In tutorial 5 we computed errors from the samples, by marginalizing over each parameter's 1D Probability Density
Function. That assumes nothing about the posterior's shape, but it requires the samples, which are thousands of
likelihood evaluations of work.

Gradients offer a shortcut. The first derivative gives the slope; the second gives the "curvature", how quickly the
slope itself changes. At the peak the slope is zero, so the curvature is all there is, and it says how sharply the
likelihood falls away as we step off the peak. A sharply curved peak means a small error, a gently curved one a
large error. The error is the curvature, read backwards.

For several parameters the second derivatives form a matrix called the "Hessian", which `jax.hessian` computes by
the same autodiff machinery as `jax.grad`. The errors follow from:

\[ C = -H^{-1}, \qquad \sigma_{i} = \sqrt{C_{ii}} \]

Where:

- \( H \): the Hessian, the matrix of second derivatives of the log likelihood at the peak.
- \( C \): the covariance matrix, the negative inverse of the Hessian.
- \( \sigma_{i} \): the 1 sigma error on parameter \( i \), the square root of the \( i \)th diagonal entry
  of \( C \).

In words: invert the curvature, flip its sign, and the square roots of its diagonal are the errors. We evaluate this
at the maximum likelihood vector found by `Nautilus`.
"""
samples = result.samples

max_log_likelihood_vector = samples.max_log_likelihood(as_instance=False)

hessian = jax.hessian(log_likelihood_from_vector)(
    jnp.array(max_log_likelihood_vector, dtype=float)
)

covariance = -jnp.linalg.inv(hessian)

errors_from_curvature = jnp.sqrt(jnp.diag(covariance))

print("Errors from the curvature of the likelihood (Laplace approximation):\n")
print(f"centre        = {errors_from_curvature[0]}")
print(f"normalization = {errors_from_curvature[1]}")
print(f"sigma         = {errors_from_curvature[2]}")

"""
Now the comparison. Below we print the errors the `Nautilus` samples give, with the `errors_at_upper_sigma` and
`errors_at_lower_sigma` methods of tutorial 5. Tutorial 5 used `sigma=3.0`, because a result quoted in a paper wants
a conservative interval; we use `sigma=1.0`, because that is what the curvature calculation produces.
"""
errors_at_upper_sigma = samples.errors_at_upper_sigma(sigma=1.0, as_instance=False)
errors_at_lower_sigma = samples.errors_at_lower_sigma(sigma=1.0, as_instance=False)

print("Errors from the samples (upper, at 1.0 sigma confidence):")
print(errors_at_upper_sigma)

print("\nErrors from the samples (lower, at 1.0 sigma confidence):")
print(errors_at_lower_sigma)

"""
The two sets of numbers agree closely, and that is not a coincidence, it is the "Laplace approximation" working:
near its peak a well behaved likelihood looks like a Gaussian, whose width the curvature sets. If the posterior
really is Gaussian, the curvature knows everything the samples know.

The catch is in the words "if" and "near". The curvature is measured at a single point, so it describes only the
peak it was measured on. If the posterior has a second peak the Hessian has no idea; if it is skewed, cut off by a
prior edge, or banana shaped because two parameters are degenerate, a symmetric Gaussian centred on the peak
misdescribes it, and the errors are wrong in ways the numbers do not reveal. Sampling assumes none of that.

One last check is visual rather than numerical. Instead of summarising the posterior at all, we can take a handful
of the models it contains and plot each over the data. Below we use the final five samples the search accepted,
which for nested sampling are its highest likelihood ones, turning each into an instance with
`sample.instance_for_model` as tutorial 5 did. If the draws all trace the data closely, and their spread is
comparable to the error bars, the fit is telling a consistent story. This is a "posterior predictive check", and it
catches what no single number can.
"""
plt.errorbar(
    x=xvalues,
    y=data,
    yerr=noise_map,
    linestyle="",
    color="k",
    ecolor="k",
    elinewidth=1,
    capsize=2,
)

for sample in samples.sample_list[-5:]:
    instance = sample.instance_for_model(model=samples.model)

    plt.plot(xvalues, instance.model_data_from(xvalues=xvalues), "--")

plt.title("Five models from the posterior, plotted over the data.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

r"""
__Wrap Up__

This tutorial took the one sentence tutorial 3 used to describe how an MLE search moves and unpacked it:

1. **Gradients**: the gradient of the log likelihood is a vector with one entry per free parameter, evaluated at a
point, pointing in the direction the likelihood increases fastest.

2. **Finite differencing versus autodiff**: a gradient can be estimated by nudging each parameter and re-evaluating,
but this costs \( 2N \) evaluations and forces a step size which is biased if too large and noisy if too small.
Autodiff applies the chain rule to the operations the likelihood performs, giving the exact gradient for a small fixed
multiple of the cost of one evaluation, whatever the number of parameters.

3. **MLE**: `LBFGS` finite differences its own gradient from a single starting point, which is expensive and
fragile. `MultiStartAdam` follows exact gradients from many starting points at once, evaluated together in one
compiled call, making it far harder to trap in a local maximum.

4. **MCMC**: `Emcee` proposes random steps and accepts or rejects them, never asking which way is up.
`BlackJAXNUTS` rolls a ball across the landscape using the gradient to shape its trajectory, producing far less
correlated samples for the same number of steps.

5. **Nested sampling**: `Nautilus` replaces low likelihood live points with higher likelihood ones drawn from the
priors, a procedure with no direction of travel for a gradient to inform. JAX speeds it up by evaluating batches of
points in one compiled call, but the algorithm is unchanged.

6. **Curvature errors**: the second derivatives at the peak give a covariance matrix and hence 1 sigma errors on
every parameter, essentially for free. They agree with the errors from samples whenever the peak is single and
Gaussian shaped, and quietly mislead when it is not.

Gradients are therefore not a different way of fitting models, they are extra information about the same likelihood
surface we have explored since tutorial 3, and each family of search makes its own use of it, or none at all.

Whether a gradient fit succeeds depends on details we have skated over: what happens when a search steps outside its
priors, when two parameters are degenerate, or when the likelihood is defined somewhere its gradient is not.
Tutorial 7 looks at the details that decide whether these searches succeed.
"""
