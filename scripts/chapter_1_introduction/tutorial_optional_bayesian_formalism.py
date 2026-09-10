r"""
Tutorial Optional: Bayesian Formalism
=====================================

Every tutorial in this chapter has been an exercise in Bayesian inference, and not one of them wrote down a single
equation of probability theory. That was deliberate. The **HowToFit** lectures are built on the conviction that you
can learn to fit models to data properly, and interpret the results correctly, without first sitting through a
course on Bayesian statistics.

This optional tutorial writes those equations down anyway.

Nothing in it will make your fits better. What it does is connect the phenomenological picture built up over
tutorials 1 to 7 -- guessing models, likelihood surfaces, walkers, live points, gradients -- to the handful of
symbols that papers and documentation use for exactly the same things. When a referee asks whether you quoted the
maximum likelihood model or the median of the marginalized posterior, this tutorial tells you they are talking
about quantities you have already computed.

My suggestion is to read it once, run it once, and then forget most of it.

__Contents__

This tutorial is split into the following sections:

- **Bayes Theorem**: The equation that every tutorial so far has been solving without writing it down.
- **Data**: Load the 1D Gaussian dataset that every equation in this tutorial is evaluated on.
- **The Model**: The parameters of tutorial 1 as the thing Bayes' theorem is about.
- **The Likelihood**: Why the chi-squared and noise normalization of tutorial 2 are the log of a Gaussian likelihood.
- **The Prior**: The priors of tutorial 3 as a probability distribution, and the unit cube as its inverse.
- **The Posterior**: The samples of tutorial 5 as draws from the posterior, and a numerical check that log prior plus log likelihood is the log posterior.
- **Maximum Likelihood Vs Maximum A Posteriori**: Two different "best-fit" answers and when they differ.
- **Marginalization**: The one-dimensional PDFs and errors of tutorial 5 as integrals over the other parameters.
- **The Evidence**: The Bayesian evidence of tutorial 5 as the normalization of Bayes' theorem, and why nested sampling computes it.
- **Gradients**: The gradient of tutorial 6 as the derivative of the log posterior.
- **Wrap Up**: Why HowToFit hides the maths, and why it helps to know it anyway.

__Bayes Theorem__

Here is the equation that every tutorial in this chapter has been solving without ever writing it down:

\[ P(\theta \mid D) = \frac{P(D \mid \theta)\,P(\theta)}{P(D)} \]

Where:

- \( \theta \): the model parameters, for our 1D Gaussian the three numbers `centre`, `normalization` and `sigma`.

- \( D \): the data, for us the 1D array of 100 values loaded from `data.json` together with its noise-map.

- \( P(\theta \mid D) \): the "posterior", the probability of the parameters given the data, and the quantity we
  actually want.

- \( P(D \mid \theta) \): the "likelihood", the probability of the data given the parameters, computed by the
  `log_likelihood_function` of tutorial 3 before we take its logarithm.

- \( P(\theta) \): the "prior", the probability we assign to parameter values before looking at the data, the
  `UniformPrior` and `LogUniformPrior` objects of tutorial 3.

- \( P(D) \): the "evidence", a single number which normalizes the right hand side, the `log_evidence` of tutorial 5.

In plain words: the answer we want is the goodness-of-fit multiplied by our prior beliefs, divided by a normalizing
number. Every algorithm in this chapter is a strategy for evaluating the right hand side without ever computing the
integral hiding inside \( P(D) \).

Each tutorial supplied exactly one piece. Tutorial 1 built \( \theta \) and the forward model. Tutorial 2 built
\( P(D \mid \theta) \), calling it the log likelihood. Tutorial 3 built \( P(\theta) \) and the searches which
explore the product of the two. Tutorial 4 showed what goes wrong when the posterior has more than one peak,
tutorial 5 read the answer back out, and tutorials 6 and 7 differentiated it and asked what the numbers mean.

So when this tutorial says "the posterior", you have already computed one. We are only relabelling.
"""

# from autofit import setup_notebook; setup_notebook()

from os import path
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import autofit as af

"""
__Data__

We use the same `gaussian_x1` dataset that tutorials 2 and 3 fitted, so every equation below is evaluated on a fit
you have already seen.
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

r"""
__The Model__

The symbol \( \theta \) is nothing more exotic than the list of numbers you passed to `instance_from_vector` in
tutorial 1. Bayes' theorem is a statement about a probability distribution over that list, and nothing else.

We therefore begin as tutorial 1 did, by re-pasting the `Gaussian` class in full. Its constructor arguments are the
entries of \( \theta \), and its `model_data_from` method is the forward model: the function which, given
\( \theta \), predicts what the data should look like.
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

    def model_data_from(self, xvalues: np.ndarray) -> np.ndarray:
        """
        Returns a 1D Gaussian on an input list of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Gaussian, via its `centre`.

        The output is referred to as the `model_data` to signify that it is a representation of the data from the
        model.

        Parameters
        ----------
        xvalues
            The x coordinates in the original reference frame of the data.

        Returns
        -------
        np.array
            The Gaussian values at the input x coordinates.
        """
        transformed_xvalues = np.subtract(xvalues, self.centre)
        return np.multiply(
            np.divide(self.normalization, self.sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(transformed_xvalues, self.sigma))),
        )


r"""
Composing the model gives the two pieces of book-keeping which make \( \theta \) concrete. The `prior_count` is the
number of dimensions the posterior is a distribution over, and `paths` gives the order of the entries in the vector.
"""
model = af.Model(Gaussian)

print("Number of parameters in theta:", model.prior_count)
print("Order of the parameters in theta:", model.paths)

r"""
A specific \( \theta \) becomes a model, and the model becomes predicted data, in the two lines below. This is the
forward modelling of tutorial 1, and it is what lets us ask "how probable is the data, given these parameters?".
"""
instance = model.instance_from_vector(vector=[50.0, 25.0, 10.0])

model_data = instance.model_data_from(xvalues=xvalues)

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
plt.title("The data D, with the forward model prediction for one theta.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

r"""
__The Likelihood__

In tutorial 2 we built a chain of quantities -- residuals, a chi-squared and a noise normalization -- and combined
the last two into a log likelihood by multiplying their sum by -0.5. At the time I said the reason for the -0.5 was
not critical. Here it is.

We assume each data point is its predicted value plus a random draw from a Gaussian of width equal to that point's
noise. The probability of the whole dataset is then a product of one Gaussian probability per data point:

\[ P(D \mid \theta) = \prod_i \frac{1}{\sqrt{2\pi\sigma_i^2}} \exp\left(-\frac{(d_i - m_i)^2}{2\sigma_i^2}\right) \]

Taking the logarithm turns that product into a sum:

\[ \ln P(D \mid \theta) = -\frac{1}{2} \sum_i \frac{(d_i - m_i)^2}{\sigma_i^2} - \frac{1}{2} \sum_i \ln(2\pi\sigma_i^2) \]

Where:

- \( d_i \): the value of the data at point \( i \), the `data` array.

- \( m_i \): the value of the model data at point \( i \), computed by `model_data_from` for this \( \theta \).

- \( \sigma_i \): the noise at point \( i \), the `noise_map` array.

Compare the second equation with tutorial 2 line by line. The first sum is the `chi_squared`, and the second is the
`noise_normalization`, the log of the product of the Gaussian prefactors. Both are multiplied by -0.5, which is
where the mysterious factor comes from: it is not a convention, it is the 2 in the denominator of the exponent of a
Gaussian. The noise normalization really is a constant, because it depends only on the noise-map and never on
\( \theta \), so it shifts every log likelihood equally and can never change which model wins.

Below we compute the log likelihood twice: by tutorial 2's chi-squared route, and by summing the log of a Gaussian
probability density at every data point using `scipy`. They are the same number.
"""
residual_map = data - model_data
chi_squared = np.sum((residual_map / noise_map) ** 2.0)
noise_normalization = np.sum(np.log(2 * np.pi * noise_map**2.0))

log_likelihood = -0.5 * (chi_squared + noise_normalization)

log_likelihood_via_pdf = np.sum(
    scipy.stats.norm.logpdf(data, loc=model_data, scale=noise_map)
)

print("Log likelihood via tutorial 2's chi-squared = ", log_likelihood)
print("Log likelihood via a sum of Gaussian PDFs   = ", log_likelihood_via_pdf)

r"""
__The Prior__

The priors of tutorial 3 were introduced as a way of telling the search where to look. Formally they are
\( P(\theta) \), a probability distribution over the parameters written down before we look at the data. Because our
parameters are independent, the prior is a product of one distribution per parameter and its logarithm is a sum,
\( \ln P(\theta) = \sum_j \ln P(\theta_j) \).

The two priors on the default `Gaussian` model have simple forms. A `UniformPrior` is constant inside its limits and
zero outside them, so its log is a constant we may set to zero inside and negative infinity outside. A
`LogUniformPrior` is proportional to \( 1 / \theta_j \), the statement "every order of magnitude is equally likely",
so its log falls off as \( -\ln \theta_j \).

**PyAutoFit** hands you those numbers directly. Below we print the priors of the model, then evaluate the log prior
of each parameter inside the prior limits, and then for a `centre` of 150.0 which lies outside the `UniformPrior`
limits of 0.0 to 100.0.
"""
print(model.info)

print("Log prior of each parameter, theta = [50.0, 25.0, 10.0]:")
print(model.log_prior_list_from_vector(vector=[50.0, 25.0, 10.0]))

print("Log prior of each parameter, theta = [150.0, 25.0, 10.0]:")
print(model.log_prior_list_from_vector(vector=[150.0, 25.0, 10.0]))

r"""
The first entry of the second list is `-inf`, an infinitely improbable model. This is the formal version of tutorial
3's statement that priors define the valid parameter space: a model outside the priors is not penalised, it is
excluded. The middle entry is not zero, because the `LogUniformPrior` on `normalization` prefers smaller values and
\( -\ln(25) \) is about -3.2.

Priors enter a fit a second way, and it is how nested sampling works. Every prior can be inverted: instead of asking
"how probable is this value?", we ask "which value sits at this fraction of the distribution?". That inverse is the
inverse cumulative distribution function, and in **PyAutoFit** it is the `value_for` method, which maps a number
between 0 and 1 -- the "unit cube" -- onto a physical parameter value. Feeding in 0.5 asks each prior for its
median, and the lines below show why a unit value is not the same thing as a parameter value.
"""
uniform_prior = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
log_uniform_prior = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
gaussian_prior = af.GaussianPrior(mean=10.0, sigma=5.0)

print("UniformPrior(0.0, 100.0).value_for(0.5) = ", uniform_prior.value_for(0.5))
print(
    "LogUniformPrior(0.01, 100.0).value_for(0.5) = ", log_uniform_prior.value_for(0.5)
)
print(
    "GaussianPrior(mean=10, sigma=5).value_for(0.5) = ", gaussian_prior.value_for(0.5)
)

print("The whole model, unit cube [0.5, 0.5, 0.5] mapped to physical values:")
print(model.vector_from_unit_vector(unit_vector=[0.5, 0.5, 0.5]))

r"""
This is the key to a sentence in tutorial 3 which may have seemed arbitrary: nested sampling "draws live points from
the priors". A search which picks unit values uniformly at random between 0 and 1 and passes them through
`value_for` is drawing samples from the prior distribution itself. It is also why a nested sampling search cannot be
given a starting point: its first step is defined to be a draw from \( P(\theta) \).

__The Posterior__

We now have both pieces of the numerator of Bayes' theorem, so we can fit the model and look at the left hand side.
The `Analysis` class below is the one from tutorial 3, whose `log_likelihood_function` returns exactly the equation
verified above, and we fit it with the nested sampling search of tutorial 3.
"""


class Analysis(af.Analysis):
    def __init__(self, data: np.ndarray, noise_map: np.ndarray):
        """
        The `Analysis` class acts as an interface between the data and model in **PyAutoFit**, exactly as it did in
        tutorial 3.

        Parameters
        ----------
        data
            A 1D numpy array containing the data (e.g. a noisy 1D signal) fitted in the workspace examples.
        noise_map
            A 1D numpy array containing the noise values of the data.
        """
        super().__init__()

        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance) -> float:
        """
        Returns the log likelihood of a fit of a 1D Gaussian to the dataset, which is the log of a product of
        Gaussian probabilities, one per data point, as derived in the `The Likelihood` section above.
        """
        xvalues = np.arange(self.data.shape[0])

        model_data = instance.model_data_from(xvalues=xvalues)
        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        chi_squared = np.sum(chi_squared_map)
        noise_normalization = np.sum(np.log(2 * np.pi * self.noise_map**2.0))

        return -0.5 * (chi_squared + noise_normalization)


analysis = Analysis(data=data, noise_map=noise_map)

search = af.DynestyStatic(
    sample="rwalk",  # This makes dynesty run faster, dont worry about what it means for now!
)

print(
    """
    The non-linear search has begun running.
    This Jupyter notebook cell with progress once the search has completed - this could take a few minutes!
    """
)

model = af.Model(Gaussian)

result = search.fit(model=model, analysis=analysis)

print("The search has finished run - you may now continue the notebook.")

r"""
The `Samples` object of tutorial 5 is the posterior. Each entry of `parameter_lists` is a \( \theta \) the search
accepted, and each entry of `weight_list` says how much that sample counts when the posterior is summed up. Together
they approximate \( P(\theta \mid D) \): a cloud of points, denser where the posterior is higher.

Tutorial 5 also told you, without justification, that the log posterior is the log likelihood plus the log prior.
That is Bayes' theorem logged, with the evidence dropped because it is the same number for every sample:

\[ \ln P(\theta \mid D) = \ln P(D \mid \theta) + \ln P(\theta) - \ln P(D) \]

Below we check the first few samples, printing the log likelihood, log prior, log posterior, and the difference
between the log posterior and the sum of the first two.
"""
samples = result.samples

total_printed = min(5, len(samples.log_likelihood_list))

print("Sample : log likelihood : log prior : log posterior : difference")

for sample_index in range(total_printed):
    log_likelihood = samples.log_likelihood_list[sample_index]
    log_prior = samples.log_prior_list[sample_index]
    log_posterior = samples.log_posterior_list[sample_index]

    print(
        sample_index,
        log_likelihood,
        log_prior,
        log_posterior,
        log_posterior - (log_likelihood + log_prior),
    )

r"""
The final column is zero. There is no approximation in that identity, it is the definition of the log posterior, and
it means every number tutorial 5 printed was already a Bayesian quantity. Note also how much larger the log
likelihoods are than the log priors: for this fit the data is far more informative than our prior beliefs.

__Maximum Likelihood Vs Maximum A Posteriori__

There are two "best-fit" models hiding in a set of samples, and tutorial 5 printed both without saying they could
disagree:

- The "maximum likelihood" model, `samples.max_log_likelihood()`, is the \( \theta \) fitting the data best,
  ignoring the priors entirely.

- The "maximum a posteriori" model, usually shortened to MAP, `samples.max_log_posterior()`, is the \( \theta \)
  which maximizes likelihood times prior.

If every prior is uniform the log prior is the same constant inside the limits, so the two are the same model. They
separate only when a prior varies appreciably across the region the data allows.
"""
print("Maximum likelihood theta   = ", samples.max_log_likelihood(as_instance=False))
print("Maximum a posteriori theta = ", samples.max_log_posterior(as_instance=False))

r"""
For this fit they agree, or very nearly, because the log-uniform prior on `normalization` is almost flat over the
tiny range of normalizations the data permits. Had we instead placed a tight Gaussian prior on `centre`, say a mean
of 49.0 with a sigma of 0.05, the log prior would fall away steeply across the region the data allows and the MAP
model would be pulled towards 49.0, away from what the data alone prefers.

Neither answer is wrong, they answer different questions, and a paper quoting one while calling it the other is
making a real mistake. It is also why most results are quoted as neither, but as the median of the marginalized
posterior.

__Marginalization__

Tutorial 5 quoted a value and an error for each parameter individually, and plotted their 1D histograms via
`aplt.corner_cornerpy`. That requires an operation with a formal name, "marginalization", which means integrating
the posterior over the parameters you are not currently interested in:

\[ P(\theta_1 \mid D) = \int P(\theta \mid D) \, d\theta_2 \, d\theta_3 \ldots \]

Where:

- \( \theta_1 \): the parameter we want a one-dimensional answer for, for example `centre`.

- \( \theta_2, \theta_3, \ldots \): every other parameter of the model, integrated away.

That integral looks intimidating, and done analytically it would be. This is the greatest practical advantage of
having samples: marginalizing them means ignoring the columns you do not care about and histogramming the one you
do, weighted by `weight_list`. The integral becomes a counting exercise, and the cell below marginalizes over
`normalization` and `sigma` by never mentioning them.
"""
centre_list = [parameters[0] for parameters in samples.parameter_lists]

plt.hist(centre_list, bins=30, weights=samples.weight_list, color="k")
plt.title("Marginalized 1D posterior of the Gaussian's centre.")
plt.xlabel("centre of the Gaussian")
plt.ylabel("Posterior probability")
plt.show()
plt.close()

"""
Every summary statistic of tutorial 5 is read off that curve. The `median_pdf` value is the point with half the
weight either side of it, and the values at a given sigma are the points enclosing the corresponding fraction of it,
68 per cent at one sigma.
"""
print("Median of the marginalized posterior:")
print(samples.median_pdf(as_instance=False))

print("Values at 1.0 sigma confidence (lower, upper) for each parameter:")
print(samples.values_at_sigma(sigma=1.0, as_instance=False))

r"""
__The Evidence__

The one term of Bayes' theorem we have ignored is the denominator, \( P(D) \), the evidence. It is the integral of
the numerator over the whole of parameter space:

\[ P(D) = \int P(D \mid \theta) \, P(\theta) \, d\theta \]

Where:

- The integral runs over every value of every parameter the priors allow.

Two facts about this integral explain a lot of what you have seen. First, it does not depend on \( \theta \), which
is why MCMC never needs it: a walker deciding whether to move compares posteriors as a ratio and the evidence
cancels. Second, it is genuinely hard, an integral over a many-dimensional space in which almost all of the volume
contributes almost nothing.

Nested sampling gets it almost for free, and this is the real reason the algorithm is built the way it is. Tutorial
3 described it as replacing the lowest-likelihood live point over and over, and each replacement shrinks the prior
volume still under consideration by a known factor, so summing likelihood times shrinking volume as it goes is the
integral above. The parameter estimates are the by-product, the evidence is what the method actually computes, and
because it penalises models which spread their prior probability over regions the data rejects, comparing the
evidence of two models is the Bayesian way to ask whether extra parameters were worth adding.
"""
log_evidence = samples.log_evidence

if log_evidence is not None:
    print("Log evidence of this model = ", log_evidence)
else:
    print("This search does not compute a Bayesian evidence, so log_evidence is None.")

r"""
__Gradients__

Tutorial 6 introduced the gradient: the direction in which the log likelihood increases fastest, computed exactly by
automatic differentiation. In this notation, what the gradient searches climb is the derivative of the log
posterior:

\[ \nabla_\theta \ln P(\theta \mid D) = \nabla_\theta \ln P(D \mid \theta) + \nabla_\theta \ln P(\theta) \]

Where:

- \( \nabla_\theta \): the vector of derivatives with respect to each parameter in turn.

The evidence has vanished, and its disappearance is the point. Because \( \ln P(D) \) does not depend on
\( \theta \), differentiating it gives zero: the hardest term in Bayes' theorem is invisible to any method which
only ever looks at slopes.

This is the formal reason gradient methods scale so well. Hamiltonian Monte Carlo and its self-tuning variant NUTS,
which tutorial 6 introduced, use this gradient to propose moves following the contours of the posterior rather than
stumbling across them at random. In many dimensions, knowing which way is uphill is worth an enormous number of the
blind guesses tutorial 4 showed we could not afford to make.

There is no code here, because the numbers were computed in tutorial 6. All that is new is the name of the quantity
being differentiated.

__Wrap Up__

The **HowToFit** lectures teach model-fitting without formal statistics on purpose. Most scientists who fit models
to data do not need the equations to do it well: they need to know what a log likelihood measures, why priors
matter, which search suits their problem and how to tell a good fit from a bad one. Every one of those lessons was
taught in this chapter without an integral in sight.

Knowing the formalism afterwards is still worth an hour of your time. It tells you that the maximum likelihood model
and the median of the marginalized posterior answer different questions, and what the log evidence in `result.info`
is an integral of. And it lets you read a methods section, where these quantities go by their symbols rather than
their attribute names.

So, with your own problem in mind:

- Which of the two "best-fit" answers above is the one your field quotes, and is it the one you have been quoting?

- How informative are your priors compared with your data, and would your results change if someone else chose them?

- Is there a question in your science which is really a model comparison, and therefore really about the evidence?

If you are unsure, that is fine, and it is no obstacle to fitting models. Go back to tutorial 5, run a fit and look
at the numbers again. They will mean a little more than they did before.
"""
