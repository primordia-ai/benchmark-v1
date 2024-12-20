import functools
from typing import Union

import numpy as np
from scipy import stats
from scipy.optimize import root_scalar


def mixture_entropy(weights, means, stds):
    """Compute entropy of a Gaussian mixture."""
    x = np.linspace(
        min(means - 3 * max(stds)),
        max(means + 3 * max(stds)),
        1000
    )

    pdf = np.zeros_like(x)
    for k in range(len(weights)):
        pdf += weights[k] * stats.norm.pdf(x, means[k], stds[k])

    entropy = -np.sum(pdf[pdf > 0] * np.log(pdf[pdf > 0])) * (x[1] - x[0])
    return entropy


def mixture_stats(weights, means, stds) -> tuple[np.float64, np.float64]:
    """Compute mean and std of a Gaussian mixture."""
    mean = np.sum(weights * means)
    var = np.sum(weights * (stds ** 2 + means ** 2)) - mean ** 2
    return mean, np.sqrt(var)


def _vectorize_float(f):
    vectorized = np.vectorize(f, otypes=[float], signature="(),()->()")

    @functools.wraps(f)
    def wrapper(*args):
        return vectorized(*args)

    return wrapper


class MixtureDistribution:
    """
    Mixture Distribution class

    Provides pdf, cdf, and ppf methods

    Credit to James Atkins:
    https://www.jamesatkins.com/posts/quantile-function-of-mixture-distributions-in-python/
    """

    def __init__(self, distributions, weights):
        self._distributions = list(distributions)
        self._weights = list(weights)

        if not (all(w >= 0 for w in self._weights) and abs(1 - sum(self._weights)) < 1e-6):
            raise ValueError("Invalid weight vector.")

        if len(self._distributions) != len(self._weights):
            raise ValueError("Mixtures and weights must have the same length.")

        if len(self._distributions) < 2:
            raise ValueError("Must have at least two component distributions.")

    @_vectorize_float
    def pdf(self, x):
        return sum(w * d.pdf(x) for w, d in zip(self._weights, self._distributions))

    @_vectorize_float
    def cdf(self, x):
        return sum(w * d.cdf(x) for w, d in zip(self._weights, self._distributions))

    @_vectorize_float
    def ppf(self, p):
        bracket = [min(dist.ppf(p) for dist in self._distributions),
            max(dist.ppf(p) for dist in self._distributions)]

        r = root_scalar(
            f=lambda x: self.cdf(x) - p,
            fprime=self.pdf,
            bracket=bracket,
            x0=0
        )
        assert r.converged
        return r.root

    @classmethod
    def compute_quantiles(
        cls,
        distributions: Union[list[stats.norm], list[tuple[float, float]]],
        weights,
        quantiles
    ) -> np.ndarray:
        """
        Compute quantiles of a mixture distribution

        Args:
            distributions:  List of distributions or list of tuples of means and stds
            weights:        List of mixture weights
            quantiles:      List of quantiles

        Returns:
            List of quantiles
        """
        # distributions = [stats.norm(loc=mean, scale=std) for mean, std in zip(means, stds)]
        items = [stats.norm(dist[0], dist[1]) if isinstance(dist, tuple) else dist for dist in distributions]
        # Filter items with zero weight
        filtered_items, filtered_weights = [], []
        for item, w in zip(items, weights):
            if w > 1e-6:
                filtered_items.append(item)
                filtered_weights.append(w)

        mixture = cls(filtered_items, filtered_weights)
        return mixture.ppf(quantiles)
