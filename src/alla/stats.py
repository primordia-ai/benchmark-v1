import functools
from typing import Union
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import root_scalar
from alla.logger import logger


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
        items = [stats.norm(dist[0], dist[1]) if isinstance(dist, tuple) else dist for dist in distributions]

        if abs(1 - sum(weights)) > 1e-4:
            raise ValueError(f"Mixture weight sum exceeds 1e-4 padding, sum = {sum(weights)}")

        if abs(1 - sum(weights)) > 1e-6:
            # Normalize weights if needed
            logger.warning(f"Renormalizing Mixture weights, sum = {sum(weights)}")
            weights /= np.sum(weights)

        mixture = cls(items, weights)

        return mixture.ppf(quantiles)


def log_probability_score(predicted_probs : list, true_class : int) -> float:
    """
    Calculate the log probability score for the true class.

    Parameters:
    predicted_probs (list): List of predicted probabilities for each class.
    true_class (int): The index of the true class.

    Returns:
    float: The log probability score for the true class.
    """
    return np.log(predicted_probs[true_class])

def fit_distribution_to_quantiles(quantiles : list, true_growth : float) -> float:
    """
    Fits a Gaussian distribution to the given quantiles and returns the log probability density 
    of the true growth value under the fitted distribution.
    Parameters:
    quantiles (list): A list of quantile values to fit the Gaussian distribution to.
    true_growth (float): The true growth value for which the log probability density is calculated.
    Returns:
    float: The log probability density of the true growth value under the fitted Gaussian distribution.
    """

    mean = np.mean(quantiles)
    std = np.std(quantiles)
    distribution = stats.norm(loc=mean, scale=std)
    return distribution.logpdf(true_growth)

def quantile_loss(alpha: float, true: float, pred: float) -> float:
    """
    Calculate the quantile loss for a given prediction.
    The quantile loss is used to measure the accuracy of quantile forecasts.
    It is asymmetric and assigns different penalties to overestimation and underestimation.
    Parameters:
    alpha (float): The quantile to be estimated, a value between 0 and 1.
    true (float): The true observed value.
    pred (float): The predicted value.
    Returns:
    float: The quantile loss.
    """

    return alpha * (true - pred) if true > pred else (1 - alpha) * (pred - true)


def non_parametric_quantile_loss(predicted_quantiles : list, true_growth : float, alphas : list) -> float:
    """
    Calculate the non-parametric quantile loss for a set of predicted quantiles.

    Parameters:
    predicted_quantiles (list of float): The predicted quantile values.
    true_growth (float): The true growth value.
    alphas (list of float): The quantile levels corresponding to the predicted quantiles.

    Returns:
    float: The total quantile loss.
    """
    loss = 0
    for predicted_gq, alpha in zip(predicted_quantiles, alphas):
        loss += quantile_loss(predicted_gq, true_growth, alpha)
    return loss

def evaluate_performance(predictions : dict, ground_truth : dict) -> pd.DataFrame:
    """
    Evaluate the performance of predictions against ground truth data.
    Parameters:
    predictions (dict): A dictionary where each key is an asset and the value is another dictionary containing:
        - 'class_probs' (list of float): The predicted probabilities for each class.
        - 'growth_quantiles' (list of float): The predicted growth quantiles.
        - 'alphas' (list of float): The alpha values for non-parametric quantile loss calculation.
    ground_truth (dict): A dictionary where each key is an asset and the value is another dictionary containing:
        - 'true_class' (int): The true class label.
        - 'true_growth' (float): The true growth value.
    Returns:
    pd.DataFrame: A DataFrame containing the evaluation scores for each asset with columns:
        - 'asset' (str): The asset identifier.
        - 'log_class_score' (float): The log probability score for the class prediction.
        - 'class_score' (float): The exponentiated log class score.
        - 'parametric_score' (float): The score from fitting a distribution to the growth quantiles.
        - 'non_parametric_score' (float): The non-parametric quantile loss score.
    """
    scores = []
    for asset in predictions:
        predicted_probs = predictions[asset]['class_probs']
        true_class = ground_truth[asset]['true_class']
        log_class_score = log_probability_score(predicted_probs, true_class)
        class_score = np.exp(log_class_score)

        quantiles = predictions[asset]['growth_quantiles']
        true_growth = ground_truth[asset]['true_growth']
        parametric_score = fit_distribution_to_quantiles(quantiles, true_growth)

        alphas = predictions[asset]['alphas']
        non_parametric_score = non_parametric_quantile_loss(quantiles, true_growth, alphas)

        scores.append({
            'asset': asset,
            'log_class_score': log_class_score,
            'class_score': class_score,
            'parametric_score': parametric_score,
            'non_parametric_score': non_parametric_score
        })
    return pd.DataFrame(scores)
