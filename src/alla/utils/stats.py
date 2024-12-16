import numpy as np
from scipy import stats


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


def synthetic_indicator_noise(likelihoods: np.ndarray) -> np.ndarray:
    """
    Simulate soft-indicator noise incurred by the soft-indicator extractor, using the color confusion matrix.

    :param likelihoods:   [..., 3] array of indicator values
    :return:              [..., 3] array with applied noise
    """
    color_confusion = np.array(
        [
            [0.8, 0.15, 0.05],
            [0.1, 0.8, 0.1],
            [0.05, 0.15, 0.8],
        ]
    )
    return np.dot(likelihoods, color_confusion)
