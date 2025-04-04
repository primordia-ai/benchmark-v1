import functools
from typing import TypedDict, Union

import numpy as np
import pandas as pd
import pymc
from scipy import stats
from scipy.optimize import root_scalar

from cristal.logger import logger


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

        if not abs(1 - sum(self._weights)) < 1e-6:
            raise ValueError("Invalid weight vector sum.")

        if len(self._distributions) != len(self._weights):
            raise ValueError("Mixtures and weights must have the same length.")

        if sum(w > 0 for w in self._weights) == 1:
            # Singular distribution
            selected_d, selected_w = None, None
            for w, d in zip(self._weights, self._distributions):
                if w > 0:
                    selected_w, selected_d = w, d
                    break
            self._distributions = [selected_d]
            self._weights = [selected_w]

    @_vectorize_float
    def pdf(self, x):
        return sum(w * d.pdf(x) for w, d in zip(self._weights, self._distributions))

    @_vectorize_float
    def cdf(self, x):
        return sum(w * d.cdf(x) for w, d in zip(self._weights, self._distributions))

    @_vectorize_float
    def ppf(self, p):
        if len(self._weights) == 1:
            # If only one active component, return its ppf directly
            return self._distributions[0].ppf(p)

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


def quantile_loss(alpha: float, true: float, pred: float) -> float:
    """
    Calculate the quantile loss for a given prediction.
    The quantile loss is used to measure the accuracy of quantile forecasts.
    It is asymmetric and assigns different penalties to overestimation and underestimation.

    Args:
        alpha (float): The quantile to be estimated, a value between 0 and 1.
        true (float): The true observed value.
        pred (float): The predicted value.

    Returns:
        float: The quantile loss.
    """

    return alpha * (true - pred) if true > pred else (1 - alpha) * (pred - true)


def non_parametric_quantile_loss(predicted_quantiles: list, true_growth: float, alphas: list) -> float:
    """
    Calculate the non-parametric quantile loss for a set of predicted quantiles.

    Args:
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


class PredictionResult(TypedDict):
    growth: tuple[float, float]  # ( (<quantile>, <growth_value>), ... )
    category: dict[str, float]  # {<category>: <probability>, ...}
    truth: dict[str, float | str]  # {"category": <str>, "growth": <float>}


    
def make_classification_eval(predictions: dict[str, PredictionResult], output_file: str):
    """
    Save the confusion matrix and performance metrics to a file.
    Args:
        predictions (dict): A dictionary where each key is an asset and the value is a prediction PredictionResult
        output_file (str): The file path to save the confusion matrix and metrics.
    """
    true_labels = []
    pred_labels = []
    for asset, data in predictions.items():
        if "error" in data:
            continue
        true_labels.append(data['truth']['category'])
        pred_labels.append(max(data['category'], key=data['category'].get))
    print(true_labels)
    print(pred_labels)
    cm = confusion_matrix(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    mcc = matthews_corrcoef(true_labels, pred_labels)

    plt.figure(figsize=(3, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                 xticklabels=list(set(true_labels)),
                 yticklabels=list(set(true_labels)))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(output_file)
    # Write metrics to a CSV file
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC'],
        'Value': [accuracy, precision, recall, f1, mcc]
    })
    metrics_df.to_csv(output_file.replace('.png', '.csv'), index=False)

    # Write metrics to a LaTeX table
    with open(output_file.replace('.png', '.tex'), 'w') as f:
        metrics_df['Value'] = metrics_df['Value'].round(4)
        f.write(metrics_df.to_latex(index=False))

    logger.info(f"Confusion matrix and metrics saved to {output_file}, {output_file.replace('.png', '.csv')}, and {output_file.replace('.png', '.tex')}.")
    logger.info(f"Confusion matrix and metrics saved to {output_file}.")

def evaluate_performance(
    predictions: dict[str, PredictionResult], mixture_params: dict[str, dict], in_place: bool = False
) -> pd.DataFrame | dict:
    """
    Evaluate the performance of predictions against ground truth data.
    Args:
        predictions (dict): A dictionary where each key is an asset and the value is a prediction PredictionResult
        mixture_params (dict): Dictionary as {"mu": {<category>: <mean>, ...}, "sigma": {<category>: <std>, ...}}
        in_place (bool): If True, the evaluation scores are added to the predictions dictionary.

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation scores for each asset with columns:
            - 'asset' (str): The asset identifier.
            - 'log_class_score' (float): The log probability score for the class prediction.
            - 'class_score' (float): The exponentiated log class score.
            - 'parametric_score' (float): The score from fitting a distribution to the growth quantiles.
            - 'non_parametric_score' (float): The non-parametric quantile loss score.
    """
    # We have to handle error'd assets carefully.
    valid_assets = []
    atleast_one = False
    for asset_id, data in predictions.items():
        if "error" in data and data['error']:
            logger.warning(f"Excluding asset {asset_id} from evaluation due to error.")
            continue
        atleast_one = True
        valid_assets.append(asset_id)

    if not atleast_one:
        logger.warning("No valid assets to evaluate.")
        return pd.DataFrame() if not in_place else predictions

    n_asset = len(valid_assets)
    classes = list(mixture_params['mu'].keys())
    n_class = len(classes)

    weights = np.zeros((n_asset, n_class))
    mus = np.zeros((n_asset, n_class))
    sigmas = np.zeros((n_asset, n_class))
    true_growths = np.zeros(n_asset, dtype=float)

    for i, asset in enumerate(valid_assets):
        data = predictions[asset]
        true_growths[i] = data['truth']['growth']
        for j, cat in enumerate(classes):
            weights[i, j] = data['category'][cat]
            mus[i, j] = mixture_params['mu'][cat]
            sigmas[i, j] = mixture_params['sigma'][cat]

    with pymc.Model() as model:
        dist = pymc.NormalMixture.dist(w=weights, mu=mus, sigma=sigmas)
        logp = pymc.logp(dist, true_growths)

    parametric_scores = logp.eval()

    scores = []
    for i, asset in enumerate(valid_assets):
        data = predictions[asset]
        true_class = data['truth']['category']
        class_pred = data['category'][true_class]
        if class_pred < 1e-8:
            class_loss = -1e8
        else:
            class_loss = -np.log(class_pred)

        alphas = [q[0] for q in data['growth']]
        growth_quantiles = [q[1] for q in data['growth']]
        non_parametric_loss = non_parametric_quantile_loss(growth_quantiles, true_growths[i], alphas)

        if in_place:
            predictions[asset]['class_loss'] = class_loss
            predictions[asset]['parametric_score'] = parametric_scores[i]
            predictions[asset]['non_parametric_loss'] = non_parametric_loss
        else:
            scores.append(
                {
                    'asset': asset,
                    'class_loss': class_loss,
                    'parametric_score': parametric_scores[i],
                    'non_parametric_loss': non_parametric_loss
                }
            )

    if not in_place:
        return pd.DataFrame(scores)

    return predictions


def dirichlet_bayes(likelihood: np.ndarray, p_k: np.ndarray, alphas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Bayesian update of the asset's class prediction.
    Does not modify p_k and alphas in place, returns updated versions.

    :param likelihood:      Likelihood matrix for the indicator
    :param p_k:             Class predictions
    :param alphas:          Dirichlet prior alpha parameters
    :return:                Updated class prediction posterior, and alphas
    """
    ###
    # MAP method.
    # From some brief experimenting, is essentially same outcome but very slightly worse variance
    # alphas_star = alphas + posterior - 1
    # alphas += alphas_star / np.sum(alphas_star)
    ###
    posterior = p_k * likelihood
    posterior /= np.sum(posterior)
    # Avoiding modifying in place, just in case
    _alphas = alphas + posterior
    return posterior, _alphas

def dirichlet_bayes_codegen(likelihood: np.ndarray, p_k: np.ndarray, alphas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    load the codegen function from cristal.llm.codegen.dirichlet_bayes and return it
    """
    raise NotImplementedError("dirichlet_bayes_codegen not implemented yet")

def indicator_inference(prior: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """
    Update an indicator likelihood prior using Bayesian inference, from N observations of indicator colors

    Args:
        prior: Array of shape (n_category, n_indicator, n_color) for the indicator likelihoods across
            category, indicator, and color.
        observations: Array of shape (N, 1 + n_indicator), where the i-th row observation is n integer
            vector, where j=0 is the category (integer), and j=1:n_indicator is the indicator color (as integer)

    Returns:
        np.ndarray: the updated prior.
    """
    posterior = prior.copy()
    n_cat = prior.shape[0]
    J = prior.shape[1]
    n_color = prior.shape[2]

    for k_val in range(n_cat):
        for j in range(J):  # Iterate over indicators
            for c in range(n_color):  # Iterate over colors
                posterior[k_val, j, c] = prior[k_val, j, c] + np.sum(
                    (observations[:, j+1] == c) & (observations[:, 0] == k_val)
                )
    epsilon = 1e-10  # Small value to prevent division by zero
    normalized = posterior / (np.sum(posterior, axis=2, keepdims=True) + epsilon)
    return normalized

def outer_indicator_inference(data: dict) -> np.ndarray:
    """
    Updates the likelihood matrix based on the observed data.

    Args:
        data: A dictionary containing the observed data for each asset. The keys are the asset names,
            and the values are dictionaries with 'category' and 'indicators' keys. The 'category' key contains
            the class label of the asset, and the 'sampled_colors' key contains a dictionary of the observed
            indicators their their true colors.

    Returns:
        np.ndarray: The likelihood matrix with dimensions (n_category, n_indicator, n_color)
    """
    # Reshape the dictionary input data to a NumPy array
    # N: number of observations, J: number of indicators
    indicators: list[str] = list(data.values())[0]['sampled_colors'].keys()
    categories = ["high_growth", "stable", "time_bomb"]
    n_category = len(categories)
    n_color = 3

    N = len(data)
    J = len(indicators)
    # Each row an asset; Each value is 0,1,2 based on the color of the asset i and indicator j
    # j=0 is category
    observations = np.zeros((N, J+1), dtype=int)

    for i, (asset_name, asset_data) in enumerate(data.items()):
        # First column is the category
        observations[i, 0] = categories.index(asset_data['category'])
        # Remaining columns are the indicators
        for j, indicator in enumerate(indicators):
            x_ij = asset_data['sampled_colors'][indicator]
            x_ij = 0 if x_ij == 'red' else 1 if x_ij == 'yellow' else 2  # Convert color to integer
            observations[i, j+1] = x_ij

    # Update beta (likelihood matrix).
    prior = np.ones((n_category, J, n_color))
    posterior = indicator_inference(prior, observations)
    return posterior

#
# def outer_indicator_inference_old(data: dict) -> np.ndarray:
#     """
#     Updates the likelihood matrix based on the observed data.
#
#     Args:
#         data: A dictionary containing the observed data for each asset. The keys are the asset names,
#             and the values are dictionaries with 'category' and 'indicators' keys. The 'category' key contains
#             the class label of the asset, and the 'sampled_colors' key contains a dictionary of the observed
#             indicators their their true colors.
#
#     Returns:
#         np.ndarray: The likelihood matrix with dimensions (n_category, n_indicator, n_color)
#     """
#     # Reshape the dictionary input data to a NumPy array
#     # N: number of observations, J: number of indicators
#     indicators: list[str] = list(data.values())[0]['sampled_colors'].keys()
#     categories = ["high_growth", "stable", "time_bomb"]
#     n_category = len(categories)
#     n_color = 3
#
#     N = len(data)
#     J = len(indicators)
#     # Each row an asset; Each value is 0,1,2 based on the color of the asset i and indicator j
#     X = np.zeros((N, J), dtype=int)
#
#     for i, (asset_name, asset_data) in enumerate(data.items()):
#         for j, indicator in enumerate(indicators):
#             x_ij = asset_data['sampled_colors'][indicator]
#             x_ij = 0 if x_ij == 'red' else 1 if x_ij == 'yellow' else 2  # Convert color to integer
#             X[i, j] = x_ij
#
#     # Update beta (likelihood matrix).
#     prior = np.ones((n_category, J, n_color))
#     posterior = prior.copy()
#     for j in range(J):  # Iterate over indicators
#         for k_val, category in enumerate(categories):  # Iterate over categories
#             for c in range(prior.shape[2]):  # Iterate over colors
#                 posterior[k_val, j, c] = prior[k_val, j, c] + np.sum(
#                     (X[:, j] == c) & (np.array([asset['category'] for asset in data.values()]) == category)
#                 )
#     epsilon = 1e-10  # Small value to prevent division by zero
#     normalized = posterior / (np.sum(posterior, axis=2, keepdims=True) + epsilon)
#     return normalized
#
#     self.indicators['likelihood'] = (('class', 'indicator', 'color'),
#     posterior / (np.sum(posterior, axis=2, keepdims=True) + epsilon))
#
#
