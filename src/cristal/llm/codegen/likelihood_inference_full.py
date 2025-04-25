import numpy as np
from pydantic import model_validator, ValidationError

from cristal.llm.codegen.validator import LLMCodeGenValidator, load_code, LLMCodeGenException
from cristal.synthetic import get_ground_truth_samples


class LikelihoodInferenceFull(LLMCodeGenValidator):
    """
    Full formulation for performing inference provided only the dictionary of asset data observations to build
    the indicator likelihood matrix. Each indicator has a different set of likelihoods given category.

    Args:
        data: A dictionary containing the observed data for each asset. The keys are the asset names,
            and the values include dictionaries with 'category' and 'sampled_colors' keys. Each 'category' item contains
            the category label of the asset, and the 'sampled_colors' item is a dictionary keyed by indicator name,
            and observed color (str). Colors are one of three ['red', 'yellow', 'green'], corresponding to
            the asset's performance in this indicator dimension.
    Returns:
        np.ndarray: The likelihood matrix with dimensions (n_category, n_indicator, n_color)
    """
    is_function = True
    function_name = "likelihood_inference_full"

    prompt_description = """
    Consider the following statistical model and code its solution. 
    We are trying to categorize a company into one of K classes and we observe J categorical indicators, 
    with the same number of categories (colors) C. 
    Each indicator has a different set of likelihoods given classes. We represent the sampling distributions as

    k ~ Categorical(p)
    x_j ~ Categorical(A_{j,:,k}) # The array A (J x C x K) contains the likelihoods of each indicator given each of the latent company class k.

    From a data dictionary of asset category and indicator color observations, use Bayesian inference to
    return the indicator likelihood matrix A.
    
    Here is a description of the input data dictionary:
    "The keys are the asset names, and the values include dictionaries with 'category' and 'sampled_colors' keys. 
    Each 'category' item contains the category label of the asset, and the 'sampled_colors' item is a dictionary 
    keyed by indicator name, and observed color (str). Colors are one of three ['red', 'yellow', 'green'], 
    corresponding to the asset's performance in this indicator dimension."
    """

    required_args = tuple(
        [
            {
                "arg": "data",
                "description": "Data dictionary of asset observations",
                "datatype": dict
            },
        ]
    )
    returned_data = tuple(
        [
            {
                "name": "Likelihood Matrix",
                "description": "The likelihood matrix with dimensions (n_category, n_indicator, n_color)",
                "type": np.ndarray
            }
        ]
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @model_validator(mode="after")
    def test_output(self):
        """
        Verify some aspects of the return value

        We can compare output values with the function above, indicator_likelihood()
        """
        # Load some sample training data
        n_shots, train_size = 5, 100  # Produces 15 samples
        training_data = get_ground_truth_samples(n_shots, train_size)
        # Get likelihood solution
        solution = indicator_likelihood(training_data)

        # Get solution from the function
        func = load_code(self.code, self.function_name)
        try:
            result = func(training_data)
        except Exception:
            raise ValidationError("Function failed to run with the provided data.")

        # Compare result with solution
        if result.shape != solution.shape:
            raise LLMCodeGenException(f"Result shape {result.shape} does not match solution shape {solution.shape}")

        if not np.allclose(result.sum(axis=2), 1):
            raise LLMCodeGenException("Posterior probabilities do not sum to 1")

        # Relative tolerance
        eps = 1e-6
        if not np.allclose(result, solution, rtol=eps):
            raise LLMCodeGenException(f"Result does not match the expected solution.\nresult[0] = {result[0]}")

        return self


def test_code(code: str) -> None:
    """
    Externalize the testing procedure so that the Critic can receive precise information
    """
    # Load some sample training data
    n_shots, train_size = 5, 100  # Produces 15 samples
    training_data = get_ground_truth_samples(n_shots, train_size)
    # Get likelihood solution
    solution = indicator_likelihood(training_data)

    # Get solution from the function
    func = load_code(code, LikelihoodInferenceFull.function_name)
    try:
        result = func(training_data)
    except Exception:
        raise ValidationError("Function failed to run with the provided data.")

    # Compare result with solution
    if result.shape != solution.shape:
        raise LLMCodeGenException(f"Result shape {result.shape} does not match solution shape {solution.shape}")

    if not np.allclose(result.sum(axis=2), 1):
        raise LLMCodeGenException("Posterior probabilities do not sum to 1")

    # Relative tolerance
    eps = 1e-6
    if not np.allclose(result, solution, rtol=eps):
        raise LLMCodeGenException(f"Result does not match the expected solution.\nresult[0] = {result[0]}")

    return None

# For testing
solution_string = '''
import numpy as np

def likelihood_inference_full(data: dict) -> np.ndarray:
    """
    Updates the likelihood matrix based on the observed data.

    Args:
        data: A dictionary containing the observed data for each asset. The keys are the asset names,
            and the values include dictionaries with 'category' and 'sampled_colors' keys. Each 'category' item contains
            the category label of the asset, and the 'sampled_colors' item is a dictionary keyed by indicator name,
            and observed color (str). Colors are one of three ['red', 'yellow', 'green'], corresponding to
            the asset's performance in this indicator dimension.
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
    observations = np.zeros((N, J + 1), dtype=int)

    for i, (asset_name, asset_data) in enumerate(data.items()):
        # First column is the category
        observations[i, 0] = categories.index(asset_data['category'])
        # Remaining columns are the indicators
        for j, indicator in enumerate(indicators):
            x_ij = asset_data['sampled_colors'][indicator]
            x_ij = 0 if x_ij == 'red' else 1 if x_ij == 'yellow' else 2  # Convert color to integer
            observations[i, j + 1] = x_ij

    # Update beta (likelihood matrix).
    prior = np.ones((n_category, J, n_color))
    posterior = prior.copy()
    n_cat = prior.shape[0]
    J = prior.shape[1]
    n_color = prior.shape[2]

    for k_val in range(n_cat):
        for j in range(J):  # Iterate over indicators
            for c in range(n_color):  # Iterate over colors
                posterior[k_val, j, c] = prior[k_val, j, c] + np.sum(
                    (observations[:, j + 1] == c) & (observations[:, 0] == k_val)
                )
    epsilon = 1e-10  # Small value to prevent division by zero
    normalized = posterior / (np.sum(posterior, axis=2, keepdims=True) + epsilon)
    return normalized

'''

TASK_HARD = """
We are trying to categorize a company into one of K classes and we observe J categorical indicators.  
Each indicator can take one of C colors, that characterize the company's value in the dimension of that indicator.
Each indicator has a different set of likelihoods given classes. We represent the sampling distributions as

k ~ Categorical(p)
x_j ~ Categorical(A_{j,:,k}) # The array A (J x C x K) contains the likelihoods of each indicator given each of the latent company class k.

The goal is to update an indicator likelihood prior using Bayesian inference, from N observations of indicator colors and 
categories. Each indicator has a different set of likelihoods given classes.

Here is a description of the input data, and the requirement of the output format:

Args:
    data: A dictionary containing the observed data for each asset. The keys are the asset names,
        and the values are dictionaries with 'category' and 'indicators' keys. The 'category' key contains
        the class label of the asset, and the 'sampled_colors' key contains a dictionary of the observed
        indicators their their true colors.

Returns:
    np.ndarray: The likelihood matrix with dimensions (n_category, n_indicator, n_color)

"""

TASK_EASY = TASK_HARD + """

Tip:
    First work out the closed-form Bayesian update for the class probability vector p.
    Then extend this to assume the likelihood matrix is also unknown and we must simultaneously infer its parameters.
    
Recall that the unnormalized Bayesian posterior is:
posterior = likelihood * prior
where the likelihood is the indicator likelihood given the class and the prior is the prior distribution of classes.

"""









##############################
# For reference and comparison


def indicator_likelihood(data: dict) -> np.ndarray:
    """
    Updates the likelihood matrix based on the observed data.

    Args:
        data: A dictionary containing the observed data for each asset. The keys are the asset names,
            and the values include dictionaries with 'category' and 'sampled_colors' keys. Each 'category' item contains
            the category label of the asset, and the 'sampled_colors' item is a dictionary keyed by indicator name,
            and observed color (str). Colors are one of three ['red', 'yellow', 'green'], corresponding to
            the asset's performance in this indicator dimension.
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
    observations = np.zeros((N, J + 1), dtype=int)

    for i, (asset_name, asset_data) in enumerate(data.items()):
        # First column is the category
        observations[i, 0] = categories.index(asset_data['category'])
        # Remaining columns are the indicators
        for j, indicator in enumerate(indicators):
            x_ij = asset_data['sampled_colors'][indicator]
            x_ij = 0 if x_ij == 'red' else 1 if x_ij == 'yellow' else 2  # Convert color to integer
            observations[i, j + 1] = x_ij

    # Update beta (likelihood matrix).
    prior = np.ones((n_category, J, n_color))
    posterior = prior.copy()
    n_cat = prior.shape[0]
    J = prior.shape[1]
    n_color = prior.shape[2]

    for k_val in range(n_cat):
        for j in range(J):  # Iterate over indicators
            for c in range(n_color):  # Iterate over colors
                posterior[k_val, j, c] = prior[k_val, j, c] + np.sum(
                    (observations[:, j + 1] == c) & (observations[:, 0] == k_val)
                )
    epsilon = 1e-10  # Small value to prevent division by zero
    normalized = posterior / (np.sum(posterior, axis=2, keepdims=True) + epsilon)
    return normalized


def indicator_likelihood_alt(data: dict) -> np.ndarray:
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
    X = np.zeros((N, J), dtype=int)

    for i, (asset_name, asset_data) in enumerate(data.items()):
        for j, indicator in enumerate(indicators):
            x_ij = asset_data['sampled_colors'][indicator]
            x_ij = 0 if x_ij == 'red' else 1 if x_ij == 'yellow' else 2  # Convert color to integer
            X[i, j] = x_ij

    # Update beta (likelihood matrix).
    prior = np.ones((n_category, J, n_color))
    posterior = prior.copy()
    for j in range(J):  # Iterate over indicators
        for k_val, category in enumerate(categories):  # Iterate over categories
            for c in range(prior.shape[2]):  # Iterate over colors
                posterior[k_val, j, c] = prior[k_val, j, c] + np.sum(
                    (X[:, j] == c) & (np.array([asset['category'] for asset in data.values()]) == category)
                )
    epsilon = 1e-10  # Small value to prevent division by zero
    normalized = posterior / (np.sum(posterior, axis=2, keepdims=True) + epsilon)
    return normalized
