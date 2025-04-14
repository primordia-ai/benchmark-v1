"""
Test an LLM actually trying to generate codegen code.
"""
import os

import numpy as np

from cristal import synthetic as synth
from cristal.env import SYSTEM_USER
from cristal.cristal import CRISTAL
from cristal.synthetic import Indicators
from cristal.llm import get_llm
from cristal.llm.codegen.likelihood_inference import IndicatorLikelihoodInference
from cristal.llm.codegen.model import LLMCodeGen
from cristal.llm.codegen.validator import LLMCodeGenException
from cristal.logger import logger


def format_training_data(data: dict) -> tuple[np.ndarray, np.ndarray]:
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

    prior = np.ones((n_category, J, n_color)) / n_color
    return prior, observations


if __name__ == '__main__':
    SEED = 101010
    from cristal.logger import DEBUG, reset_logger

    reset_logger(DEBUG)

    if SYSTEM_USER == "dsfhswf":
        llm_cls = get_llm("mistral-nemo")
    else:
        llm_cls = get_llm("llama-small")

    logger.info(f"Initializing LLM {llm_cls.code}")
    llm = llm_cls(temperature=0, seed=SEED)

    llm_log = "./tmp/test_codegen_with_llm_likelihoods.log"
    # Remove if exists
    if os.path.exists(llm_log):
        os.remove(llm_log)
    llm.log_to_file(llm_log)

    codegen_llm = LLMCodeGen(llm)
    max_retries, budget = 5, 60

    # Get training samples
    logger.info("Testing on sample training data")
    n_shots, train_size = 5, 100
    training_data = synth.get_ground_truth_samples(n_shots, train_size)
    prior, observations = format_training_data(training_data)

    try:
        code = codegen_llm.generate_function(IndicatorLikelihoodInference, max_retries, budget)
    except LLMCodeGenException as e:
        logger.error(f"Failed to generate code: {e}")
        raise e

    # Test the generated code
    try:
        posterior = code(prior, observations)
    except LLMCodeGenException as e:
        logger.error(f"Validation failed: {e}")
        raise e

    # Print output
    logger.info(f"Resulting posterior:\n{posterior}")

    # Now test with the CRISTAL likelihood bit
    stocks = synth.load_stocks(test=True, train_size=train_size)  # synth.stocks_to_xarray(stocks)
    m = CRISTAL(
        stocks, llm, asset_memory_depth=3, use_codegen=True,
    )
    # Instead of loading again we just directly add it
    m.loaded_codegens[IndicatorLikelihoodInference.function_name] = code

    # Perform the inference stage
    m.likelihood_inference(training_data)

    # Test likelihoods
    alphas = m.stocks.alphas.values[-1, :, -1].copy()
    p_k = m.stocks.prior.values[-1, :, -1].copy()
    indi = m.ind_likelihood("brand_strength")
    print(f"MW INDI: {indi.shape} | {indi}")

    # Process an asset
    first_asset = m.stocks.asset.values[0]
    m.process_asset(first_asset)

    # for ind in Indicators.names:
    #     l_pre = m.ind_likelihood(ind).copy()
    #     color_pred = np.eye(3)[0]
    #     likelihood = m.ind_likelihood(ind) @ color_pred.reshape(-1, 1)
    #     likelihood = likelihood.reshape(3, )
    #     print(f"MW DEBUG: {ind} | {l_pre} | {likelihood}")
