import numpy as np
import pandas as pd
from scipy.stats import norm
import argparse
import json
import os
import numpy as np
from tqdm import tqdm
from alla.logger import logger
from alla import synthetic as synth, utils
from alla.alla import ALLA, BudgetMode
from alla.llm import get_llm, utils as llm_utils
from alla.llm.wrangling import read_reports

from alla.stats import evaluate_performance


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1: Compare ALLA vs LLM estimation of asset growth and category."
    )
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode"
    )
    parser.add_argument(
        "--llm-code", type=str, default="llama-tiny", help="LLM identifier (code or alias). Default 'llama-tiny'."
    )
    parser.add_argument(
        "--name", type=str, default="latest", help="Experiment name. Default 'latest'."
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help=f"Random seed. Default {SEED}."
    )
    parser.add_argument(
        "--temp", type=float, default=0.0, help="Temperature for LLM. Default 0.0."
    )
    parser.add_argument(
        "--raise-exception", action="store_true", help="Raise exceptions on certain iteration failures."
    )

    args = parser.parse_args()

    llm_code = args.llm_code
    run_name = args.name
    testing = args.test
    temp = args.temp
    raise_exception = args.raise_exception
    # Seed rng
    seed = args.seed
    np.random.seed(seed)
    if testing:
        predictions = {
            'asset1': {
                'class_probs': [0.1, 0.7, 0.2],
                'growth_quantiles': [0.05, 0.1, 0.15],
                'alphas': [0.1, 0.5, 0.9]
            },
            'asset2': {
                'class_probs': [0.3, 0.4, 0.3],
                'growth_quantiles': [0.02, 0.08, 0.12],
                'alphas': [0.1, 0.5, 0.9]
            }
        }

        ground_truth = {
            'asset1': {
                'true_class': 1,
                'true_growth': 0.12
            },
            'asset2': {
                'true_class': 2,
                'true_growth': 0.1
            }
        }
    else:
        ##to be implemented
        #read predictions and ground_truth from files
        #ground_truth = synth.load_stocks(as_dataset=True)
        #predictions = synth.load_stocks(as_dataset=True)
        predictions = None
        ground_truth = None
            
    performance_df = evaluate_performance(predictions, ground_truth)
    # Create results directory if it doesn't exist
    results_dir = os.path.join(utils.PROJECT_ROOT, f"results/exp-1/{run_name}")
    os.makedirs(results_dir, exist_ok=True)

    # Save the performance DataFrame to a CSV file
    performance_df.to_csv(os.path.join(results_dir, f'performance_results_{llm_code}.csv'), index=False)
    print(performance_df)
    print(performance_df.mean(numeric_only=True))
    # Calculate mean and standard deviation of performance metrics
    performance_summary = performance_df.describe().loc[['mean', 'std']]
    # Save the performance summary to a CSV file
    performance_summary.to_csv(os.path.join(results_dir, f'performance_summary_{llm_code}.csv'))

if __name__ == "__main__":

    SEED = 101010

    main()
