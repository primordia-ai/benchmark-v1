"""
Here we compare the performance of hard-coded ALLA vs LLM estimation of asset growth and category
on our synthetic dataset of assets.

This uses the LLM-generated synthetic reports found in data/reports,
and the LLM-produced Soft-Indicator Extractor indicator color results that were based off these reports.

In this first experiment we provide an LLM (we'll use llama) with the synthetic reports, the soft-indicator
extraction results, likelihood matrices, etc., and ask it to estimate next year's growth and asset category.

With ALLA, we will used our hardcoded version that performs inference on the soft-indicator extracted
indicators, and the logic around optimized selection given resources, etc. See ALLA.process_asset.

In this experiment, for each asset the agents are allowed a fixed budget of wall-time to process an asset.
ALLA will perform bayesian inference on the indicators, and the LLM will converse back with itself during
its allotted time.

TODO -- Questions
    1. Is "Actual" in our setup the underlying mixture distribution parameters (mu_g) or the sampled 2025_growth?
    2. How to get unbiased growth estimates from llama if we are giving it the three mixture means?
        a. Conversely, without the three mixture means, we will have to describe the range of growth values somehow.
    3. Need the rest of the reports for other assets.
    4. Need to construct actual budget mechanics.

TODO -- Ablations
    1. Chain-of-thought: https://python.useinstructor.com/concepts/prompting/#chain-of-thought
"""

import argparse
import json
import instructor
import logging
import os

import numpy as np
from tqdm import tqdm

# Set this up before importing local classes
logger = logging.getLogger("alla")
logger.setLevel(logging.INFO)

from alla import synthetic as synth, utils
from alla.alla import ALLA, BudgetMode
from alla.llm import get_llm, utils as llm_utils
from alla.llm.wrangling import read_reports

# Set seed
# 101010 happens to be 42 in binary. How charming.
SEED = 101010


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

    results_dir = os.path.join(utils.PROJECT_ROOT, f"results/exp-1/{run_name}")
    os.makedirs(results_dir, exist_ok=True)

    stocks = synth.load_stocks(as_dataset=True)
    categories = ["high_growth", "stable", "time_bomb"]
    assets = {cat: [] for cat in categories}

    if testing:
        # If testing, we will have 12 assets, and 4 per category
        n_asset = 9
        n_asset_per = n_asset / 3
        budget_mode = BudgetMode.WALL
        total_budget = 1e5
        budget_per_task = 20
        asset_memory_depth = 20
    else:
        # Otherwise, just run all assets
        n_asset = 200
        n_asset_per = None
        budget_mode = BudgetMode.WALL
        total_budget = 1e8  # arbitrarily large
        budget_per_task = 20
        asset_memory_depth = 100

    # Initialize our LLM
    llm = get_llm(llm_code)(temperature=temp)

    # Create ALLA model
    m = ALLA(
        stocks, llm, asset_memory_depth=asset_memory_depth,
        budget_mode=budget_mode, budget=total_budget,
    )

    # We can just compile asset names here
    # Only for assets that we have reports for
    report_dir = os.path.join(utils.PROJECT_ROOT, "data/reports")
    truth_data = {}
    for asset_id in os.listdir(report_dir):
        # Check category
        try:
            asset = m.stocks.sel(asset=asset_id)
        except KeyError as e:
            if raise_exception:
                raise e
            # Else log and continue
            print(f"Asset {asset_id} not found in stocks. Continuing.")
            continue

        category = asset.category.values.item()
        if testing and len(assets[category]) >= n_asset_per:
            # We have enough assets for this category
            continue

        # Add asset to category and record truth data
        assets[category].append(asset_id)
        truth_data[asset_id] = {
            "category": category,
            "growth": asset.growth_2025.values.item()
        }

    # Save run data to config.yaml
    run_data = {
        "seed": seed,
        "test": testing,
        "llm": llm.to_json(),
        "alla": m.to_json(),
        "budget_mode": str(budget_mode),
        "budget_per_task": budget_per_task,
        "total_budget": total_budget,
        "n_asset": n_asset,
        "n_asset_per": n_asset_per,
        "assets": {cat: len(reports) for cat, reports in assets.items()}
    }
    config_path = os.path.join(results_dir, "config.json")
    with open(config_path, 'w') as f:
        logger.info(f"Saving run data to {config_path}")
        json.dump(run_data, f, indent=4)

    # Also save the truth data
    truth_data_path = os.path.join(results_dir, "truth_data.json")
    with open(truth_data_path, 'w') as f:
        logger.info(f"Saving truth data to {truth_data_path}")
        json.dump(truth_data, f, indent=4)

    for cat, ids in assets.items():
        print(f"{len(ids)} reports found for '{cat}':\n{ids}")
    #
    # #####################
    # # ALLA portion
    #
    # for cat, a_ids in assets.items():
    #     print(f"\n=============  ALLA  =============\nProcessing '{cat}' assets ({len(a_ids)})\n")
    #     for asset_id in tqdm(a_ids, desc="Assets"):
    #         try:
    #             success, reason = m.process_asset(asset_id, max_budget=budget_per_task)
    #         except Exception as e:
    #             if raise_exception:
    #                 raise e
    #             # Else log and continue
    #             logger.warning(f"Exception processing asset {asset_id}: {e}")
    #         else:
    #             if not success:
    #                 if raise_exception:
    #                     raise ValueError(f"Failed to process asset {asset_id}: {reason}")
    #                 # Else log and continue
    #                 logger.warning(f"Failed to process asset {asset_id}: {reason}")
    #
    # # Save the alla model
    # m.save_model(
    #     model_name=f"alla-{run_name}",
    #     save_dir=os.path.join(results_dir, "models")
    # )

    #####################
    # LLM Portion
    #
    # For each asset combine its report text and the pre-extract indicators into one prompt

    # The llm that generated the indicators doesn't necessarily have to be the subject model
    # (At least for this experiment)
    indicator_llm_code = "llama3:8b"
    llm_results = {}

    for cat, a_ids in assets.items():
        print(f"\n=============  {llm.name}  =============\nProcessing '{cat}' assets ({len(a_ids)})\n")
        for asset_id in tqdm(a_ids, desc="Assets"):
            # Load indicators
            indicator_file = os.path.join(
                utils.PROJECT_ROOT, "data/extracted_indicators", indicator_llm_code, f"{asset_id}.json"
            )
            with open(indicator_file, 'r') as f:
                indicators = json.load(f)
            # Get report text
            report_file = os.path.join(utils.PROJECT_ROOT, f"data/reports/{asset_id}")
            reports_text = read_reports(report_file)

            prompt = llm_utils.prompt_exp1_task(asset_id, reports_text, indicators)
            try:
                evaluation = llm.evaluate_asset(prompt, max_retries=5)
            except Exception as e:
                if raise_exception:
                    raise e
                # Else log and continue
                logger.warning(f"Exception processing asset {asset_id}: {e}")
                llm_results[asset_id] = {
                    "error": str(e),
                    "truth": truth_data[asset_id]
                }
            else:
                result = evaluation.model_dump()
                result["truth"] = truth_data[asset_id]
                llm_results[asset_id] = result

    # Save the llm results
    llm_results_path = os.path.join(results_dir, "llm_results.json")
    with open(llm_results_path, 'w') as f:
        logger.info(f"Saving LLM results to {llm_results_path}")
        json.dump(llm_results, f, indent=4)


if __name__ == '__main__':
    main()
