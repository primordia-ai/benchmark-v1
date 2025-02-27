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

TODO -- Ablations
    1. Chain-of-thought: https://python.useinstructor.com/concepts/prompting/#chain-of-thought
    2. LLM budgeting and conversing with itself
"""


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1: Compare ALLA vs LLM estimation of asset growth and category.",
        epilog="""
        - Choose from one of three modes for your experiment: 'alla', 'llm', or 'both'.
        - Pre-extracted as True loads indicator colors that were previously extracted. This is useful for debugging 
        and development.
        - Test is a reduced run that selects only a couple assets for each category, and has a reduced 
        budget for tasks for a more hasty run-through.
        """
    )
    parser.add_argument(
        "--mode", type=str, help="Whether to run 'alla' or 'llm' or 'both'. Note that 'alla' still requires an "
                                 "llm-code for its usage."
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
        "--strict", action="store_true", help="Do not ignore exceptions on asset evaluation. By default this "
                                              "is excluded, so that other assets can continue to be evaluated."
    )
    parser.add_argument(
        "--pre-extracted", type=str, default=None, help="Specify a directory (relative to data/extracted_indicators/) "
                                                        "for pre-extracted indicators"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO", help="Logging level. Default 'INFO'."
    )
    parser.add_argument(
        "--no-input", action="store_true", help="Skip any input prompting"
    )
    parser.add_argument(
        "--budget", type=int, default=30, help="Budget per task in seconds. Default 30."
    )

    args = parser.parse_args()

    mode = args.mode
    if mode not in ["alla", "llm", "both"]:
        raise ValueError(f"Invalid mode '{mode}'")

    # Seed rng
    seed = args.seed
    np.random.seed(seed)
    # Log level
    loglevel = args.log_level
    file_loglevel = "DEBUG" if loglevel == "DEBUG" else "INFO"
    if loglevel != "INFO":
        reset_logger(loglevel)

    # Run dir
    run_name = args.name
    results_dir = os.path.join(PROJECT_ROOT, f"results/exp-1/{run_name}")
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "out.log")
    llm_log_path = os.path.join(results_dir, "llm.log")
    # Set logging to file
    log_to_file(log_path, level=file_loglevel)

    # Get some info about the run for adding to the config file
    no_input = args.no_input
    description = ""
    command_str = "python " + " ".join(sys.argv)
    if not no_input:
        # Prompt user for description
        desc_path = os.path.join(results_dir, "description.txt")
        description = capture_description(logger)

    llm_code = args.llm_code
    testing = args.test
    temp = args.temp
    strict = args.strict
    budget_per_task = args.budget
    pre_extract_dir = args.pre_extracted
    if pre_extract_dir:
        pre_extract = True
        if not os.path.exists(pre_extract_dir):
            raise ValueError(f"Pre-extracted directory '{pre_extract_dir}' does not exist")
    else:
        pre_extract = False

    categories = ["high_growth", "stable", "time_bomb"]
    mixture_params = {
        "mu": {
            "high_growth": synth.GROWTH_MU[0],
            "stable": synth.GROWTH_MU[1],
            "time_bomb": synth.GROWTH_MU[2],
        },
        "sigma": {
            "high_growth": synth.GROWTH_SCALE[0],
            "stable": synth.GROWTH_SCALE[1],
            "time_bomb": synth.GROWTH_SCALE[2],
        },
    }
    quantiles = [0.025, 0.16, 0.5, 0.84, 0.975]

    assets = {cat: [] for cat in categories}

    if testing:
        # If testing, we will have 12 assets, and 4 per category
        n_asset = 15
        n_asset_per = n_asset / 3
        budget_mode = BudgetMode.WALL
        total_budget = 300  # 5 minutes
        max_retries = 3
        asset_memory_depth = 100
    else:
        # Otherwise, just run all assets
        n_asset = 200
        n_asset_per = None
        budget_mode = BudgetMode.WALL
        total_budget = 1e8  # arbitrarily large
        max_retries = 10
        asset_memory_depth = 100

    if mode in ['llm', 'both'] and budget_mode != BudgetMode.WALL:
        raise ValueError("LLM mode only available for WALL budget")

    # Initialize our LLM
    llm = get_llm(llm_code)(temperature=temp, seed=seed)
    # Activate logging LLM
    logger.info(f"Piping LLM logs to {llm_log_path}")
    llm.log_to_file(llm_log_path)

    # We can just compile asset names here
    # Only for assets that we have reports for
    report_dir = os.path.join(PROJECT_ROOT, "data/reports", llm.path_code)
    truth_data = {}
    stocks = synth.load_stocks(as_dataset=False)
    financials = {}  # For use by LLM
    for asset_id in os.listdir(report_dir):
        # Check category
        category = stocks[asset_id]['category']
        if n_asset_per and len(assets[category]) >= n_asset_per:
            # We have enough assets for this category
            continue

        # Add asset to category and record truth data
        assets[category].append(asset_id)
        truth_data[asset_id] = {
            "category": category,
            "growth": stocks[asset_id]['growth_2025'],
        }
        financials[asset_id] = stocks[asset_id]['financials']

    # Save run data to config.yaml
    run_data = {
        "command": command_str,
        "description": description,
        "seed": seed,
        "test": testing,
        "llm": llm.to_json(),
        "budget_mode": str(budget_mode),
        "budget_per_task": budget_per_task,
        "total_budget": total_budget,
        "n_asset": n_asset,
        "n_asset_per": n_asset_per,
        "assets": {cat: len(reports) for cat, reports in assets.items()}
    }

    # Convert stocks to dataset now
    stocks = synth.stocks_to_xarray(stocks)
    if mode in ['alla', 'both']:
        # Create ALLA model
        m = ALLA(
            stocks, llm, asset_memory_depth=asset_memory_depth,
            budget_mode=budget_mode, budget=total_budget,
        )
        run_data['alla'] = m.to_json()

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
        logger.info(f"{len(ids)} reports found for '{cat}'")

    if mode in ["alla", "both"]:
        #####################
        # ALLA portion
        alla_results = {}

        with logging_redirect_tqdm(loggers=[logger]):
            for cat, a_ids in assets.items():
                logger.info(f"\n=============  ALLA  =============\nProcessing '{cat}' assets ({len(a_ids)})\n")
                for asset_id in tqdm(a_ids, desc="Assets"):
                    extracted_colors = None
                    if pre_extract:
                        # Load the pre-extracted indicators
                        # Technically you don't need to use the {llm.code} subset
                        extract_path = os.path.join(
                            PROJECT_ROOT, f"data/extracted_indicators", pre_extract_dir, f"{asset_id}.json"
                        )
                        # Result is list of {"value": .., "indicator": ..} items
                        extracted_colors = llm_utils.load_extracted_indicators(
                            extract_path, keys=["value", "indicator"]
                        )
                        # Convert to dict of values
                        extracted_colors = {item['indicator']: item['value'] for item in extracted_colors}

                    try:
                        success, reason = m.process_asset(
                            asset_id, max_budget=budget_per_task, pre_extracted=extracted_colors
                        )
                    except Exception as e:
                        if strict:
                            raise e
                        # Else log and continue
                        logger.warning(f"Exception processing asset {asset_id}: {e}")
                        alla_results[asset_id] = {
                            "error": str(e),
                            "truth": truth_data[asset_id]
                        }
                    else:
                        if not success:
                            alla_results[asset_id] = {
                                "error": reason,
                                "truth": truth_data[asset_id]
                            }
                            if strict:
                                raise ValueError(f"Failed to process asset {asset_id}: {reason}")
                            # Else log and continue
                            logger.warning(f"Failed to process asset {asset_id}: {reason}")
                            continue

                        # Record the growth quantiles and the category distribution
                        quantile_pred = m.stocks.sel(asset=asset_id).quantile_pred.values[:, -1]
                        cats = m.stocks.sel(asset=asset_id).prior.values[:, -1]
                        alla_results[asset_id] = {
                            "growth": tuple([(q, p) for q, p in zip(m.quantiles, quantile_pred)]),
                            "category": {c: p for c, p in zip(['high_growth', 'stable', 'time_bomb'], cats)},
                            "truth": truth_data[asset_id]
                        }

        # Add in-place performance evaluation
        stats.evaluate_performance(alla_results, mixture_params, in_place=True)

        # Save ALLA results
        alla_results_path = os.path.join(results_dir, "alla_results.json")
        with open(alla_results_path, 'w') as f:
            logger.info(f"Saving ALLA results to {alla_results_path}")
            json.dump(alla_results, f, indent=4)

        # Save the alla model
        m.save_model(
            model_name=f"alla-{run_name}",
            save_dir=os.path.join(results_dir, "models")
        )

    if mode in ['llm', 'both']:
        #####################
        # LLM Portion
        #

        llm_results = {}
        with logging_redirect_tqdm(loggers=[logger]):
            for cat, a_ids in assets.items():
                logger.info(f"\n=============  {llm.name}  =============\nProcessing '{cat}' assets ({len(a_ids)})\n")
                for asset_id in tqdm(a_ids, desc="Assets"):
                    indicators = None
                    if pre_extract:
                        # Load the pre-extracted indicators
                        # Note that this includes both quantitative and qualitative indicators
                        indicator_file = os.path.join(
                            PROJECT_ROOT, f"data/extracted_indicators", pre_extract_dir, f"{asset_id}.json"
                        )
                        with open(indicator_file, 'r') as f:
                            indicators = json.load(f)
                    else:
                        # We should actually still provide the quantitative indicators since these are 'gimmes'
                        indicators = synth.indicators_from_financials(
                            financials[asset_id], as_colors=True
                        )

                    # Get report text
                    report_file = os.path.join(report_dir, f"{asset_id}")
                    reports_text = read_reports(report_file)

                    # Modifies the prompt if indicators are provided, but they are not required
                    prompt = llm_utils.prompt_exp1_task(reports_text, indicators)
                    # For checking
                    start = time.time()
                    # We might wrap this in try-except, but the method itself does so for LLM request
                    evaluation, success = llm.evaluate_asset(prompt, max_retries=max_retries, budget=budget_per_task)
                    end = time.time()
                    if end - start > budget_per_task:
                        logger.warning(f"Asset '{asset_id}' eval exceeded budget: {end - start} > {budget_per_task}")

                    if not success:
                        error = evaluation['error']
                        logger.warning(f"Failed to evaluate asset {asset_id}: {error}")
                        llm_results[asset_id] = {
                            "error": error,
                            "truth": truth_data[asset_id]
                        }
                    else:
                        # Successful eval
                        # Compute growth quantiles
                        mix_weights = [evaluation['high_growth'], evaluation['stable'], evaluation['time_bomb']]
                        quantile_pred: np.ndarray = synth.compute_growth_quantiles(mix_weights, quantiles=quantiles)
                        llm_results[asset_id] = {
                            "growth": tuple([(q, p) for q, p in zip(quantiles, quantile_pred)]),
                            "category": evaluation,
                            "truth": truth_data[asset_id],
                        }

        # Add in-place performance evaluation
        stats.evaluate_performance(llm_results, mixture_params, in_place=True)

        # Save the llm results
        llm_results_path = os.path.join(results_dir, "llm_results.json")
        with open(llm_results_path, 'w') as f:
            logger.info(f"Saving LLM results to {llm_results_path}")
            json.dump(llm_results, f, indent=4)


if __name__ == '__main__':

    import argparse
    import json
    import os
    import time
    import sys

    import numpy as np
    from tqdm import tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm

    from alla.logger import logger, reset_logger, log_to_file, capture_description

    from alla import synthetic as synth, stats
    from alla.alla import ALLA
    from alla.utils import BudgetMode
    from alla.env import PROJECT_ROOT
    from alla.llm import get_llm, utils as llm_utils
    from alla.llm.wrangling import read_reports
    import pytensor

    pytensor.config.exception_verbosity = 'high'

    # Set seed
    # 101010 happens to be 42 in binary. How charming.
    SEED = 101010

    main()
