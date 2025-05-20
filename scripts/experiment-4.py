"""
Here we compare the performance of hard-coded CRISTAL vs LLM estimation of asset growth and category
on our synthetic dataset of assets.

This uses the LLM-generated synthetic reports found in data/reports,
and the LLM-produced Soft-Indicator Extractor indicator color results that were based off these reports.

In this first experiment we provide an LLM (we'll use llama) with the synthetic reports, the soft-indicator
extraction results, likelihood matrices, etc., and ask it to estimate next year's growth and asset category.

With CRISTAL, we will used our hardcoded version that performs inference on the soft-indicator extracted
indicators, and the logic around optimized selection given resources, etc. See CRISTAL.process_asset.

In this experiment, for each asset the agents are allowed a fixed budget of wall-time to process an asset.
CRISTAL will perform bayesian inference on the indicators, and the LLM will converse back with itself during
its allotted time.

The ablation is: the high growth, and time bomb labels are swapped to check if the methods are doing actual maths or
are using prior/learnt knowledge to make the decisions.
This is a test of the LLM's ability to understand the reports and indicators, and to make predictions based on them.

"""
import xarray as xr
import numpy as np

def swap_labels(stocks):
    """
    Swap the labels 'high_growth' and 'time_bomb' in the given stocks data.

    Args:
        stocks (xarray.Dataset or dict): Stock data with categories.

    Returns:
        xarray.Dataset or dict: Updated stocks with swapped labels.
    """
    if isinstance(stocks, dict):
        for asset_id, data in stocks.items():
            category = data['category']
            if isinstance(category, np.ndarray):
                category = category.item() if category.size == 1 else category.tolist()
            if category == "high_growth":
                data['category'] = "time_bomb"
            elif category == "time_bomb":
                data['category'] = "high_growth"
        return stocks

    elif isinstance(stocks, (xr.Dataset, xr.DataArray)):
        stocks = stocks.copy()  # Avoid modifying the original xarray object
        stocks['category'] = stocks['category'].where(
            stocks['category'] != 'high_growth', 'temp_label'
        )
        stocks['category'] = stocks['category'].where(
            stocks['category'] != 'time_bomb', 'high_growth'
        )
        stocks['category'] = stocks['category'].where(
            stocks['category'] != 'temp_label', 'time_bomb'
        )
        return stocks

    else:
        raise TypeError("Unsupported data type for stocks. Expected dict or xarray.Dataset/DataArray.")

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3: Compare CRISTAL vs LLM estimation of asset growth and category.",
        epilog="""
        - Choose from one of three modes for your experiment: 'cristal', 'llm', or 'both'.
        - Pre-extracted as True loads indicator colors that were previously extracted. This is useful for debugging 
        and development.
        - Test is a reduced run that selects only a couple assets for each category, and has a reduced 
        budget for tasks for a more hasty run-through.
        """
    )
    parser.add_argument(
        "--mode", type=str, help="Whether to run 'cristal' or 'llm' or 'both'. Note that 'cristal' still requires an "
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
    parser.add_argument(
        "--extractor", type=str, default=None, help="LLM code for the extractor. Default is None."
    )
    parser.add_argument(
        "--compress-reports", action="store_true", help="Attempt to compress the reports through recursive summarizing"
    )
    parser.add_argument(
        "--reports-dir", type=str, default=None, help="Reports directory override over env REPORTS_DIR."
    )
    parser.add_argument(
        "--n-shots", type=int, default=0, help="Number of examples to use for the LLM. Default 0."
    )
    parser.add_argument(
        "--train-size", type=int, default=500, help="Number of examples to use for training. Default 500."
    )

    args = parser.parse_args()

    mode = args.mode
    if mode not in ["cristal", "llm", "both"]:
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
    results_dir = os.path.join(PROJECT_ROOT, f"results/exp-4/{run_name}")
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "out.log")
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
    llm_extractor_code = args.extractor
    testing = args.test
    temp = args.temp
    strict = args.strict
    budget_per_task = args.budget
    compress_reports = args.compress_reports
    REPORTS_DIR = args.reports_dir or _REPORTS_DIR
    # Check reports is valid
    if not os.path.exists(REPORTS_DIR):
        raise ValueError(f"Reports directory '{REPORTS_DIR}' does not exist")

    pre_extract_dir = args.pre_extracted
    if pre_extract_dir:
        pre_extract = True
        if not os.path.exists(pre_extract_dir):
            raise ValueError(f"Pre-extracted directory '{pre_extract_dir}' does not exist")
    else:
        pre_extract = False

    categories = ["high_growth", "stable", "time_bomb"]
    # swap them out here
    mixture_params = {
        "mu": {
            "high_growth": synth.GROWTH_MU[2],
            "stable": synth.GROWTH_MU[1],
            "time_bomb": synth.GROWTH_MU[0],
        },
        "sigma": {
            "high_growth": synth.GROWTH_SCALE[2],
            "stable": synth.GROWTH_SCALE[1],
            "time_bomb": synth.GROWTH_SCALE[0],
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
        budget_per_task = 30
        max_retries = 5
        asset_memory_depth = 24

    if mode in ['llm', 'both'] and budget_mode != BudgetMode.WALL:
        raise ValueError("LLM mode only available for WALL budget")

    #####################
    # Configure LLMs
    #
    llm = None
    if mode in ['llm', 'both']:
        # LLM-Base
        llm_log_path = os.path.join(results_dir, "llm.log")
        llm = get_llm(llm_code)(temperature=temp, seed=seed)
        # Activate logging LLM
        logger.info(f"Piping LLM logs to {llm_log_path}")
        llm.log_to_file(llm_log_path)

    extractor = None
    if mode in ['cristal', 'both'] and llm_extractor_code:
        # Extractor LLM if needed
        extractor_log_path = os.path.join(results_dir, "llm-extractor.log")
        extractor = get_llm(llm_extractor_code)(temperature=temp, seed=seed)
        logger.info(f"Piping Extractor logs to {extractor_log_path}")
        extractor.log_to_file(extractor_log_path)

    # We can just compile asset names here
    # Only for assets that we have reports for
    truth_data = {}
    stocks = synth.load_stocks(as_dataset=False, test=True, train_size=args.train_size)
    # Swap the labels "high_growth" and "time_bomb" in the stocks
    stocks = swap_labels(stocks)
    financials = {}  # For use by LLM
    for asset_id in os.listdir(REPORTS_DIR):
        if asset_id not in stocks:
            logger.debug(f"Asset ID '{asset_id}' not found in stocks. Skipping.")
            continue

        # Check category
        category = stocks[asset_id]['category']

        # Check if we are in test mode
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

    #####################
    # Save details to config file
    #
    run_data = {
        "command": command_str,
        "description": description,
        "seed": seed,
        "test": testing,
        "base_llm": llm.to_json() if llm else None,
        "extractor_llm": extractor.to_json() if llm_extractor_code else None,
        "reports_dir": REPORTS_DIR,
        "budget_mode": str(budget_mode),
        "budget_per_task": budget_per_task,
        "total_budget": total_budget,
        "n_asset": n_asset,
        "n_asset_per": n_asset_per,
        "assets": {cat: len(reports) for cat, reports in assets.items()}
    }

    #####################
    # Configure CRISTAL
    #
    stocks = synth.load_stocks(test=True, train_size=args.train_size)  # synth.stocks_to_xarray(stocks)
    # we also need to swap labels here
    # Swap the labels "high_growth" and "time_bomb" in the stocks
     # Call the function to swap labels
    stocks = swap_labels(stocks)
    if mode in ['cristal', 'both']:
        # Create CRISTAL model -- Use Extractor LLM
        m = CRISTAL(
            stocks, extractor, asset_memory_depth=asset_memory_depth,
            budget_mode=budget_mode, budget=total_budget,
        )
        # pretrain cristal here with the train method and n_shots
        training_data = synth.get_ground_truth_samples(args.n_shots, args.train_size)
        #we also need to swap labels here
        training_data = swap_labels(training_data)
        logger.debug("predefined likelihood matrix: %s", m.indicators.likelihood.values)
        logger.debug("shape of predefined likelihood matrix: %s", m.indicators.likelihood.shape)
        m.likelihood_inference(training_data)
        logger.debug("Updated likelihood matrix: %s", m.indicators.likelihood.values)
        logger.debug("shape of updated likelihood matrix: %s", m.indicators.likelihood.shape)
        run_data['cristal'] = m.to_json()

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

    if mode in ["cristal", "both"]:
        #####################
        # CRISTAL portion
        cristal_results = {}

        with logging_redirect_tqdm(loggers=[logger]):
            for cat, a_ids in assets.items():
                logger.info(f"\n=============  CRISTAL  =============\nProcessing '{cat}' assets ({len(a_ids)})\n")
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
                        remaining_budget = budget_per_task

                        while remaining_budget > 0:
                            success, reason, remaining_budget = m.process_asset(
                                asset_id, max_budget=remaining_budget, pre_extracted=extracted_colors
                            )
                    except Exception as e:
                        if strict:
                            raise e
                        # Else log and continue
                        logger.warning(f"Exception processing asset {asset_id}: {e}")
                        cristal_results[asset_id] = {
                            "error": str(e),
                            "truth": truth_data[asset_id]
                        }
                    else:
                        if not success:
                            cristal_results[asset_id] = {
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
                        cristal_results[asset_id] = {
                            "growth": tuple([(q, p) for q, p in zip(m.quantiles, quantile_pred)]),
                            "category": {c: p for c, p in zip(['high_growth', 'stable', 'time_bomb'], cats)},
                            "truth": truth_data[asset_id]
                        }

        # Add in-place performance evaluation
        stats.evaluate_performance(cristal_results, mixture_params, in_place=True)

        # Save CRISTAL results
        cristal_results_path = os.path.join(results_dir, "cristal_results.json")
        with open(cristal_results_path, 'w') as f:
            logger.info(f"Saving CRISTAL results to {cristal_results_path}")
            json.dump(cristal_results, f, indent=4)

        # Save the cristal model
        m.save_model(
            model_name=f"cristal-{run_name}",
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
                    if "gemini" in llm.code:
                        time.sleep(3)

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
                    report_file = os.path.join(REPORTS_DIR, f"{asset_id}")
                    reports_text = read_reports(report_file)

                    # Compress the reports
                    if compress_reports:
                        # TODO -- Consider if this is within budget...For now just give it 10s
                        # Seems to work better with llama, and deepseek had a hard time
                        try:
                            summaries = summarize_document(llm, reports_text, summary_steps=3, max_retries=3, budget=10)
                        except Exception as e:
                            # Log and move on if not strict
                            if strict:
                                raise e
                            logger.warning(f"Failed to compress reports for asset {asset_id}: {e}")
                        else:
                            reports_text = summaries[-1]

                    # Modifies the prompt if indicators are provided, but they are not required
                    prompt = llm_utils.prompt_exp2_task(
                        reports_text, indicators, n_shots=args.n_shots, train_size=args.train_size
                    )
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

    from cristal.logger import logger, reset_logger, log_to_file, capture_description

    from cristal import synthetic as synth, stats
    from cristal.cristal import CRISTAL
    from cristal.utils import BudgetMode
    from cristal.env import PROJECT_ROOT, REPORTS_DIR as _REPORTS_DIR
    from cristal.llm import get_llm, utils as llm_utils
    from cristal.llm.wrangling import read_reports, summarize_document
    import pytensor

    pytensor.config.exception_verbosity = 'high'

    # Set seed
    # 101010 happens to be 42 in binary. How charming.
    SEED = 101010

    main()
