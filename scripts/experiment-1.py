"""
Here we compare the performance of hard-coded ALLA vs LLM estimation of asset growth and category
on our synthetic dataset of assets.

This uses the LLM-generated synthetic reports found in data/reports,
and the LLM-produced Soft-Indicator Extractor indicator color results that were based off these reports.

In this first experiment we provide an LLM (we'll use llama) with the synthetic reports, the soft-indicator
extraction results, (and growth range?) and ask it to estimate next year's growth and asset category.

With ALLA, we will used our hardcoded version that performs inference on the soft-indicator extracted
indicators, and the logic around optimized selection given resources, etc. See ALLA.process_asset.

TODO -- Questions
    1. Is "Actual" in our setup the underlying mixture distribution parameters (mu_g) or the sampled 2025_growth?
    2. How to get unbiased growth estimates from llama if we are giving it the three mixture means?
        a. Conversely, without the three mixture means, we will have to describe the range of growth values somehow.
    3. Need the rest of the reports for other assets.
"""

if __name__ == '__main__':
    import logging

    import os
    # Set this up before importing other classes
    logger = logging.getLogger("alla")
    logger.setLevel(logging.INFO)

    import numpy as np
    from tqdm import tqdm

    # Set seed
    SEED = 10
    np.random.seed(SEED)

    from alla import synthetic as synth, utils
    from alla.alla import ALLA

    stocks = synth.load_stocks(as_dataset=True)
    # Create ALLA model
    custom_name = "alla-exp1"
    budget_per_task = 20
    total_budget = 1e8  # arbitrarily large
    m = ALLA(stocks, asset_memory_depth=20, budget=total_budget)
    # 10 assets per category
    categories = m.classes.keys()
    n_asset = 9
    n_asset_per = n_asset / 3
    assets = {
        "high_growth": [],
        "stable": [],
        "time_bomb": [],
    }

    # We can just compile asset names here
    # Only for assets that we have reports for
    report_dir = os.path.join(utils.PROJECT_ROOT, "data/reports")
    for asset_id in os.listdir(report_dir):
        # Check category
        try:
            asset = m.stocks.sel(asset=asset_id)
        except KeyError as e:
            print(f"Asset {asset_id} not found in stocks. Continuing.")
            continue

        category = asset.category.values.item()
        if len(assets[category]) < n_asset_per:
            assets[category].append(asset_id)
        else:
            continue

    for cat, reports in assets.items():
        print(f"{len(reports)} reports found for '{cat}':\n{reports}")

    for cat, reports in assets.items():
        print(f"Processing '{cat}' assets")
        for asset_id in tqdm(reports, desc="Assets"):
            try:
                success = m.process_asset(asset_id, max_budget=budget_per_task)
            except Exception as e:
                print(f"Error processing asset {asset_id}: {e}")

    # Save the alla model
    m.save_model(custom_name)
