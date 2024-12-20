"""
Process the first asset with ALLA
"""

if __name__ == '__main__':
    import logging

    # Set this up before importing other classes
    logger = logging.getLogger("alla")
    logger.setLevel(logging.INFO)

    import numpy as np

    # Set seed
    SEED = 10
    np.random.seed(SEED)

    from alla import alla, synthetic as synth

    # Generate synthetic stocks
    n_stock = 111
    print(f"Generating {n_stock} synthetic stocks...")
    stocks = synth.generate_synthetic_stocks(n_stock)

    # Select the first stock that is of this category
    category_select = "high_growth"
    asset_name = None
    for asset in stocks.asset:
        if asset['category'] == category_select:
            asset_name = str(asset['asset'].values)
            break
    if asset_name is None:
        raise RuntimeError(f"No {category_select} asset generated")

    # Create ALLA model
    print("Initializing ALLA")
    alla = models.ALLA(stocks, asset_memory_depth=20, seed=SEED, budget=100)

    # Process the first stock
    print(f"Processing '{category_select}' stock '{asset_name}'")
    while alla.budget > 0:
        logger.info(f"Calling process_asset with remaining budget: {alla.budget}")
        success = alla.process_asset(asset_name, max_budget=None)
        if not success:
            # Insufficient budget to iterate
            break
    print("Done.")

    # Plot
    save_path = "../results/tmp/latest.png"
    alla.plot_history(asset_name, save_path=save_path)
