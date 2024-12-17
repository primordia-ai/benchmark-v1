"""
Process the first asset with ALLA
"""

if __name__ == '__main__':
    import logging

    # Set this up before importing other classes
    logger = logging.getLogger("alla")
    logger.setLevel(logging.INFO)

    from alla import models
    from alla.utils import synthetic as synth

    # Generate 5 synthetic stocks
    n_stock = 5
    print(f"Generating {n_stock} synthetic stocks...")
    stocks = synth.generate_synthetic_stocks(n_stock)

    # Create ALLA model
    print("Initializing ALLA")
    alla = models.ALLA(stocks, asset_memory_depth=20, seed=1, budget=100)

    # Process the first stock
    print("Processing stock 'asset_0'")
    while alla.budget > 0:
        logger.info(f"Calling process_asset with remaining budget: {alla.budget}")
        success = alla.process_asset('asset_0', max_budget=20)
        if not success:
            # Insufficient budget to iterate
            break
    print("Done.")

    # Plot
    save_path = "../results/tmp/latest.png"
    alla.plot_history('asset_0', save_path=save_path)
