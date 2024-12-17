"""
Process the first asset with ALLA
"""
from alla import models
from alla.utils import synthetic as synth


if __name__ == '__main__':
    # Generate 5 synthetic stocks
    n_stock = 5
    print(f"Generating {n_stock} synthetic stocks...")
    stocks = synth.generate_synthetic_stocks(n_stock)

    # Create ALLA model
    print("Initializing ALLA")
    alla = models.ALLA(stocks, asset_memory_depth=20, seed=1)

    # Process the first stock
    print("Processing stock 'asset_0'")
    alla.process_asset('asset_0', budget=20)
    print("Done.")

    # Plot
    save_path = "../results/tmp/latest.png"
    alla.plot_prior_history('asset_0', save_path=save_path)
