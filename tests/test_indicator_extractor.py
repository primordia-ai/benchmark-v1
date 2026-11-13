"""
Process the first asset with CRISTAL
"""
if __name__ == '__main__':
    from cristal import synthetic as synth
    from cristal.cristal import CRISTAL

    # Generate 5 synthetic stocks
    n_stock = 5
    print(f"Generating {n_stock} synthetic stocks...")
    stocks = synth.generate_synthetic_stocks(n_stock)

    # Create CRISTAL model
    print("Initializing CRISTAL")
    cristal = CRISTAL(stocks, asset_memory_depth=20)

    # Process the first stock
    print("Processing stock 'asset_0'")
    cristal.process_asset_with_budget('asset_0', budget=20, soft_only=True)
    print("Done.")

    # Plot
    save_path = "../results/tmp/latest.png"
    cristal.plot_prior_history('asset_0', save_path=save_path)
