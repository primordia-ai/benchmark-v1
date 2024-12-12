"""
Generate and display info of synthetic stocks
"""
if __name__ == '__main__':
    from alla.utils import synthetic as synth

    # Generate 5 synthetic stocks
    stocks = synth.generate_synthetic_stocks(5)

    # Print summary for each stock
    for i, stock in enumerate(stocks, 1):
        print(f"\n{'='*50}")
        print(f"Stock {i}")
        synth.print_stock_summary(stock)

        # Print any verification warnings
        quality_score = stock['financials']['verification']['quality_score']
        if quality_score < 0.8:
            print("\nWarning: Low quality score, check verification metrics.")


    # As xarray
    ds = synth.stocks_to_xarray(stocks)

    # Print summary
    print(f"\n{'-'*50}")
    print("\nAs xr.Dataset\n")
    print(ds)

    print("\nSample data for first stock:")
    print("\nIndicator values:")
    print(ds.val.sel(asset='asset_0'))
    print("\nIndicator colors:")
    print(ds.ryg.sel(asset='asset_0'))
