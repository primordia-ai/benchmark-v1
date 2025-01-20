"""
Process the first asset with ALLA
"""

if __name__ == '__main__':

    import numpy as np
    import os

    # Set seed
    SEED = 10
    np.random.seed(SEED)

    from alla import synthetic as synth
    from alla.alla import ALLA
    from alla.utils import PROJECT_ROOT
    from alla.logger import logger
    from alla.llm import get_llm, utils as llm_utils

    # Generate synthetic stocks
    # n_stock = 111
    # print(f"Generating {n_stock} synthetic stocks...")
    # stocks = synth.generate_synthetic_stocks(n_stock)

    # Select the first stock that is of this category
    # asset_name = None
    # category_select = "high_growth"
    # for asset in stocks.asset:
    #     if asset['category'] == category_select:
    #         asset_name = str(asset['asset'].values)
    #         break
    #
    # if asset_name is None:
    #     raise RuntimeError(f"No {category_select} asset generated")

    stocks = synth.load_stocks(as_dataset=True)

    asset_name = "asset_0"
    # Create ALLA model
    print("Initializing ALLA")
    # Initialize our LLM
    llm = get_llm('llama-tiny')(temperature=0)

    # Create ALLA model
    m = ALLA(
        stocks, llm, asset_memory_depth=20, budget=8,
    )

    extract_path = os.path.join(
        PROJECT_ROOT, f"data/extracted_indicators/llama3_8b/{asset_name}.json"
    )
    # Result is list of {"value": .., "indicator": ..} items
    extracted_colors = llm_utils.load_extracted_indicators(extract_path, keys=["value", "indicator"])
    # Convert to dict of values
    extracted_colors = {item['indicator']: item['value'] for item in extracted_colors}

    # Process the first stock
    # print(f"Processing '{category_select}' stock '{asset_name}'")
    while m.budget > 0:
        logger.info(f"Calling process_asset with remaining budget: {m.budget}")
        success = m.process_asset(asset_name, max_budget=None, pre_extracted=extracted_colors)
        if not success:
            # Insufficient budget to iterate
            break
    print("Done.")

    # Plot
    save_path = "../results/tmp/latest.png"
    m.plot_history(asset_name, save_path=save_path)
