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
    from alla.env import PROJECT_ROOT
    from alla.logger import logger
    from alla.llm import get_llm, utils as llm_utils

    stocks = synth.load_stocks(as_dataset=True)
    # Create ALLA model
    print("Initializing ALLA")
    # Initialize our LLM
    llm = get_llm('llama-tiny')(temperature=0, seed=SEED)
    # Create ALLA model
    m = ALLA(
        stocks, llm, asset_memory_depth=5, budget=30,
    )

    asset_name = "asset_0"
    extract_path = os.path.join(
        PROJECT_ROOT, f"tests/data/extracted_indicators/{asset_name}.json"
    )
    # Result is list of {"value": .., "indicator": ..} items
    extracted_colors = llm_utils.load_extracted_indicators(extract_path, keys=["value", "indicator"])
    # Convert to dict of values
    extracted_colors = {item['indicator']: item['value'] for item in extracted_colors}

    # Process the first stock
    # print(f"Processing '{category_select}' stock '{asset_name}'")
    logger.info(f"Calling process_asset with remaining budget: {m.budget}")
    success, reason = m.process_asset(asset_name, max_budget=None, pre_extracted=extracted_colors)
    if not success:
        raise Exception(f"Failed processing asset: {reason}")

    print("Done.")
