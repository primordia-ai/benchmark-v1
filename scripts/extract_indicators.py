# call a local llm and create reports for each soft indicator in the data
# Usage: python generate_reports.py
# Output: reports for each soft indicator in the data

if __name__ == '__main__':
    import logging
    import json
    import os
    import numpy as np
    from tqdm import tqdm

    # Set this up before importing other classes
    logger = logging.getLogger("alla")
    logger.setLevel(logging.INFO)

    # Set seed
    SEED = 10
    np.random.seed(SEED)

    # =========

    from alla.llm import Llama
    from alla.utils import PROJECT_ROOT

    ind = 'innovation_pipeline'  # 'geopolitical_risk'#'mismanagement'
    soft_indicators = ['disruptor_stagnant', 'incumbent_stagnant', 'turnaround_plan', 'succession_risk',
        'mismanagement', 'geopolitical_risk', 'environmental_risk', 'product_moat', 'supply_chain_resilience',
        'regulatory_pressure', 'innovation_pipeline', 'customer_concentration', 'brand_strength',
        'digital_transformation']
    asset_ids = ['asset_0', 'asset_15']  # , 'asset_25', 'asset_35', 'asset_45']
    test_list = {}

    llm = Llama(temperature=0)

    for asset in tqdm(asset_ids, desc="Asset"):
        ind_list = []
        for ind in tqdm(soft_indicators, leave=False, desc="Indicator"):
            # Read all the reports for this asset
            os.path.join(PROJECT_ROOT, f"data/reports/{asset}")
            indicator = llm.extract_indicator(
                indicator_name=ind, asset_id=asset,
            )
            # print(indicator)
            ind_list.append(indicator)
        test_list[asset] = ind_list

    with open('../tests/test_indicator_results.json', 'w') as f:
        json.dump(test_list, f, indent=4)
