# Usage: python generate_reports.py <llm_code>
# Output: LLM indicator color estimates for each soft indicator in the data

if __name__ == '__main__':
    import json
    import os
    import sys
    import numpy as np
    from tqdm import tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm

    # Set seed
    SEED = 10
    np.random.seed(SEED)

    # Check for command-line argument
    if len(sys.argv) != 2:
        print("Usage: python generate_reports.py <llm_code>")
        sys.exit(1)

    llm_code = sys.argv[1]

    # =========

    from alla.llm import get_llm
    from alla.llm.wrangling import read_reports
    from alla.utils import PROJECT_ROOT
    from alla.synthetic import SOFT_INDICATOR_COSTS, FINANCIAL_LIKELIHOOD_MATRIX, load_stocks

    soft_indicators = list(SOFT_INDICATOR_COSTS.keys())
    hard_indicator_names = list(FINANCIAL_LIKELIHOOD_MATRIX['high_growth'].keys())
    asset_ids = ['asset_1', ]  # , 'asset_25', 'asset_35', 'asset_45']
    test_list = {}

    llm_cls = get_llm(llm_code)
    llm = llm_cls(temperature=0)

    # Create the extraction directory if needed
    extract_dir = os.path.join(
        PROJECT_ROOT, f"data/extracted_indicators/{llm.path_code}"
    )
    # Create the directory if it doesn't exist
    # os.makedirs(extract_dir, exist_ok=True)

    # Load the stocks array
    stocks = load_stocks(as_dataset=True)

    with logging_redirect_tqdm():
        for asset in tqdm(asset_ids, desc="Asset"):
            ind_list = []
            stock = stocks.sel(asset=asset)
            indicators = stock.indicator.values
            colors = stock.ryg.values
            numeric_values = stock.val.values
            # First, populate with independent (hard) indicators
            for ind, color, numeric in zip(indicators, colors, numeric_values):
                if ind in hard_indicator_names:
                    ind_list.append(
                        {
                            "indicator": ind,
                            "value": color,
                            "numeric": numeric,
                            "reason": "Synthetic source",
                        }
                    )

            # Read all the reports for this asset
            reports_dir = os.path.join(PROJECT_ROOT, f"data/reports/{asset}")

            for ind in tqdm(soft_indicators, leave=False, desc="Indicator"):
                # TODO -- Consider if we toss in the entire set of reports of just the singles
                # On local machine, tiny llama truncated the max report, and even then it still chugged...
                reports_text = read_reports(reports_dir, include=[f"{ind}_report.txt"])

                extracted = llm.extract_indicator(
                    indicator_name=ind, asset_id=asset, reports_text=reports_text, verbose=True
                )

                ind_list.append(extracted)

            write_path = os.path.join(extract_dir, f"{asset}.json")

            with open(write_path, 'w') as f:
                json.dump(ind_list, f, indent=4)
