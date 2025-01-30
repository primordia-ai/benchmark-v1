# Usage: python generate_reports.py <llm_code>
# Output: LLM indicator color estimates for each soft indicator in the data

if __name__ == '__main__':
    import json
    import os
    import sys
    import numpy as np
    from tqdm import tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm
    import logging
    from alla.llm import get_llm
    from alla.llm.wrangling import read_reports
    from alla.utils import PROJECT_ROOT
    from alla.synthetic import SOFT_INDICATOR_COSTS, FINANCIAL_LIKELIHOOD_MATRIX, load_stocks
    from alla.logger import logger, reset_logger

    # Set seed
    SEED = 10
    np.random.seed(SEED)
    #TODO change the code so that we can choose the model that generated the reports and the one that extracts.
    # folder structure could be: data/extracted_indicators/{generator_code}/{extractor_code}/{asset_id}.json
    # Check for command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python generate_reports.py <generator_code> <extractor_code>\n Example: python generate_reports.py deepseek-r1:14b llama3.1:8b")
        sys.exit(1)

    generator_code = sys.argv[1]
    extractor_code = sys.argv[2]
    llm_cls = get_llm(extractor_code)
    llm = llm_cls(temperature=0)
    #cleanup the generator and extractor codes since : does not works as pathname and filename in windows
    generator_code_name = generator_code.replace(':', '_')
    extractor_code_name = extractor_code.replace(':', '_')
    # Update the extraction directory structure
    extract_dir = os.path.join(
        PROJECT_ROOT, f"data/extracted_indicators/{generator_code_name}/{extractor_code_name}"
    )
    # Create the directory if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)
    # =========

    soft_indicators = list(SOFT_INDICATOR_COSTS.keys())
    hard_indicator_names = list(FINANCIAL_LIKELIHOOD_MATRIX['high_growth'].keys())
    
    # Load the stocks array
    stocks = load_stocks(as_dataset=True)
    # Get all asset IDs from the stocks dataset
    asset_ids = stocks.asset.values
 
    #iterate over all assets
    with logging_redirect_tqdm(loggers=[logger]):
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

            print("writing to file"+extract_dir+asset+".json")
            write_path = os.path.join(extract_dir, f"{asset}.json")

            with open(write_path, 'w') as f:
                json.dump(ind_list, f, indent=4)
