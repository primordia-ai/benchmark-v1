# call a local llm and create reports for each soft indicator in the data
# Usage: python generate_reports.py
# Output: reports for each soft indicator in the data

if __name__ == '__main__':
    import logging
    import json
    import os
    import numpy as np
    from tqdm import tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm

    # Set this up before importing other classes
    logger = logging.getLogger("alla")
    logger.setLevel(logging.INFO)

    # Set seed
    SEED = 10
    np.random.seed(SEED)

    # =========

    from alla.llm import get_llm
    from alla.utils import PROJECT_ROOT
    from alla.synthetic import SOFT_INDICATOR_COSTS

    soft_indicators = list(SOFT_INDICATOR_COSTS.keys())
    asset_ids = ['asset_1', ]  # , 'asset_25', 'asset_35', 'asset_45']
    test_list = {}

    llm_cls = get_llm("llama-tiny")
    llm = llm_cls(temperature=0)

    # Create the extraction directory if needed
    extract_dir = os.path.join(
        PROJECT_ROOT, "data/extracted_indicators/", llm.code
    )
    # Create the directory if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)

    with logging_redirect_tqdm():
        for asset in tqdm(asset_ids, desc="Asset"):
            ind_list = []
            # Read all the reports for this asset
            reports_dir = os.path.join(PROJECT_ROOT, f"data/reports/{asset}")

            for ind in tqdm(soft_indicators, leave=False, desc="Indicator"):
                # TODO -- Consider if we toss in the entire set of reports of just the singles
                # On local machine, tiny llama truncated the max report, and even then it still chugged...
                reports_text = llm.read_reports(reports_dir, include=[f"{ind}_report.txt"])

                extracted = llm.extract_indicator(
                    indicator_name=ind, asset_id=asset, reports_text=reports_text, verbose=True
                )

                ind_list.append(extracted)

            write_path = os.path.join(extract_dir, f"{asset}.json")

            with open(write_path, 'w') as f:
                json.dump(ind_list, f, indent=4)
