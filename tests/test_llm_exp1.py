"""
Test the report densification utilities in llm.wrangling
"""

if __name__ == '__main__':
    import json
    import os
    import numpy as np

    # Set seed
    SEED = 10
    np.random.seed(SEED)

    # =========

    from alla.logger import logger
    from alla.llm import get_llm, utils as llm_utils
    from alla.llm import wrangling
    from alla.utils import PROJECT_ROOT

    llm_cls = get_llm("llama-tiny")
    llm = llm_cls(temperature=0)

    # Load the report
    asset = "asset_0"
    report_dir = os.path.join(PROJECT_ROOT, f"data/reports/{asset}")
    report_txt = wrangling.read_reports(
        report_dir,
        # include=[
        #     "brand_strength_report.txt",
        #     "digital_transformation_report.txt",
        #     "geopolitical_risk_report.txt",
        # ]
    )

    # indicator text
    indicator_llm_code = "llama3_8b"
    indicator_file = os.path.join(PROJECT_ROOT, "data/extracted_indicators", indicator_llm_code, f"{asset}.json")
    with open(indicator_file, 'r') as f:
        indicators = json.load(f)

    prompt = llm_utils.prompt_exp1_task(report_txt, indicators)
    print(prompt)
    result = llm.evaluate_asset(prompt, max_retries=5)
    # NOTE: Llama-tiny truncates this prompt since it can't handle it...
    print(f"\nTEST RESULT:\n{result}")
