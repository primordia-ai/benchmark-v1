"""
Test the main LLM prompt created for Exp 1
"""

if __name__ == '__main__':
    import json
    import os
    import numpy as np

    # Set seed
    SEED = 10
    np.random.seed(SEED)

    # =========

    from alla.llm import get_llm, utils as llm_utils
    from alla.llm import wrangling
    from alla.env import PROJECT_ROOT

    llm_cls = get_llm("llama-tiny")
    llm = llm_cls(temperature=0, seed=SEED)

    # Load the report
    asset = "asset_0"
    report_dir = os.path.join(PROJECT_ROOT, f"tests/data/reports/{asset}")
    report_txt = wrangling.read_reports(
        report_dir,
        # include=[
        #     "brand_strength_report.txt",
        #     "digital_transformation_report.txt",
        #     "geopolitical_risk_report.txt",
        # ]
    )

    # indicator text
    indicator_file = os.path.join(PROJECT_ROOT, "tests/data/extracted_indicators", f"{asset}.json")
    with open(indicator_file, 'r') as f:
        indicators = json.load(f)

    prompt = llm_utils.prompt_exp1_task(report_txt, indicators)
    print(prompt)
    result, success = llm.evaluate_asset(prompt, max_retries=5)
    # NOTE: Llama-tiny truncates this prompt since it can't handle it...
    print(f"\nTEST RESULT:\n{result}")
