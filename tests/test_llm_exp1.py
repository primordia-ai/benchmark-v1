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

    from alla.logger import DEBUG, reset_logger
    reset_logger(DEBUG)

    from alla.llm import get_llm, utils as llm_utils
    from alla.llm import wrangling
    from alla.env import PROJECT_ROOT, SYSTEM_USER

    if SYSTEM_USER == "dsfhswf":
        llm_cls = get_llm("deepseek-small")
        includes=[
            "brand_strength_report.txt",
            "digital_transformation_report.txt",
            "geopolitical_risk_report.txt",
        ]
    else:
        llm_cls = get_llm("llama-small")
        includes=[
            "brand_strength_report.txt",
            "digital_transformation_report.txt",
            "geopolitical_risk_report.txt",
        ]

    llm = llm_cls(temperature=0, seed=SEED)

    # Load the report
    asset = "asset_0"
    report_dir = os.path.join(PROJECT_ROOT, f"tests/data/reports/{asset}")
    report_txt = wrangling.read_reports(
        report_dir,
        include=includes
    )
    max_retries = 3
    budget = 30

    # indicator text
    indicator_file = os.path.join(PROJECT_ROOT, "tests/data/extracted_indicators", f"{asset}.json")
    with open(indicator_file, 'r') as f:
        indicators = json.load(f)

    prompt = llm_utils.prompt_exp1_task(report_txt, indicators)
    print(prompt)
    result, success = llm.evaluate_asset(prompt, max_retries=max_retries, budget=budget)
    # NOTE: Llama-tiny truncates this prompt since it can't handle it...
    print(f"\nTEST RESULT:\n{result}")
