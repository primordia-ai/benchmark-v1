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

    from alla.llm import get_llm, utils as llm_utils
    from alla.llm import wrangling
    from alla.env import PROJECT_ROOT

    llm_cls = get_llm("llama-tiny")
    llm = llm_cls(temperature=0)

    # Load the report
    asset = "asset_0"
    report_dir = os.path.join(PROJECT_ROOT, f"tests/data/reports/{asset}")
    report_txt = wrangling.read_reports(
        report_dir,
        include=[
            "brand_strength_report.txt",
            "digital_transformation_report.txt",
            "geopolitical_risk_report.txt",
        ]
    )

    print(f"MW REPORT:\n{report_txt}")
    print("\n*************\n\n")

    condensed = wrangling.summarize_document(llm, report_txt, max_retries=3, summary_steps=3)

    print(f"MW CONDENSED:\n{condensed}")
