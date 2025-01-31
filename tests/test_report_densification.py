"""
Test the report densification utilities in llm.wrangling
"""

if __name__ == '__main__':
    import os
    import numpy as np

    # Set seed
    SEED = 10
    np.random.seed(SEED)

    # =========

    from alla.llm import get_llm
    from alla.llm import wrangling
    from alla.utils import PROJECT_ROOT

    llm_cls = get_llm("llama-tiny")
    llm = llm_cls(temperature=0)

    # Load the report
    asset = "asset_0"
    report_dir = os.path.join(PROJECT_ROOT, f"data/reports/{llm.path_code}/{asset}")
    report_txt = wrangling.read_reports(
        report_dir,
        include=[
            "brand_strength_report.txt",
            "digital_transformation_report.txt",
            "geopolitical_risk_report.txt",
        ]
    )
    print(report_txt)
    print("\n===============\n\nSUMMARIZING\n\n")

    summaries = wrangling.summarize_document(llm, report_txt, summary_steps=3, max_retries=2)
    for i, summ in enumerate(summaries):
        # print(f"<><><>< Summary {i} ><><><>\n")
        print(summ)
