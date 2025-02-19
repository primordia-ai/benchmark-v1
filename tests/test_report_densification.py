"""
Test the report densification utilities in llm.wrangling
"""

if __name__ == '__main__':
    import os
    import numpy as np

    # Set seed
    SEED = 111
    np.random.seed(SEED)

    # =========

    from alla.llm import get_llm
    from alla.llm import wrangling
    from alla.env import PROJECT_ROOT, SYSTEM_USER

    if SYSTEM_USER == "dsfhswf":
        llm_cls = get_llm("llama-large")
        includes = None
    else:
        llm_cls = get_llm("deepseek-r1:7b")
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
        include=includes,
    )
    print(report_txt)
    print("\n===============\n\nSUMMARIZING\n\n")

    summaries = wrangling.summarize_document(llm, report_txt, summary_steps=4, max_retries=5)
    for i, summ in enumerate(summaries):
        print(f"<><><>< Summary {i} ><><><>\n")
        print(summ)
