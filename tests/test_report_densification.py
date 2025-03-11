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

    from cristal.logger import DEBUG, reset_logger
    reset_logger(DEBUG)

    from cristal.llm import get_llm
    from cristal.llm import wrangling
    from cristal.env import PROJECT_ROOT, SYSTEM_USER

    if SYSTEM_USER == "dsfhswf":
        llm_cls = get_llm("llama-large")
    else:
        llm_cls = get_llm("llama-small")

    llm = llm_cls(temperature=0, seed=SEED)
    # includes=[
    #     "brand_strength_report.txt",
    #     "digital_transformation_report.txt",
    #     "geopolitical_risk_report.txt",
    # ]
    includes = None

    llm_log = "./tmp/test_report_densitification.log"
    # Remove if exists
    if os.path.exists(llm_log):
        os.remove(llm_log)
    llm.log_to_file(llm_log)

    # Load the report
    asset = "asset_0"
    report_dir = os.path.join(PROJECT_ROOT, f"tests/data/reports/{asset}")
    report_txt = wrangling.read_reports(
        report_dir,
        include=includes,
    )
    print(report_txt)
    print("\n===============\n\nSUMMARIZING\n\n")

    # In debug mode, the summaries are printed out already
    summaries = wrangling.summarize_document(llm, report_txt, summary_steps=4, max_retries=3, budget=60)
