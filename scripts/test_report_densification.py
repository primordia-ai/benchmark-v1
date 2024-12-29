"""
Test the report densification utilities in llm.wrangling
"""

if __name__ == '__main__':
    import logging
    import json
    import os
    import sys
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
    from alla.llm.utils import
    from alla.utils import PROJECT_ROOT
    from alla.synthetic import SOFT_INDICATOR_COSTS

    llm_cls = get_llm("llama-tiny")
    llm = llm_cls(temperature=0)

    # Load the report
    asset = "asset_0"
    report_dir = os.path.join(PROJECT_ROOT, "data/reports")
