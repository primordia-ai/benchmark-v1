"""
Test the codegen validation workflow
"""

if __name__ == '__main__':
    from alla.llm import get_llm
    from alla.logger import logger
    from alla.env import SYSTEM_USER
    from alla.llm.codegen import dirichlet_bayes as db

    SEED = 101010

    if SYSTEM_USER == "dsfhswf":
        llm_cls = get_llm("deepseek-r1:70b")
        includes = [
            "brand_strength_report.txt",
            "digital_transformation_report.txt",
            "geopolitical_risk_report.txt",
        ]
    else:
        llm_cls = get_llm("deepseek-r1:7b")
        includes = [
            "brand_strength_report.txt",
            "digital_transformation_report.txt",
            "geopolitical_risk_report.txt",
        ]

    logger.info(f"Initializing LLM {llm_cls.code}")
    llm = llm_cls(temperature=0, seed=SEED)

    soln_str = db.solution_string
    logger.info("Validating solution-string for DirichletBayesFunction...")
    dbf = db.DirichletBayesFunction(code=soln_str)
    # Looks like validation worked
    logger.info("Validation successful.")
