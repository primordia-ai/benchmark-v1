"""
Test the codegen validation workflow
"""

if __name__ == '__main__':
    from cristal.llm import get_llm
    from cristal.logger import logger
    from cristal.env import USE_LARGE_MODELS
    from cristal.llm.codegen import dirichlet_bayes as db

    SEED = 101010

    if USE_LARGE_MODELS:
        llm_cls = get_llm("deepseek-r1:70b")
    else:
        llm_cls = get_llm("deepseek-r1:7b")

    logger.info(f"Initializing LLM {llm_cls.code}")
    llm = llm_cls(temperature=0, seed=SEED)

    soln_str = db.solution_string
    logger.info("Validating solution-string for DirichletBayesFunction...")
    dbf = db.DirichletBayesFunction(code=soln_str)
    # Looks like validation worked
    logger.info("Validation successful.")
