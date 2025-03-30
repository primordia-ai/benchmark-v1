"""
Test an LLM actually trying to generate codegen code.
"""
import os

from cristal.env import SYSTEM_USER
from cristal.llm import get_llm
from cristal.llm.codegen.dirichlet_bayes import DirichletBayesFunction
from cristal.llm.codegen.model import LLMCodeGen
from cristal.llm.codegen.validator import LLMCodeGenException
from cristal.logger import logger

if __name__ == '__main__':
    SEED = 101010
    from cristal.logger import DEBUG, reset_logger

    reset_logger(DEBUG)

    if SYSTEM_USER == "dsfhswf":
        llm_cls = get_llm("qwen-large")
    else:
        llm_cls = get_llm("qwen-tiny")

    logger.info(f"Initializing LLM {llm_cls.code}")
    llm = llm_cls(temperature=0, seed=SEED)

    llm_log = "./tmp/test_codegen_with_llm_dirichlet.log"
    # Remove if exists
    if os.path.exists(llm_log):
        os.remove(llm_log)
    llm.log_to_file(llm_log)

    codegen_llm = LLMCodeGen(llm)
    max_retries, budget = 5, 60
    try:
        code = codegen_llm.generate_function(DirichletBayesFunction, max_retries, budget)
    except LLMCodeGenException as e:
        logger.error(f"Failed to generate code: {e}")
        raise e

    # Looks like validation worked
    logger.info("Validation successful.")
    logger.info(f"Generated code:\n{code}")
