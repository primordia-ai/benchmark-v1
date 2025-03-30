"""
Test an LLM actually trying to generate codegen code.
"""
import os

from pydantic import model_validator, ValidationError

from cristal.env import SYSTEM_USER
from cristal.llm import get_llm
from cristal.llm.codegen.model import LLMCodeGen
from cristal.llm.codegen.validator import LLMCodeGenException, LLMCodeGenValidator, load_code
from cristal.logger import logger


class StrangeSortFunction(LLMCodeGenValidator):
    """
    Given list of integers, return list in strange order.
    Strange sorting, is when you start with the minimum value,
    then maximum of the remaining integers, then minimum and so on.
    """
    is_function = True
    function_name = "strange_sort"
    prompt_description = """
    Given list of integers, return the list in strange order.
    Strange sorting, is when you start with the minimum value,
    then maximum of the remaining integers, then minimum and so on.
    """

    required_args = tuple(
        [
            {
                "arg": "x",
                "description": "List of integers",
                "datatype": list
            },
        ]
    )
    returned_data = tuple(
        [
            {
                "name": "sorted_x",
                "description": "The strange-sorted list",
                "type": list
            },
        ]
    )

    @model_validator(mode="after")
    def test_array_shapes(self):
        func = load_code(self.code, self.function_name)
        x = [5,2,7,8,45,1,3]
        y = [1, 45, 2, 8, 3, 7, 5]
        if func(x) != y:
            return ValidationError(f"Strange-sort function failed on input {x}.")

        return self


if __name__ == '__main__':
    SEED = 101010
    from cristal.logger import DEBUG, reset_logger

    reset_logger(DEBUG)

    if SYSTEM_USER == "dsfhswf":
        llm_cls = get_llm("qwen-medium")
    else:
        llm_cls = get_llm("qwen-tiny")

    logger.info(f"Initializing LLM {llm_cls.code}")
    llm = llm_cls(temperature=0, seed=SEED)

    # Log to this filename
    llm_log = f"./tmp/test_codegen_with_llm_strangesort.log"

    # Remove if exists
    if os.path.exists(llm_log):
        os.remove(llm_log)
    llm.log_to_file(llm_log)

    codegen_llm = LLMCodeGen(llm)
    max_retries, budget = 5, 60

    try:
        code = codegen_llm.generate_function(StrangeSortFunction, max_retries, budget)
    except LLMCodeGenException as e:
        logger.error(f"Failed to generate code: {e}")
        raise e

    # Looks like validation worked
    logger.info("Validation successful.")
    logger.info(f"Generated code:\n{code}")
