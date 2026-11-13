"""
Test an LLM actually trying to generate codegen code.
"""
import os

from pydantic import model_validator, ValidationError

from cristal.env import USE_LARGE_MODELS
from cristal.llm import get_llm
from cristal.llm.codegen.model import LLMCodeGen
from cristal.llm.codegen.validator import LLMCodeGenException, LLMCodeGenValidator, load_code
from cristal.logger import logger


class SquareIntFunction(LLMCodeGenValidator):
    """
    Defines a function for squaring an integer.
    Input arguments include:
        1. An integer to square.
    Output:
        1. The square of the integer.
    """
    is_function = True
    function_name = "square_int"
    prompt_description = "Square an integer."
    required_args = tuple(
        [
            {
                "arg": "x",
                "description": "Integer to square",
                "datatype": int
            },
        ]
    )
    returned_data = tuple(
        [
            {
                "name": "Square",
                "description": "The square of the integer",
                "type": int
            },
        ]
    )

    @model_validator(mode="after")
    def test_output(self):
        func = load_code(self.code, self.function_name)
        inputs = [134, 234, 214, 99, 999]
        answers = [x ** 2 for x in inputs]
        for i, x in enumerate(inputs):
            if func(x) != answers[i]:
                return ValidationError(f"Function failed to square {x}.")
        # Probably okay lol
        return self


if __name__ == '__main__':
    SEED = 101010
    from cristal.logger import DEBUG, reset_logger

    reset_logger(DEBUG)

    if USE_LARGE_MODELS:
        llm_cls = get_llm("qwen-medium")
    else:
        llm_cls = get_llm("qwen-tiny")

    logger.info(f"Initializing LLM {llm_cls.code}")
    llm = llm_cls(temperature=0, seed=SEED)

    llm_log = "./tmp/test_codegen_with_llm.log"
    # Remove if exists
    if os.path.exists(llm_log):
        os.remove(llm_log)
    llm.log_to_file(llm_log)

    codegen_llm = LLMCodeGen(llm)
    max_retries, budget = 5, 60

    try:
        code = codegen_llm.generate_function(SquareIntFunction, max_retries, budget)
    except LLMCodeGenException as e:
        logger.error(f"Failed to generate code: {e}")
        raise e

    # Looks like validation worked
    logger.info("Validation successful.")
    logger.info(f"Generated code:\n{code}")

    # One additional test the product
    inputs = [123, 234, 555, 76453]
    answers = [x ** 2 for x in inputs]
    for x, y in zip(inputs, answers):
        output = code(x)
        logger.info(f"Testing: input = {x}, answer = {y} | code(x) = {output}")
        assert output == y, f"Failed to square {x} with generated code."
