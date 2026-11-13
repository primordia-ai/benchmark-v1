"""
Test the CRISTAL codegen piping
"""
import os

import numpy as np
from pydantic import model_validator, ValidationError

from cristal.cristal import CRISTAL
from cristal.env import PROJECT_ROOT, USE_LARGE_MODELS
from cristal.llm import get_llm
from cristal.llm.codegen.validator import LLMCodeGenValidator, load_code
from cristal.synthetic import stocks_to_xarray


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
    from cristal.logger import logger, DEBUG, reset_logger

    reset_logger(DEBUG)

    if USE_LARGE_MODELS:
        llm_cls = get_llm("qwen-medium")
    else:
        llm_cls = get_llm("qwen-tiny")

    logger.info(f"Initializing LLM {llm_cls.code}")
    llm = llm_cls(temperature=0, seed=SEED)

    # Log to this filename
    llm_log = f"./tmp/test_codegen_cristal_logic.log"

    # Remove if exists
    if os.path.exists(llm_log):
        os.remove(llm_log)
    llm.log_to_file(llm_log)

    # Load CRISTAL
    stocks = np.load(
        f"{PROJECT_ROOT}/data/ground-truth/assets_200/stock_all_ground_truth_dict.pkl",
        allow_pickle=True
    )
    stocks = stocks_to_xarray(stocks, train_size=0)

    print("Initializing CRISTAL")
    m = CRISTAL(
        stocks, llm, asset_memory_depth=1, budget=30, use_codegen=True
    )

    # Add load our sort function into cristal
    logger.info("Loading SquareIntFunction into CRISTAL codegen")
    m.run_codegen_suite([SquareIntFunction])
    # Use the loaded method thru cristal pipe
    x = 20
    y = 400

    codegen_input = {
        "x": x
    }
    logger.info("Testing codegen pipe")
    result = m.codegen_pipe("square_int", codegen_input)
    logger.info("Evaluating result")
    assert result == y, f"Codegen result '{result}' does not match required output '{y}'"
    logger.info("Success")
