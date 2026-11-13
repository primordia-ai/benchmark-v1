"""
Test the Codegen Critic
"""
import os
import time
from pprint import pprint

from pydantic import model_validator, ValidationError

from cristal.env import USE_LARGE_MODELS
from cristal.llm import get_llm
from cristal.llm.codegen.critic import CodeGenCritic
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
        actor_cls = get_llm("mistral-nemo")
        critic_cls = get_llm("llama-large")
    else:
        actor_cls = get_llm("llama-small")
        critic_cls = get_llm("llama-small")

    logger.info(f"Initializing Actor model {actor_cls.code} | Critic model {critic_cls.code}")
    actor_llm = actor_cls(temperature=0, seed=SEED)
    critic_llm = critic_cls(temperature=0, seed=SEED)

    actor_log = "./tmp/test_codegen_critic_actor_model.log"
    # Remove if exists
    if os.path.exists(actor_log):
        os.remove(actor_log)
    actor_llm.log_to_file(actor_log)

    critic_log = "./tmp/test_codegen_critic_critic_model.log"
    # Remove if exists
    if os.path.exists(critic_log):
        os.remove(critic_log)
    critic_llm.log_to_file(critic_log)

    # Describe the Square-Int task
    validator = SquareIntFunction
    task = "We need to create a simple utility that can return the square of an integer."

    # Initialize Codegen critic first to supply to actor
    critic = CodeGenCritic(critic_llm, task)
    actor = LLMCodeGen(actor_llm, critic=critic)

    # Set retries and budget
    actor_retries, actor_budget = 3, 20
    loop_max_iters, loop_budget = 4, 120

    # Call the loop method
    result = actor.critic_loop(
        validator, loop_max_iters, loop_budget, actor_retries, actor_budget
    )
    func = load_code(result['code'], validator.function_name)

    # Test the returned function
    assert func(4) == 16, "Insane fail"
