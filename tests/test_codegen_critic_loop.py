"""
Test the Codegen Critic
"""
import os
import time
from pprint import pprint

from pydantic import model_validator, ValidationError

from cristal.env import SYSTEM_USER
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

    if SYSTEM_USER == "dsfhswf":
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

    # This is similar ot code we would find in the critic loop
    prompt = actor._get_function_gen_prompt(validator)
    directive = critic.initial_directive_prompt()

    prompt += f"""
    ---
    To aid in thinking about the solution, consider the following directive.

    <DIRECTIVE>
    {directive}
    </DIRECTIVE>
    """

    logger.debug(
        f"INITIAL PROMPT FOR THE CODEGEN ATTEMPT, INCLUDING DIRECTIVE:\n\n{prompt}"
    )

    # Call the function gen, but don't load the code yet
    attempt: str = actor.generate_function(
        validator, max_retries=actor_retries, budget=actor_budget,
        prompt_override=prompt, load=False,
    )

    # First attempt
    logger.debug(
        f"FIRST ATTEMPT:\n{attempt}"
    )

    # Add this to the critic history
    # critic.history.append(
    #     {
    #         "code": attempt,
    #         "evaluation": directive,
    #         "approved": False
    #     }
    # )

    # Main loop
    approved = False
    start = time.time()
    i = 0
    max_critic_eval_retries = 3  # After three, it's probably not returning JSON correct.
    approved_solution = None
    while not approved:
        i += 1
        # We'll use the default evaluate retry params
        remaining_budget = loop_budget - (time.time() - start)
        if remaining_budget < 0:
            raise LLMCodeGenException(
                f"Critic loop budget exhausted without approved function. Latest attempt:\n\n{attempt}"
            )

        eval_item = critic.evaluate_function(
            attempt, budget=remaining_budget, max_retries=max_critic_eval_retries
        )
        approved = eval_item['approved']
        print("CRITIC EVAL ITEM:")
        pprint(eval_item)

        if approved:
            # Great
            logger.info(f"Codegen function approved by critic after {i} loop(s)")
            approved_solution = attempt
            break

        # Check if exhausted
        if i >= loop_max_iters:
            raise LLMCodeGenException(
                f"Max iterations ({loop_max_iters}) of critic loop reached without approved function. Latest "
                f"attempt:\n\n{attempt}"
            )

        if remaining_budget < 0:
            raise LLMCodeGenException(
                f"Critic loop budget exhausted without approved function. Latest attempt:\n\n{attempt}"
            )

        # Try another attempt
        prompt = actor._get_gen_prompt_with_eval(validator, attempt, eval_item['evaluation'])
        remaining_budget = loop_budget - (time.time() - start)
        attempt_budget = min(remaining_budget, actor_budget)
        attempt: str = actor.generate_function(
            validator, max_retries=actor_retries, budget=remaining_budget,
            prompt_override=prompt, load=False
        )

    if not approved:
        raise LLMCodeGenException("Failed to produce satisfiable codegen solution within resources")
