"""
LLMCodeGen model
"""
import time
from typing import Callable, Type
import json

from instructor.exceptions import InstructorRetryException
from instructor.utils import ChatCompletion

from cristal.llm.base import LLMBaseClass
from cristal.llm.codegen.critic import CodeGenCritic, CriticEvalItem
from cristal.llm.codegen.validator import LLMCodeGenException, LLMCodeGenValidator, load_code
from cristal.llm.utils import get_retry_control
from cristal.logger import logger


class LLMCodeGen:
    """
    LLMCodeGen model.

    Provides an interface for LLM code generation.
    Initialization requires an initialized LLMBaseClass object.

    The critic model is initialized with the given task
    """

    def __init__(self, llm: LLMBaseClass, critic: CodeGenCritic | None = None):
        self.llm: LLMBaseClass = llm
        self.critic: CodeGenCritic | None = critic

    def _call_llm(
        self,
        prompt: str,
        validator: Type[LLMCodeGenValidator],
        max_retries: int | None = None,
        budget: float | None = None
    ) -> tuple[str, bool, str | None]:
        """
        Generate a codegen function

        Note, both max_retries and budget can be passed in as valid terminating conditions.

        :param prompt:          The prompt to use for the code generation.
        :param validator:       The validator to use against the proposed code string.
        :param max_retries:     The maximum number of retries to attempt.
        :param budget:          Budget to perform the evaluation within.
        :return:                The attempt, and a boolean indicating success, and an error if present.
        """
        client = self.llm.get_client()
        if "gemini" in self.llm.code.lower():
            additional_kwargs = {}
        else:
            additional_kwargs = {
                "model": self.llm.code,
                "temperature": self.llm.temperature,
                "seed": self.llm.seed,
            }

        retry_ctrl = get_retry_control(max_retries, budget, wait_range=(0.1, 0.3))
        try:
            response, completion = client.chat.completions.create_with_completion(
                **additional_kwargs,
                messages=[
                    {
                        "role": "developer",  # OpenAI docs say this over 'system' for newer models. Hopefully is
                        # back-compatible
                        "content": self.llm.system_roles["codegen"],
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                response_model=validator,
                max_retries=retry_ctrl,
            )
        except InstructorRetryException as e:
            # Could not manage to produce a response
            last: ChatCompletion = e.last_completion
            content = last.to_dict()['choices'][-1]['message']['content']
            # See if we can load again
            try:
                py_content = json.loads(content)
                code = py_content['code']
            except Exception:
                # Idk
                code = content

            return code, False, str(e)

        return str(response.model_dump()["code"]), True, None

    def _get_function_gen_prompt(self, validator: Type[LLMCodeGenValidator]) -> str:
        args = ""
        for arg in validator.required_args:
            args += f"\n\t{arg}"
        if not args:
            args = "\n\tNo arguments"

        returns = ""
        for ret in validator.returned_data:
            returns += f"\n\t{ret}"
        if not returns:
            returns = "\n\tDoes not return data"

        prompt = f"""Please write a python function according to the following spec:
            Function name: {validator.function_name}
            Purpose: {validator.prompt_description}
            Args: {args}
            Returns: {returns}

            Return your response in JSON format as:
            {{
                "code": <str>
            }}
            where the "code" string is the entire function code.
            """
        return prompt

    def generate_function(
        self,
        validator: Type[LLMCodeGenValidator],
        max_retries: int | None = None,
        budget: float | None = None,
        prompt_override: str | None = None,
        load: bool = True,
    ) -> Callable | str:
        """
        Given a validator, query the model and attempt to generate its code.
        Optionally control LLM retries with max_retries and budget.
        Note that this is for a one-off gen, and not for Critic loops.

        :param validator:       LLMCodeGenValidator object
        :param max_retries:     Maximum number of retries
        :param budget:          Budget for the LLM
        :param prompt_override: Optional prompt override
        :param load:       Whether to return the code loaded
        :return:                Callable function
        """
        if prompt_override:
            prompt = prompt_override
        else:
            prompt = self._get_function_gen_prompt(validator)

        logger.debug(f"Codegen prompt:\n{prompt}")
        # Submit this to the LLM
        code, success, error = self._call_llm(prompt, validator, max_retries, budget)
        if not success:
            raise LLMCodeGenException(f"Failed to generate code for {validator.function_name}.\nError: {error} "
                                      f"Last attempt:\n{code}")

        # Successful code string
        logger.debug(f"Generated validated code for {validator.function_name}:\n{code}")
        if load:
            return load_code(code, validator.function_name)
        else:
            return code

    def _get_gen_prompt_with_eval(
        self, validator: Type[LLMCodeGenValidator], last_attempt: str, evaluation: str
    ) -> str:
        """
        Generate the prompt for the actor codegen call, including the evaluation result
        """
        prompt = self._get_function_gen_prompt(validator)
        prompt += f"""
        ---
        You have previously attempted a solution to this function. However, a Critic Evaluator disapproved
        of your last submission. Please consider your previous solution, enclosed in <PREV_SOLN> tags, and the
        critic's evaluation, enclosed in <EVALUATION> tags, both of which are provided now.
        
        <PREV_SOLN>
        {last_attempt}
        </PREV_SOLN>
        
        <EVALUATION>
        {evaluation}
        </EVALUATION>
        """
        return prompt

    def critic_loop(
        self,
        validator: Type[LLMCodeGenValidator],
        max_iters: int = 10,
        loop_budget: float = 120,  # two minutes
        actor_retries: int | None = None,
        actor_budget: float | None = None,
    ) -> dict:
        """
        Similar to the generate_function method, but incorporates a critic loop.
        The critic's task has already been provided to it.

        :param validator:       LLMCodeGenValidator object. Defines the output format for the actor
        :param max_iters:       Maximum number of iterations for the critic loop
        :param loop_budget:     Budget for the critic loop
        :param actor_retries:   Maximum number of retries for the actor
        :param actor_budget:    Budget for the actor
        :return:                Callable function
        """
        if not self.critic:
            raise LLMCodeGenException("Critic model not initialized")

        # Get initial code attempt
        prompt = self._get_function_gen_prompt(validator)
        directive = self.critic.initial_directive_prompt()
        logger.debug(
            f"DIRECTIVE: "
            f"\n{directive}"
        )
        prompt += f"""
        ---
        To aid in thinking about the solution, consider the following directive.
        
        <DIRECTIVE>
        {directive}
        </DIRECTIVE>
        """

        # We circumnavigate the self.generate_function method, because we want to capture and continue with whatever
        # the produce code attempt is.
        attempt, validated, error = self._call_llm(
            prompt, validator, actor_retries, actor_budget,
        )

        # Main loop
        approved = False
        start = eval_moment = time.time()
        i = 0
        max_critic_eval_retries = 3  # After three, it's probably not returning JSON correct.
        approved_solution = None

        # Gather some data about the loop
        results = []

        while not approved:
            i += 1
            # We'll use the default evaluate retry params
            remaining_budget = loop_budget - (time.time() - start)
            if remaining_budget < 0:
                raise LLMCodeGenException(
                    f"Critic loop budget exhausted without approved function. Latest attempt:\n\n{attempt}"
                )

            # Validation and validation errors are taken into account
            eval_item: CriticEvalItem = self.critic.evaluate_attempt(
                attempt, validated, error, budget=remaining_budget, max_retries=max_critic_eval_retries
            )
            approved = eval_item['approved']

            if approved:
                if not validated:
                    # Trouble. The LLM approves, but it failed our validation.
                    # TODO -- Maybe record these instances
                    logger.warning(
                        "Codegen function approved by critic, but failed internal validation. "
                        "Rejecting proposed solution."
                    )
                    # Set a custom message for passing to the critic that indicates the validation failed,
                    # and count this as a failure.
                    approved = False
                    self.critic.history[-1]['approved'] = False
                    new_eval = f"""
                    {eval_item['evaluation']}
                    ---
                    NOTE: THIS EVALUATION APPROVED THE PROPOSED SOLUTION. HOWEVER, IT FAILED INTERNAL VALIDATION,
                    AND IS THUS REJECTED AS A SOLUTION. 
                    """
                    eval_item['evaluation'] = new_eval
                    self.critic.history[-1]['evaluation'] = new_eval
                else:
                    # Great
                    logger.info(f"Codegen function approved by critic after {i} loops")
                    approved_solution = attempt
                    break

            delta = time.time() - eval_moment
            eval_moment = time.time()

            results.append([
                {
                    "code": attempt,
                    "evaluation": eval_item['evaluation'],
                    "approved": eval_item['approved'],
                    "validated": validated,
                    "validation_error": error,
                    "iter_delta": delta,
                    "total_time": time.time() - start,
                }
            ])

            # Check if exhausted
            if i >= max_iters:
                logger.warning(
                    f"Max iterations ({max_iters}) of critic loop reached without approved function. Latest "
                    f"attempt:\n\n{attempt}"
                )
                break

            remaining_budget = loop_budget - (time.time() - start)
            if remaining_budget < 0:
                logger.warning(
                    f"Critic loop budget exhausted without approved function. Latest attempt:\n\n{attempt}"
                )
                break

            # We have enough resources to have another attempt
            prompt = self._get_gen_prompt_with_eval(validator, attempt, eval_item['evaluation'])
            attempt_budget = min(actor_budget, remaining_budget)
            attempt, validated, error = self._call_llm(
                prompt, validator, actor_retries, attempt_budget,
            )

        return {
            "all_results": results,
            "last_result": results[-1],
            "approved": approved,
            "evaluation": self.critic.history[-1]['evaluation'],
            "code": approved_solution,
            "function_name": validator.function_name,
            "iters_used": i,
            "total_time": time.time() - start,
        }
