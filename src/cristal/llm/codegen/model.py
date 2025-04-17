"""
LLMCodeGen model
"""
from typing import Callable, Type

from instructor.exceptions import InstructorRetryException

from cristal.llm.base import LLMBaseClass
from cristal.llm.codegen.validator import LLMCodeGenException, LLMCodeGenValidator, load_code
from cristal.llm.utils import get_retry_control
from cristal.logger import logger


class LLMCodeGen:
    """
    LLMCodeGen model.

    Provides an interface for LLM code generation.
    Initialization requires an initialized LLMBaseClass object.


    """

    def __init__(self, llm: LLMBaseClass, critic: LLMBaseClass | None = None):
        self.llm: LLMBaseClass = llm
        self.critic: LLMBaseClass | None = critic

    def call_llm(
        self,
        prompt: str,
        validator: Type[LLMCodeGenValidator],
        max_retries: int | None = None,
        budget: float | None = None
    ) -> tuple[str | dict, bool]:
        """
        Generate a codegen function

        Note, both max_retries and budget can be passed in as valid terminating conditions.

        :param prompt:          The prompt to use for the code generation.
        :param validator:       The validator to use against the proposed code string.
        :param max_retries:     The maximum number of retries to attempt.
        :param budget:          Budget to perform the evaluation within.
        :return:                The validated code string, and a boolean indicating success
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
            error_result = {
                "error": str(e),
                "last_completion": e.last_completion,  # e.messages[-1]
            }
            return error_result, False

        return response.model_dump()["code"], True

    def generate_function(
        self,
        validator: Type[LLMCodeGenValidator],
        max_retries: int | None = None,
        budget: float | None = None,
        prompt_override: str | None = None,
    ) -> Callable:
        """
        Given a validator, query the model and attempt to generate its code.
        Optionally control LLM retries with max_retries and budget.

        :param validator:       LLMCodeGenValidator object
        :param max_retries:     Maximum number of retries
        :param budget:          Budget for the LLM
        :param prompt_override: Optional prompt override
        :return:                Callable function
        """
        if prompt_override:
            prompt = prompt_override
        else:
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

        logger.debug(f"Codegen prompt:\n{prompt}")
        # Submit this to the LLM
        code, success = self.call_llm(prompt, validator, max_retries, budget)
        if not success:
            logger.debug(
                f"Codegen failed for function {validator.function_name}. Last attempt:\n"
                f"{code['last_completion']}"
            )

            raise LLMCodeGenException(f"Failed to generate code for {validator.function_name}")

        # Successful code string
        logger.debug(f"Generated validated code for {validator.function_name}:\n{code}")
        return load_code(code, validator.function_name)
