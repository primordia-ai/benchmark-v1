"""
LLMCodeGen model
"""
from typing import Callable, Type

from cristal.llm.base import LLMBaseClass
from cristal.llm.codegen.validator import LLMCodeGenException, LLMCodeGenValidator, load_code
from cristal.logger import logger


class LLMCodeGen:
    """
    LLMCodeGen model.

    Provides an interface for LLM code generation.
    Initialization requires an initialized LLMBaseClass object.
    """

    def __init__(self, llm: LLMBaseClass):
        self.llm: LLMBaseClass = llm

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
                where the "code" string the entire function code.
                """

        logger.debug(f"Codegen prompt:\n{prompt}")
        # Submit this to the LLM
        code, success = self.llm.generate_codegen_function(prompt, validator, max_retries, budget)
        if not success:
            raise LLMCodeGenException(f"Failed to generate code for {validator.function_name}")

        # Successful code string
        return load_code(code, validator.function_name)
