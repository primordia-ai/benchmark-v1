"""
Validators and formatting models for LLM codegen
"""
from typing import Optional, ClassVar
import ast

from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError


class LLMCodeGenException(ValidationError):
    """
    Custom error for LLM codegen
    TODO -- Doesn't work atm...
    """

    def __init__(self, message: str, exception: Exception):
        # Format the message
        message = f"LLMCodeGenException: {message}\nSource Exception: {exception}"
        super().__init__(message)
        self.exception = exception


INSECURE_METHODS = {
    "exec", "eval", "compile", "globals", "locals",
    "open", "input", "file", "os", "sys", "subprocess",
    "shutil", "ctypes", "getattr", "setattr", "delattr",
    "vars", "socket", "requests", "http.client", "urllib",
    "pathlib"
}

GREEN_IMPORTS = {
    # Imports that we trust
    # Built-ins
    "functools", "itertools", "math", "copy", "datetime", "random", "json",
    # Science
    "alla", "cristal", "numpy", "pandas", "scipy", "sklearn", "pymc", "xarray", "jax",
}


class LLMCodeGenValidator(BaseModel):
    """
    Base validator for LLM-generated code
    """
    code: str = Field(
        ..., title="Code",
        description="Python code for execution."
    )
    # Self-referencing other methods in multimethod scenarios
    sibling: 'Optional[LLMCodeGenValidator]' = None

    # Private attributes that avoid regular validation stream
    is_function: ClassVar[bool | None] = None  # Must be set to specified to declare if returned code is a function or
    # not
    function_name: ClassVar[str | None] = None
    required_args: ClassVar[tuple | None] = None  # Tuple

    def __init__(self, **data):
        super().__init__(**data)
        if self.is_function is None:
            raise ValueError(f"{self} must set 'is_function' to True or False.")

    @field_validator("code", mode="after")
    @classmethod
    def base_code_validator(cls, code: str):
        """
        A base code validator that covers some desirable defaults
        """
        tree = ast.parse(code)

        # Check imports if any
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if alias.name.split(".")[0] not in GREEN_IMPORTS:
                        raise ValueError(f"Import '{alias.name}' is not allowed.")

        # Prevent unsafe operations
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in INSECURE_METHODS:
                    raise ValueError(f"Usage of '{node.func.id}' is not allowed.")

        return code

    @field_validator("code", mode="after")
    @classmethod
    def function_meta_validator(cls, code: str):
        """
        Uses AST to validate:
        - Correct function name
        - Correct function signature
        """
        if not cls.is_function:
            return code

        tree = ast.parse(code)

        # Ensure there's only one function definition
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if len(functions) != 1:
            raise ValueError("There must be exactly one function definition.")

        func = functions[0]

        # Validate function name
        if func.name != cls.function_name:
            raise ValueError(f"Function name must be '{cls.function_name}'.")

        # Validate required argument names
        actual_args = tuple([arg.arg for arg in func.args.args])

        if actual_args != cls.required_args:
            raise ValueError(f"Expected arguments {cls.required_args}, got {actual_args}.")

        return code
