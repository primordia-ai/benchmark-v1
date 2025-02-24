"""
Validators and formatting models for LLM codegen
"""
import ast

from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError


class LLMCodeGenException(ValidationError):
    """
    Custom error for LLM codegen
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
    "alla", "numpy", "pandas", "scipy", "sklearn", "pymc", "xarray", "jax",
}


class LLMCodeGen(BaseModel):
    """
    Base validator for LLM-generated code
    """
    code: str = Field(
        ..., title="Code",
        description="Python code for execution."
    )

    # Private attributes that avoid regular validation stream
    _is_function = None  # Must be set to specified to declare if returned code is a function or not
    _function_name = None
    _required_args = None  # Tuple

    @model_validator("before")
    @classmethod
    def model_validate(cls):
        if cls._is_function is None:
            raise ValueError(f"Class {cls} must set _is_function to True or False.")

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
        if not cls._is_function:
            return code

        tree = ast.parse(code)

        # Ensure there's only one function definition
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if len(functions) != 1:
            raise ValueError("There must be exactly one function definition.")

        func = functions[0]

        # Validate function name
        if func.name != cls._function_name:
            raise ValueError(f"Function name must be '{cls._function_name}'.")

        # Validate required argument names
        actual_args = [arg.arg for arg in func.args.args]

        if actual_args != cls._required_args:
            raise ValueError(f"Expected arguments {cls._required_args}, got {actual_args}.")

        return code
