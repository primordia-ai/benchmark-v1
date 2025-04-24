"""
Validators and formatting models for LLM codegen
"""
import ast
from typing import Any, Callable, ClassVar

from cristal.logger import logger
from pydantic import BaseModel, Field, field_validator, ValidationError


class LLMCodeGenException(ValidationError):
    """
    Custom error for LLM codegen
    """

    # TODO -- Doesn't work atm
    # def __init__(self, message: str, exception: Exception):
    #     # Format the message
    #     message = f"LLMCodeGenException: {message}\nSource Exception: {exception}"
    #     super().__init__(message)
    #     self.exception = exception


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


class CodeGenInputVar(BaseModel):
    """
    Defines the formatting of a codegen function input argument.
    Helps the LLM understand what data is incoming.
    """
    arg: str
    description: str
    datatype: Any


class CodeGenOutputVar(BaseModel):
    """
    Defines the formatting of a codegen function output argument.
    Helps the LLM understand what data to return.
    """
    name: str
    description: str
    datatype: Any


def load_code(code: str, func_name) -> Callable:
    """
    Load code into AST and add to the namespace. Return the function.
    """
    tree = ast.parse(code)
    # Check for green imports and add them to the namespace
    # Also provide some standards
    namespace = {}
    # The following is to try and import green libs on the fly,
    # but it seems to be a bit buggy and the internal function fails to use them
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name.split(".")[0] in GREEN_IMPORTS:
                    # Add the import to the namespace
                    namespace[alias.name.split(".")[0]] = __import__(alias.name)
                else:
                    raise LLMCodeGenException(f"Import '{alias.name}' is not allowed in codegen")

    # Create local namespace for this function
    # exec(code, {}, namespace)
    exec(code, namespace)
    # Load the imports in from the namespace (those that are func_name)
    return namespace[func_name]


class LLMCodeGenValidator(BaseModel):
    """
    Base validator for LLM-generated code
    """
    code: str = Field(
        ..., title="Code",
        description="Python code for execution."
    )
    # Self-referencing other methods in multimethod scenarios
    # Disable this for now
    # sibling: 'Optional[LLMCodeGenValidator]' = None

    # ---
    # Private attributes that avoid regular validation stream
    is_function: ClassVar[bool | None] = None  # Must be set to specified to declare if returned code is a function or
    # not
    function_name: ClassVar[str | None] = None
    required_args: ClassVar[tuple[CodeGenInputVar] | None] = None
    returned_data: ClassVar[tuple[CodeGenOutputVar] | None] = None
    # Optional description to pass into LLM for codegen. We default to the validator docstring if this is null.
    # The args and returned data objects will automatically be included in the prompt.
    prompt_description: ClassVar[str | None] = None

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

        arg_names = tuple([arg['arg'] for arg in cls.required_args])
        if actual_args != arg_names:
            raise ValueError(f"Expected arguments {arg_names}, got {actual_args}.")

        # Much less clear how to validate output data, but we can at least check the number of returns
        # TODO -- Evaluating return datatype? Might have to be through descendant classes
        for stmt in func.body:
            if isinstance(stmt, ast.Return):
                if stmt.value is None:
                    raise ValueError("Explicit 'return' with no value is not allowed.")

                elif isinstance(stmt.value, ast.Tuple):
                    if len(stmt.value.elts) != len(cls.returned_data):
                        raise ValueError(
                            f"Expected {len(cls.returned_data)} return values, got {len(stmt.value.elts)}."
                        )

                elif len(cls.returned_data) != 1:
                    raise ValueError(
                        f"Expected {len(cls.returned_data)} return values, got 1."
                    )

        return code
