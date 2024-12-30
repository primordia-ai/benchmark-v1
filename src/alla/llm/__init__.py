from typing import Type

from .base import LLMBaseClass
from .llama import LlamaSmall, LlamaSmallAlt, LlamaTiny, LlamaMamma

AVAILABLE_MODELS = {
    # Model codes to classes
    LlamaSmall.code: LlamaSmall,
    LlamaSmallAlt.code: LlamaSmallAlt,
    LlamaTiny.code: LlamaTiny,
    LlamaMamma.code: LlamaMamma,
}


def get_llm(identifier: str) -> "Type[LLMBaseClass]":
    """
    Return an LLM class from an identifier.
    The identifier can either be the code, name, or alias (in that hierarchical order).
    """
    # First try to match across codes
    for model in AVAILABLE_MODELS.values():
        if model.code == identifier:
            return model

    # Then try to match across names
    for model in AVAILABLE_MODELS.values():
        if model.name == identifier:
            return model

    # Finally, try to match across aliases
    for model in AVAILABLE_MODELS.values():
        if identifier in model.aliases:
            return model

    # If we get here, we've failed
    raise ValueError(f"Model with identifier '{identifier}' not found. Available models are {AVAILABLE_MODELS.keys()}")
