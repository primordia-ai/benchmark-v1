"""
Consolidate LLM models, and LLM utils etc.
"""
from abc import ABC
from alla.asset import Asset


MODELS = {
    "llama": "placeholder.llama.class",
}


class LLMBaseClass(ABC):
    """
    Base class for all LLM models
    """
    name = None

    def extract_soft_indicator(self, asset: Asset, indicator_name):
        ...

    def __str__(self):
        """Return name and class dot notation"""
        return f"{self.name} ({self.__class__.__module__}.{self.__class__.__name__})"

    def __repr__(self):
        """Return name and class dot notation"""
        return self.__str__()
