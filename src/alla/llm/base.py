"""
Consolidate LLM models, and LLM utils etc.
"""
import logging
import os
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, create_model, Field

from alla.asset import Asset
from alla.utils import PROJECT_ROOT

logger = logging.getLogger("alla")


class GrowthEstimate(BaseModel):
    mean: float
    std: float


class CategoryDistribution(BaseModel):
    high_growth: float
    stable: float
    time_bomb: float


class AssetEvaluation(BaseModel):
    growth: GrowthEstimate
    category: CategoryDistribution


class LLMBaseClass(ABC):
    """
    Base class for all LLM models
    """
    name = None
    code = None

    def __init__(
        self,
        temperature: float = 0.0,
        prompt_header: str = "",
        prompt_footer: str = "",
        history_summary: str = "",
        history_tokens: int = 1024,
        seed: int = 101010,
    ):
        self.temperature = temperature
        self.prompt_header = prompt_header
        self.prompt_footer = prompt_footer
        self.history_summary = history_summary
        self.history_tokens = history_tokens
        self.seed = seed

    def to_json(self):
        return {
            "name": self.name,
            "code": self.code,
            "seed": self.seed,
            "temperature": self.temperature,
            "prompt_header": self.prompt_header,
            "prompt_footer": self.prompt_footer,
            "history_summary": self.history_summary,
            "history_tokens": self.history_tokens,
        }

    @staticmethod
    def _create_indicator_class():
        """
        Creates a Pydantic model for sending over OpenAI API.
        :return:                A Pydantic model for the indicator.
        """
        return create_model(
            'LLMIndicator',
            value=(Literal['red', 'yellow', 'green'], Field(
                ...,
                title="Indicator value",
                description="The value of the indicator, represented by the colors red, yellow, and green."
            )),
            reason=(str, Field(
                ...,
                title="Value reason",
                description="The reason behind the indicator's value based on the company reports. This text can be long and detailed."
            ))
        )

    @abstractmethod
    def extract_indicator(
        self,
        indicator_name: str,
        asset_id: str,
        report_dir: str = os.path.join(PROJECT_ROOT, "data/reports"),
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        reports_text: str | None = None,
        verbose=False
    ) -> dict:
        """
        Retrieves the value of a specified indicator for a given asset ID by generating a report using a local LLM.

        :param indicator_name:  The name of the indicator to retrieve.
        :param asset_id:        The ID of the asset for which to retrieve the indicator.
        :param include:         Optional list of specific reports to comprise in the report.
        :param exclude:         Optional list of specific reports to exclude from the report.
        :param report_dir:      The path to the directory containing the reports.
        :param reports_text:    Optionally provide reports to skip the compilation.
        :param verbose:         If True, prints additional information during execution.

        :return:                A dictionary with keys "name", "value", "reason"
        """
        raise NotImplementedError()

    def evaluate_asset(self, asset: Asset, reports_txt: str, prompt: str) -> AssetEvaluation:
        """
        Evaluates an asset given its reports and indicators.
        Produces this model's estimate of the asset's growth forecast and category distribution.

        :param asset:           The asset to evaluate.
        :param reports_txt:     The reports data in a block of text.
        :param prompt:          The prompt to use for the evaluation.
        :return:                AssetEvaluation
        """
        raise NotImplementedError()

    def __str__(self):
        """Return name and class dot notation"""
        return f"{self.name} ({self.__class__.__module__}.{self.__class__.__name__})"

    def __repr__(self):
        """Return name and class dot notation"""
        return self.__str__()

