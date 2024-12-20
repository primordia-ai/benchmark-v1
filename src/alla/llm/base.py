"""
Consolidate LLM models, and LLM utils etc.
"""
import logging
import os
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import create_model, Field

from alla.asset import Asset
from alla.utils import PROJECT_ROOT

logger = logging.getLogger("alla")


class LLMBaseClass(ABC):
    """
    Base class for all LLM models
    """
    name = None
    code = None

    def __init__(self, temperature: float = 0.0):
        self.temperature = temperature

    def __str__(self):
        """Return name and class dot notation"""
        return f"{self.name} ({self.__class__.__module__}.{self.__class__.__name__})"

    def __repr__(self):
        """Return name and class dot notation"""
        return self.__str__()

    @staticmethod
    def _create_indicator_class(indicator_name: str):
        """
        Creates a Pydantic model for sending over OpenAI API.

        :param indicator_name:  The name of the indicator to create the model for.
        :return:                A Pydantic model for the indicator.
        """
        return create_model(
            'LLMIndicator',
            indicator=(Literal[indicator_name], Field(
                ...,
                title="Indicator name",
                description="The name of the indicator to evaluate the company's performance as given by the prompt."
            )),
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

    def read_reports(
        self,
        report_dir: str = os.path.join(PROJECT_ROOT, "reports"),
        include: list = None,
        exclude: list = None,
    ):
        if not os.path.exists(report_dir):
            raise FileNotFoundError(f"Report directory {report_dir} not found")

        # Read the documents for the asset, with recursion if no reports list included
        reports = ""

        for filename in os.listdir(report_dir):
            if filename in exclude:
                continue
            if include is not None and filename not in include:
                continue

            file_path = os.path.join(report_dir, filename)
            logger.debug(f"Reading file {file_path}")
            with open(file_path, 'r') as file:
                reports += "\n<START DOCUMENT>" + file.read() + "\n<END DOCUMENT>"

        return reports

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
