"""
Consolidate LLM models, and LLM utils etc.
"""
import os
from abc import ABC
from typing import Literal

import instructor
from pydantic import create_model, Field
from tqdm import tqdm

from alla.llm import utils as llm_utils
from alla.llm.wrangling import read_reports
from alla.logger import logger
from alla.utils import PROJECT_ROOT


class LLMBaseClass(ABC):
    """
    Base class for all LLM models
    """
    name = None
    code = None
    path_code = None  # Certain characters--looking at you colon (wait what?)--are bad for making file paths (
    # defaults to code)
    context_window = None
    aliases = []

    def __init__(
        self,
        temperature: float = 0.0,
        seed: int = 101010,
    ):
        self.temperature = temperature
        self.seed = seed
        self.path_code = self.path_code or self.code
        self._client = None

    def to_json(self):
        return {
            "name": self.name,
            "code": self.code,
            "path_code": self.path_code,
            "seed": self.seed,
            "context_window": self.context_window,
            "temperature": self.temperature,
        }

    def init_client(self) -> None:
        raise NotImplementedError()

    def get_client(self, cached: bool = True) -> instructor.Instructor | instructor.AsyncInstructor:
        """Return the client, initializing if needed. Use cached=False to re-init"""
        if not cached or not self._client:
            self.init_client()
        return self._client

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
        if not report_dir and not reports_text:
            raise ValueError("Either report_dir or reports_text must be provided")
        if report_dir and reports_text:
            # Hmm...
            logger.warning("Received report_dir and reports_text. Opting for reports_text.")

        if not reports_text:
            reports_text = read_reports(report_dir, include, exclude)

        if verbose is True:
            tqdm.write(f"Inferring value from report for '{asset_id}' for {indicator_name}")

        try:
            prompt = llm_utils.prompt_se_context(indicator_name, reports_text)
        except KeyError:
            raise ValueError(f'Prompt for {indicator_name} not found')
        except FileNotFoundError:
            raise FileNotFoundError(f'File not found for asset_id {asset_id}')

        client = self.get_client()
        response, completion = client.chat.completions.create_with_completion(
            model=self.code,
            temperature=self.temperature,
            messages=[
                {"role": "user", "content": prompt},
            ],
            response_model=self._create_indicator_class(),
            max_retries=3,
            seed=self.seed,
        )
        py_response = response.model_dump()
        # Add the indicator value
        py_response["indicator"] = indicator_name

        return py_response

    def evaluate_asset(
        self, prompt: str, max_retries: int = 3, budget: float | None = None
    ) -> llm_utils.AssetEvaluationV2:
        """
        Evaluates an asset given its reports and indicators.
        Produces this model's estimate of the asset's growth forecast and category distribution.

        :param prompt:                  The prompt to use for the evaluation.
        :param max_retries:             The maximum number of retries to attempt.
        :param budget:                  Budget to perform the evaluation within.
        :return:                        Asset evaluation
        """
        client = self.get_client()
        response, completion = client.chat.completions.create_with_completion(
            model=self.code,
            temperature=self.temperature,
            messages=[
                {"role": "user", "content": prompt},
            ],
            response_model=llm_utils.AssetEvaluationV2,
            max_retries=max_retries,
            seed=self.seed,
        )
        return response

    def __str__(self):
        """Return name and class dot notation"""
        return f"{self.name} ({self.__class__.__module__}.{self.__class__.__name__})"

    def __repr__(self):
        """Return name and class dot notation"""
        return self.__str__()
