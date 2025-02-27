"""
Consolidate LLM models, and LLM utils etc.
"""
from abc import ABC

import instructor
import time
from instructor.exceptions import InstructorRetryException
from tenacity import Retrying, stop_after_attempt, stop_after_delay, wait_none, wait_random, RetryCallState
from tqdm import tqdm

from alla.llm import utils as llm_utils
from alla.llm.wrangling import read_reports
from alla.logger import logger
from alla.utils import BudgetMode


def retry_error_callback(state: RetryCallState) -> None:
    exception = state.outcome.exception()
    result = f"{exception.__class__.__name__} {exception})"
    start = state.start_time
    elapsed = float(round(time.monotonic() - start, 2))
    logger.debug(
        f"Retry ({state.attempt_number}) Error. Elapsed time {elapsed}. Reason: {result}"
    )


def before_retry_callback(state: RetryCallState) -> None:
    if state.attempt_number > 1:
        start = state.start_time
        elapsed = float(round(time.monotonic() - start, 2))
        slept = float(round(state.idle_for, 2))
        logger.debug(
            f"Retry ({state.attempt_number}) Start. Slept {slept}. Elapsed time {elapsed}..."
        )


def get_retry_control(
    max_retries: int | None = None, budget: float | None = None, wait_range: tuple[float, float] | None = None
) -> Retrying:
    # Wait time between requests
    wait = wait_random(wait_range[0], wait_range[1]) if wait_range else wait_none()

    if max_retries and budget:
        stop = (stop_after_attempt(max_retries) | stop_after_delay(budget))
    elif max_retries:
        stop = stop_after_attempt(max_retries)
    elif budget:
        stop = stop_after_delay(budget)
    else:
        logger.warning("No retry control specified. Defaulting to 3 attempts.")
        stop = stop_after_attempt(3)

    return Retrying(
        stop=stop,
        wait=wait,
        # TODO -- Seems to cause an error
        before=before_retry_callback,
        retry_error_callback=retry_error_callback,
    )

def init_llm_prompt_logger(log_path: str):
    """Initialize a logger specifically for LLM prompts"""
    import logging
    # From the Instructor github, it looks like they use 'instructor' as logger name
    logger = logging.getLogger("instructor")
    logger.setLevel(logging.DEBUG)

    # Create a file handler for logging prompts
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Add formatter for better output
    formatter = logging.Formatter('[ {name}:{levelname} ]\t{message}', style='{')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

    return logger


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

    system_roles = {
        "financial_analyst": """
        You are a financial analyst tasked with evaluating companies, or 'assets', based on their financials.
        The financials may come in the form of quantitative figures, or qualitative reports. 
        """,
        "alla": f"""
        You are ALLA, an AI assistant tasked with evaluating companies, or 'assets', based on their financials.
        The financials may come in the form of quantitative figures, or qualitative reports.
        You are a Python class with access to a variety of methods and attributes to help you with your tasks.
        """,  # Inserting the alla.utils.describe_class(ALLA) would be a circular import
    }

    def __init__(
        self,
        temperature: float = 0.0,
        seed: int = 101010,
        budget_mode: BudgetMode = BudgetMode.WALL,
    ):
        self.temperature = temperature
        self.seed = seed
        self.budget_mode = budget_mode
        self.path_code = self.path_code or self.code
        self._client = None
        # Separate LLM prompt logger
        self.logger = None

    def to_json(self):
        return {
            "name": self.name,
            "code": self.code,
            "path_code": self.path_code,
            "budget_mode": str(self.budget_mode),
            "seed": self.seed,
            "context_window": self.context_window,
            "temperature": self.temperature,
        }

    def log_to_file(self, file_path: str):
        """
        Initialize a log file to record LLM-specific events (namely, prompts).
        Instructor automatically logs at the DEBUG level:
            https://python.useinstructor.com/concepts/logging/
        """
        if not self.logger:
            self.logger = init_llm_prompt_logger(file_path)

    def init_client(self) -> None:
        raise NotImplementedError()

    def get_client(self, cached: bool = True) -> instructor.Instructor | instructor.AsyncInstructor:
        """Return the client, initializing if needed. Use cached=False to re-init"""
        if not cached or not self._client:
            self.init_client()
        return self._client

    def extract_indicator(
        self,
        indicator_name: str,
        asset_id: str,
        report_dir: str | None = None,  # TODO -- should end with path_code
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        reports_text: str | None = None,
        verbose=False,
        budget: float | None = None,
        max_retries: int | None = None,
    ) -> tuple[dict, bool]:
        """
        Retrieves the value of a specified indicator for a given asset ID by generating a report using a local LLM.

        Note, both max_retries and budget can be passed in as valid terminating conditions.

        :param indicator_name:  The name of the indicator to retrieve.
        :param asset_id:        The ID of the asset for which to retrieve the indicator.
        :param include:         Optional list of specific reports to comprise in the report.
        :param exclude:         Optional list of specific reports to exclude from the report.
        :param report_dir:      The path to the directory containing the reports.
        :param reports_text:    Optionally provide reports to skip the compilation.
        :param verbose:         If True, prints additional information during execution.
        :param budget:          Budget to perform the evaluation within.
        :param max_retries:     The maximum number of retries to attempt. Default is None.
        :return:                Tuple[A dictionary with keys "name", "value", "reason", Boolean of LLM success]
        """
        if max_retries is None and budget is None:
            raise ValueError("One of max_retries or budget must be provided")

        if budget and self.budget_mode != BudgetMode.WALL:
            raise NotImplementedError(f"Budget mode '{self.budget_mode}' is not supported for 'extract_indicator'")

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
        try:
            response, completion = client.chat.completions.create_with_completion(
                model=self.code,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "developer",  # OpenAI docs say this over 'system' for newer models. Hopefully is
                        # back-compatible
                        "content": self.system_roles["financial_analyst"],
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                response_model=llm_utils.ExtractedIndicator,
                max_retries=get_retry_control(max_retries, budget, wait_range=(0.1, 1.0)),
                seed=self.seed,
            )
        except InstructorRetryException as e:
            response = e.last_completion  # e.messages[-1]
            py_response = response.model_dump()
            # Add the indicator value
            py_response["indicator"] = indicator_name
            py_response["error"] = str(e)
            py_response["last_completion"] = e.messages[-1]
            # py_response["reason"] = "Failed to extract soft indicator from reports"
            # py_response["value"] = None
            # py_response["numeric"] = None
            return py_response, False
        except AttributeError as e:
            print(f"MW EXCEPTION -- {e}")
            # This is a bug in the instructor library
            py_response = {
                "error": str(e),
                "reason": "Failed to extract soft indicator from reports",
                "value": None,
                "numeric": None,
            }
            return py_response, False
        else:
            # Success
            py_response = response.model_dump()
            # Add the indicator value
            py_response["indicator"] = indicator_name

            return py_response, True

    def evaluate_asset(
        self, prompt: str, max_retries: int | None = None, budget: float | None = None
    ) -> tuple[dict, bool]:
        """
        Evaluates an asset given its reports and indicators.
        Produces this model's estimate of the asset's growth forecast and category distribution.

        Note, both max_retries and budget can be passed in as valid terminating conditions.

        :param prompt:                  The prompt to use for the evaluation.
        :param max_retries:             The maximum number of retries to attempt.
        :param budget:                  Budget to perform the evaluation within.
        :return:                        Asset evaluation and bool indicating response success
        """
        client = self.get_client()
        try:
            response, completion = client.chat.completions.create_with_completion(
                model=self.code,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                response_model=llm_utils.AssetEvaluationV2,
                max_retries=get_retry_control(max_retries, budget, wait_range=(0.1, 1.0)),
                seed=self.seed,
            )
        except InstructorRetryException as e:
            result = {
                "error": str(e),
                "last_completion": e.last_completion,  # e.messages[-1]
            }
            return result, False
        else:
            # Success
            return response.model_dump(), True

    def __str__(self):
        """Return name and class dot notation"""
        return f"{self.name} ({self.__class__.__module__}.{self.__class__.__name__})"

    def __repr__(self):
        """Return name and class dot notation"""
        return self.__str__()
