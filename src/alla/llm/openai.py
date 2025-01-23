import json
import os

import instructor
from openai import OpenAI
from tqdm import tqdm

from alla.llm import utils as llm_utils
from alla.llm.base import AssetEvaluation, LLMBaseClass
from alla.llm.wrangling import read_reports
from alla.logger import logger
from alla.utils import PROJECT_ROOT


def get_api_key():
    secrets_path = os.path.join(PROJECT_ROOT, ".secrets.json")
    with open(secrets_path, "r") as f:
        secrets = json.load(f)
    api_key = secrets.get("OPENAI_API_KEY")
    return api_key


class GPT(LLMBaseClass):
    """
    GPT class for interacting with the GPT-4o-mini model.
    """

    name = "gpt-4o-mini"
    code = "gpt-4o-mini"
    api_key = get_api_key()
    aliases = [
        "gpt-4o-mini",
        "gpt4omini",
    ]

    def __init__(self, temperature: float = 0.0, seed: int = 101010):
        super().__init__(
            temperature=temperature,
            seed=seed
        )
        # set the api key and configure it for the gemini model

    def process_asset(self, asset):
        pass

    def extract_indicator(
        self,
        indicator_name: str,
        asset_id: str,
        report_dir: str = os.path.join(PROJECT_ROOT, "data/reports"),
        include: list = None,
        exclude: list = None,
        reports_text: str | None = None,
        verbose=False
    ) -> dict:
        """See base class for description"""
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

        # Use LangChain Ollama API to call a local LLM and generate a report
        # response = ollama.generate(model=model_name, prompt=prompt)

        client = instructor.from_openai(OpenAI(api_key=self.api_key), mode=instructor.Mode.TOOLS_STRICT)

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
    ) -> AssetEvaluation:
        """See base class for description"""
        client = instructor.from_openai(OpenAI(api_key=self.api_key), mode=instructor.Mode.TOOLS_STRICT)

        response, completion = client.chat.completions.create_with_completion(
            model=self.code,
            temperature=self.temperature,
            messages=[
                {"role": "user", "content": prompt},
            ],
            response_model=AssetEvaluation,
            max_retries=max_retries,
            seed=self.seed,
        )
        return response


class GPT4o(GPT):
    """
    GPT4o is a subclass of the GPT class representing a specific version of the GPT model.
    Attributes:
        name (str): The name of the model.
        code (str): The code identifier for the model version.
        api_key (str): The API key used for authentication.
        aliases (list): A list of alternative names for the model.
    """

    name = "gpt-4o"
    code = "gpt-4o-2024-08-06"
    api_key = get_api_key()
    aliases = [
        "gpt-4o",
        "gpt4o",
    ]
