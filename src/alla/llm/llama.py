import os

import instructor
from openai import OpenAI
from tqdm import tqdm

from alla.llm import utils as llm_utils
from alla.llm.base import AssetEvaluation, LLMBaseClass
from alla.llm.wrangling import read_reports
from alla.logger import logger
from alla.utils import PROJECT_ROOT


class LlamaSmall(LLMBaseClass):
    """
    Small Llama model
    """
    name = "Llama 3.1 8B"
    code = "llama3.1:8b"
    context_window = 128_000
    aliases = [
        "llama-small",
        "llama-8b",
    ]

    def __init__(self, temperature: float = 0.0, seed: int = 101010):
        super().__init__(
            temperature=temperature,
            seed=seed
        )

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

        client = instructor.from_openai(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # required, but unused
            ),
            mode=instructor.Mode.JSON,
        )

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
        self, prompt: str, max_retries: int = 3
    ) -> AssetEvaluation:
        """See base class for description"""
        client = instructor.from_openai(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # required, but unused
            ),
            mode=instructor.Mode.JSON,
        )

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


class LlamaSmallAlt(LlamaSmall):
    """Another 8B model, that works with older versions of Ollama : / """
    name = "Llama 3 8B"
    code = "llama3:8b"
    context_window = 8192
    aliases = []


class LlamaTiny(LlamaSmall):
    name = "Llama 3.2 1B"
    code = "llama3.2:1b"
    context_window = 128_000
    aliases = [
        "llama-tiny",
        "llama-1b",
    ]


class LlamaMamma(LlamaSmall):
    name = "Llama 3 70B"
    code = "llama3:70b"
    context_window = 128_000
    aliases = [
        "llama-70b",
        "llama-mamma",
    ]
