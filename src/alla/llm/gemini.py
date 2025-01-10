import os
import json
import instructor
from openai import OpenAI
import google.generativeai as genai
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
    api_key = secrets.get("GEMINI_API_KEY")
    return api_key

class Gemini15(LLMBaseClass):
    """
    gemini model
    """
    name = "gemini_1.5"
    code = "models/gemini-1.5-flash-latest"
    api_key = get_api_key()
    aliases = [
        "gemini1.5",
        "gemini-1.5",
    ]

    def __init__(self, temperature: float = 0.0, seed: int = 101010):
        super().__init__(
            temperature=temperature,
            seed=seed
        )
        # set the api key and configure it for the gemini model
        genai.configure(api_key=self.api_key)

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

        client = instructor.from_gemini(
                client=genai.GenerativeModel(
                    model_name=self.code,
                ),
                mode=instructor.Mode.GEMINI_JSON,
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
        client = instructor.from_gemini(
                client=genai.GenerativeModel(
                    model_name=self.code,
                ),
                mode=instructor.Mode.GEMINI_JSON,
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
    

class Gemini2(Gemini15):
    name = "gemini_2.0"
    code = "models/gemini-2.0-flash-exp"
    api_key = get_api_key()
    aliases = [
        "gemini2",
        "gemini-2",
    ]
