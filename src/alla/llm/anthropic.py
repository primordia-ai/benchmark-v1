import os
import json
import instructor
from openai import OpenAI
from anthropic import Anthropic
from tqdm import tqdm

from alla.llm import utils as llm_utils
from alla.llm.base import AssetEvaluation, LLMBaseClass
from alla.llm.wrangling import read_reports
from alla.logger import logger
from alla.utils import PROJECT_ROOT

def get_api_key() -> str:
    """
    Retrieves the Anthropic API key from a .secrets.json file located in the project root directory.
    The function constructs the path to the .secrets.json file, reads its contents, and extracts the value
    associated with the "ANTHROPIC_API_KEY" key.
    Returns:
        str: The Anthropic API key if found, otherwise None.
    """
    secrets_path = os.path.join(PROJECT_ROOT, ".secrets.json")
    with open(secrets_path, "r") as f:
        secrets = json.load(f)
    api_key = secrets.get("ANTHROPIC_API_KEY")
    return api_key

class AnthropicModel(LLMBaseClass):
    """
    Anthropic class for interacting with the Claude-3-5-Sonnet model.
    Attributes:
        name (str): The name of the model.
        code (str): The code identifier for the model.
        api_key (str): The API key for accessing the model.
        aliases (list): List of aliases for the model.
    Methods:
        __init__(temperature: float = 0.0, seed: int = 101010):
            Initializes the Anthropic instance with specified temperature and seed.
        process_asset(asset):
            Placeholder method for processing an asset.
        extract_indicator(
            Extracts the specified indicator from the asset's report.
            Args:
                indicator_name (str): The name of the indicator to extract.
                asset_id (str): The ID of the asset.
                report_dir (str, optional): Directory containing the reports. Defaults to os.path.join(PROJECT_ROOT, "data/reports").
                include (list, optional): List of report filenames to include. Defaults to None.
                exclude (list, optional): List of report filenames to exclude. Defaults to None.
                reports_text (str | None, optional): Text of the reports. Defaults to None.
                verbose (bool, optional): If True, prints additional information. Defaults to False.
            Returns:
                dict: A dictionary containing the extracted indicator value.
        evaluate_asset(prompt: str, max_retries: int = 3) -> AssetEvaluation:
            Evaluates the asset based on the provided prompt.
            Args:
                prompt (str): The prompt to evaluate the asset.
                max_retries (int, optional): Maximum number of retries for the evaluation. Defaults to 3.
            Returns:
                AssetEvaluation: The evaluation result of the asset.
    """
    
    name = "claude-3-5-sonnet"
    code = "claude-3-5-sonnet-20240620"
    api_key = get_api_key()
    aliases = [
        "claude35",
        "claude3",
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

        client = instructor.from_anthropic(Anthropic(api_key=self.api_key))

        response, completion = client.chat.completions.create_with_completion(
            model=self.code,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
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
        client = instructor.from_anthropic(Anthropic(api_key=self.api_key))

        response, completion = client.chat.completions.create_with_completion(
            model=self.code,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "user", "content": prompt},
            ],
            response_model=AssetEvaluation,
            max_retries=max_retries,
            seed=self.seed,
        )
        return response