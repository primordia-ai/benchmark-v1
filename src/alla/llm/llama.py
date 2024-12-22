import json
import logging
import os

import instructor
from openai import OpenAI
from tqdm import tqdm

from alla.llm import utils as llm_utils
from alla.llm.base import LLMBaseClass
from alla.utils import PROJECT_ROOT

logger = logging.getLogger("alla")


class Llama(LLMBaseClass):
    """
    Llama model
    """
    name = "Llama"
    code = 'llama3.1:8b'


    def __init__(self, temperature: float = 0.0):
        super().__init__(temperature)

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
            reports_text = self.read_reports(report_dir, include, exclude)

        if verbose is True:
            tqdm.write(f'Inferring value from report for asset_id {asset_id} for {indicator_name}')

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
        response = client.chat.completions.create(
            model=self.code,
            temperature=self.temperature,
            messages=[
                {"role": "user", "content": prompt},
            ],
            response_model=self._create_indicator_class(indicator_name),
        )

        return response.model_dump()
