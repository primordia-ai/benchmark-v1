import json
import os

import google.generativeai as genai
import instructor

from alla.llm.base import LLMBaseClass
from alla.utils import PROJECT_ROOT


def get_api_key():
    secrets_path = os.path.join(PROJECT_ROOT, ".secrets.json")
    with open(secrets_path, "r") as f:
        secrets = json.load(f)
    api_key = secrets.get("GEMINI_API_KEY")
    return api_key


class Gemini15(LLMBaseClass):
    """
    Gemini15 is a class that extends LLMBaseClass and provides functionality for interacting with the Gemini 1.5 model.
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

    def init_client(self) -> None:
        self._client = instructor.from_gemini(
            client=genai.GenerativeModel(
                model_name=self.code,
            ),
            mode=instructor.Mode.GEMINI_JSON,
        )


class Gemini2(Gemini15):
    """
    Gemini2 class inherits from Gemini15 and represents the Gemini 2.0 model.
    Attributes:
        name (str): The name of the model.
        code (str): The code identifier for the model.
        api_key (str): The API key for accessing the model.
        aliases (list): A list of alternative names for the model.
    """

    name = "gemini_2.0"
    code = "models/gemini-2.0-flash-exp"
    api_key = get_api_key()
    aliases = [
        "gemini2",
        "gemini-2",
    ]
