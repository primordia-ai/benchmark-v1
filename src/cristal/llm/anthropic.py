import json
import os

import instructor
from anthropic import Anthropic

from cristal.llm.base import LLMBaseClass
from cristal.env import PROJECT_ROOT


def get_api_key() -> str | None:
    """
    Retrieves the Anthropic API key from the ANTHROPIC_API_KEY environment variable,
    or from a .secrets.json file located in the project root directory.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        return api_key
    secrets_path = os.path.join(PROJECT_ROOT, ".secrets.json")
    if os.path.exists(secrets_path):
        with open(secrets_path, "r") as f:
            secrets = json.load(f)
        return secrets.get("ANTHROPIC_API_KEY")
    return None


class AnthropicModel(LLMBaseClass):
    """
    Anthropic class for interacting with the Claude-3-5-Sonnet model.
    """
    name = "Claude 3.5 Sonnet"
    code = "claude-3-5-sonnet-20240620"
    path_code = "claude3_5_sonnet_20240620"
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

    def init_client(self):
        """Init the client and cache it"""
        self._client = instructor.from_anthropic(Anthropic(api_key=self.api_key))

    def process_asset(self, asset):
        pass
