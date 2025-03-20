import instructor
from openai import OpenAI

from cristal.llm.base import LLMBaseClass


class MistralSmall(LLMBaseClass):
    """
    Mistral Small 3 model
    """
    name = "Mistral Small 3"
    code = "mistral-small"
    context_window = 32_000
    path_code = "mistral_small"
    # context_window = 128_000
    aliases = [
        "mistral-24b",
    ]

    def __init__(self, temperature: float = 0.0, seed: int = 101010):
        super().__init__(
            temperature=temperature,
            seed=seed
        )

    def process_asset(self, asset):
        pass

    def init_client(self) -> None:
        self._client = instructor.from_openai(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # required, but unused
            ),
            mode=instructor.Mode.JSON,
        )


class MistralNemo(MistralSmall):
    """Smaller Mistral model"""
    name = "Mistral Nemo 3"
    code = "mistral-nemo"
    context_window = 128_000
    path_code = "mistral_nemo"
    aliases = [
        "mistral-12b",
    ]
