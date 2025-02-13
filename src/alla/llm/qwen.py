import instructor
from openai import OpenAI

from alla.llm.base import LLMBaseClass


class QwenCoder(LLMBaseClass):
    """
    Qwen-Coder model
    """
    name = "Qwen2.5-Coder 32B"
    code = "qwen2.5-coder:32b"
    path_code = "qwen2.5_coder_32b"
    # context_window = 128_000
    aliases = [
        "qwen-large",
        "qwen-32b",
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


class QwenCoderTiny(QwenCoder):
    """Tiny Qwen-Coder model"""
    name = "Qwen2.5-Coder 0.5B"
    code = "qwen2.5-coder:0.5b"
    path_code = "qwen2.5_coder_0.5b"
    aliases = [
        "qwen-tiny",
        "qwen-0.5b",
    ]
