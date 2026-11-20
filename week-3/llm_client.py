import os
import time
import requests

from dataclasses import dataclass
from abc import ABC, abstractmethod

from google import genai
from google.genai import types


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int


class LLMProvider(ABC):

    @abstractmethod
    def generate(self, prompt):
        pass


# Gemini Provider with automatic fallback
class GeminiProvider(LLMProvider):

    provider_name = "Gemini"

    def __init__(self, api_key, model="gemini-3.8-flash"):

        self.model = model

        self.models = [
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.5-flash-lite"
        ]

        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=120000,
                retry_options=types.HttpRetryOptions(
                    attempts=1
                )
            )
        )

    def generate(self, prompt):

        last_error = None

        for model in self.models:

            try:
                print(f"\nTrying Gemini model: {model}")

                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=512
                    )
                )

                usage = response.usage_metadata

                input_count = getattr(
                    usage, "prompt_token_count", 0
                ) or 0

                output_count = getattr(
                    usage, "candidates_token_count", 0
                ) or 0

                if not response.text:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                self.model = model

                print(f"Gemini model selected: {model}")

                return LLMResult(
                    text=response.text,
                    input_tokens=input_count,
                    output_tokens=output_count
                )

            except Exception as error:

                last_error = error

                status = getattr(error, "code", None)

                if status is None:
                    status = getattr(error, "status_code", None)

                error_name = type(error).__name__.lower()

                retryable = (
                    status in (404, 429, 500, 502, 503, 504)
                    or "timeout" in error_name
                    or "unavailable" in error_name
                    or "deadline" in error_name
                    or "resourceexhausted" in error_name
                )

                if not retryable:
                    raise

                print(
                    f"Model {model} unavailable: {error}"
                )

                if model != self.models[-1]:
                    print("Switching to the next backup model...")

        raise last_error


# Ollama Provider
class OllamaProvider(LLMProvider):

    provider_name = "Ollama"

    def __init__(self, model="llama3.2"):

        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt):

        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            },
            timeout=(5, 120)
        )

        response.raise_for_status()

        data = response.json()

        return LLMResult(
            text=data.get("response", ""),
            input_tokens=data.get("prompt_eval_count", 0) or 0,
            output_tokens=data.get("eval_count", 0) or 0
        )


# Common LLM Client
class LLMClient:

    def __init__(self, provider, usage_logger):

        self.provider = provider
        self.usage_logger = usage_logger

    def generate(self, prompt):

        start = time.perf_counter()
        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                result = self.provider.generate(prompt)
                break
            except requests.RequestException:
                if attempt == max_attempts - 1:
                    raise
                time.sleep(2 ** attempt)

        elapsed_seconds = time.perf_counter() - start
        self.usage_logger.log_call({
            "provider": getattr(
                self.provider, "provider_name", type(self.provider).__name__
            ),
            "model": getattr(self.provider, "model", None),
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "total_tokens": result.input_tokens + result.output_tokens,
            "elapsed_seconds": elapsed_seconds,
            "status": "success",
        })

        return result.text