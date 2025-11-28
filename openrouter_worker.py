"""Helpers for running OpenRouter completion calls inside multiprocessing workers."""

from typing import Tuple

from openai import OpenAI

_WORKER_CLIENT = None
_WORKER_MODEL = None


def init_openrouter_worker(base_url: str, api_key: str, model_name: str) -> None:
    """Initializer executed in each worker process to configure the OpenRouter client."""
    global _WORKER_CLIENT, _WORKER_MODEL
    _WORKER_CLIENT = OpenAI(base_url=base_url, api_key=api_key)
    _WORKER_MODEL = model_name


def run_openrouter_completion(task: Tuple[str, int, float]) -> str:
    """Execute a single completion request using the worker-local client."""
    prompt, max_tokens, temperature = task

    response = _WORKER_CLIENT.completions.create(
        model=_WORKER_MODEL,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False
    )
    return response.choices[0].text
