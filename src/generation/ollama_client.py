from __future__ import annotations

from typing import Any

import requests


class OllamaClient:
    """Small local client for the Ollama generation API."""

    def __init__(
        self,
        model: str = "qwen3:8b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
    ) -> str:
        """Generate text from the local Ollama model."""
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                },
            },
            timeout=self.timeout,
        )

        response.raise_for_status()

        data: dict[str, Any] = response.json()

        answer = data.get("response", "").strip()

        if not answer:
            raise RuntimeError("Ollama returned an empty response.")

        return answer