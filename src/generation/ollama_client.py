from __future__ import annotations

import json
import base64
from collections.abc import Iterator

import requests


class OllamaClient:
    def __init__(
        self,
        model: str = "qwen3:8b",
        vision_model: str = "qwen3-vl:4b",
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
    ) -> None:
        self.model = model
        self.vision_model = vision_model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def generate_url(self) -> str:
        return f"{self.base_url}/api/generate"

    def generate(self, prompt: str) -> str:
        response = requests.post(
            self.generate_url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,

                # Disable reasoning for fast grounded QA.
                "think": False,

                # Keep the model resident in memory.
                "keep_alive": -1,

                "options": {
                    "temperature": 0.1,
                    "num_predict": 512,
                },
            },
            timeout=self.timeout,
        )

        response.raise_for_status()

        payload = response.json()

        return str(payload.get("response", "")).strip()

    def analyze_image(self, prompt: str, image: bytes) -> str:
        response = requests.post(
            self.generate_url,
            json={
                "model": self.vision_model,
                "prompt": prompt,
                "images": [base64.b64encode(image).decode("ascii")],
                "stream": False,
                "think": False,
                "keep_alive": -1,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 512,
                },
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("response", "")).strip()

    def stream(
        self,
        prompt: str,
        image: bytes | None = None,
    ) -> Iterator[str]:
        payload = {
            "model": self.vision_model if image is not None else self.model,
            "prompt": prompt,
            "stream": True,
            "think": False,
            "keep_alive": -1,
            "options": {
                "temperature": 0.1,
                "num_predict": 512,
            },
        }
        if image is not None:
            payload["images"] = [base64.b64encode(image).decode("ascii")]

        response = requests.post(
            self.generate_url,
            json=payload,
            stream=True,
            timeout=self.timeout,
        )

        response.raise_for_status()

        for line in response.iter_lines(
            decode_unicode=True
        ):
            if not line:
                continue

            payload = json.loads(line)

            chunk = payload.get("response", "")

            if chunk:
                yield str(chunk)

            if payload.get("done"):
                break
