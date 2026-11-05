"""LLM client abstractions for ATLAS."""

from __future__ import annotations

from dataclasses import dataclass

import requests


@dataclass
class OllamaClient:
    """Client for communicating with a local Ollama server."""

    model: str = "llama3:latest"
    base_url: str = "http://localhost:11434"
    timeout: int = 120

    def generate(self, prompt: str) -> str:
        """Generate a JSON response from the configured Ollama model.

        Args:
            prompt: Prompt to send to the model.

        Returns:
            Raw JSON response returned by Ollama.

        Raises:
            ValueError: If the prompt is empty.
            requests.HTTPError: If Ollama returns an HTTP error.
            requests.RequestException: If the request fails.
        """

        if not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "format": "json",
                "stream": False,
            },
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        return data["response"]