"""LLM client abstractions for ATLAS."""

from __future__ import annotations

from dataclasses import dataclass

import requests




@dataclass
class OllamaClient:
    """Client for communicating with a local Ollama server."""

    model: str = "llama3:latest"
    base_url: str = "http://localhost:11434"

    def generate(self, prompt: str) -> str:
        """Generate a response from the configured Ollama model."""

        if not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )

        response.raise_for_status()

        data = response.json()

        return data["response"]

