"""High-level research methodology extraction using an LLM."""

from __future__ import annotations

from atlas.llm.client import OllamaClient
from atlas.llm.parser import ResearchSpecParser
from atlas.llm.prompts import build_research_spec_prompt
from atlas.llm.schemas import ResearchSpec


class ResearchExtractor:
    """Extract a structured research specification from paper text."""

    def __init__(
        self,
        client: OllamaClient | None = None,
        parser: ResearchSpecParser | None = None,
    ) -> None:
        self.client = client or OllamaClient()
        self.parser = parser or ResearchSpecParser()

    def extract(self, paper_text: str) -> ResearchSpec:
        """Extract and validate a ResearchSpec from paper text."""

        if not paper_text.strip():
            raise ValueError("Paper text must not be empty.")

        prompt = build_research_spec_prompt(paper_text)

        response = self.client.generate(prompt)

        return self.parser.parse(response)