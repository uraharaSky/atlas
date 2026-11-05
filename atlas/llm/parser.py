"""Parse and validate structured research specifications returned by an LLM."""

from __future__ import annotations

import json

from pydantic import ValidationError

from atlas.llm.schemas import ResearchSpec


class ResearchSpecParser:
    """Convert raw LLM output into a validated ResearchSpec."""

    def parse(self, response: str) -> ResearchSpec:
        """Parse an LLM response containing a JSON research specification.

        Args:
            response: Raw text returned by an LLM.

        Returns:
            A validated ResearchSpec instance.

        Raises:
            ValueError: If the response is empty or does not contain valid JSON.
            ValidationError: If the JSON does not match ResearchSpec.
        """

        if not response.strip():
            raise ValueError("LLM response must not be empty.")

        data = self._extract_json(response)

        try:
            return ResearchSpec.model_validate(data)
        except ValidationError:
            raise

    @staticmethod
    def _extract_json(response: str) -> dict:
        """Extract a JSON object from an LLM response.

        The model may return either plain JSON or JSON wrapped in
        a Markdown code block.
        """

        text = response.strip()

        if text.startswith("```"):
            lines = text.splitlines()

            if len(lines) < 3:
                raise ValueError("Invalid JSON code block.")

            # Remove opening and closing Markdown fences.
            text = "\n".join(lines[1:-1]).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM response does not contain valid JSON.") from exc

        if not isinstance(data, dict):
            raise ValueError("LLM response must contain a JSON object.")

        return data
