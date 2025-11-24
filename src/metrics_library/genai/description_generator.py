"""GenAI description generator for metrics.

This module provides a simple class that uses the :class:`PromptManager` to
retrieve a description template for a metric and then formats it with the
calculated result. In a real‑world scenario this could be replaced with a call
to a large language model, but the interface is kept generic so that the
implementation can be swapped out without changing the metric classes.
"""

from typing import Any

from .prompts import PromptManager
from ..core.base import MetricResult


class DescriptionGenerator:
    """Generates a natural‑language description for a metric result.

    The generator loads a prompt template for the metric name from the YAML
    prompt file (see ``data/prompts.yaml``) and formats it with the ``value``
    attribute of a :class:`MetricResult`. If no template is found, it falls
    back to a generic description.
    """

    def __init__(self, prompt_file: str | None = None):
        self._prompt_manager = PromptManager(prompt_file=prompt_file)

    def generate(self, result: MetricResult) -> str:
        """Return a description string for *result*.

        Args:
            result: The :class:`MetricResult` containing the metric name and
                calculated value.

        Returns:
            A human‑readable description.
        """
        template = self._prompt_manager.get_template(result.name)
        if template:
            try:
                return template.format(value=result.value)
            except Exception:
                # If formatting fails, fall back to generic description
                pass
        # Generic fallback
        return f"The {result.name} metric value is {result.value:.4f}."
