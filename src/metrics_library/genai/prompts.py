"""Prompt management for Gen AI descriptions.

This module handles loading and managing prompts used for generating
metric descriptions.
"""

import os
from typing import Any, Dict, Optional

import yaml


class PromptManager:
    """Manages prompts for Gen AI descriptions.

    This class handles loading prompts from a local YAML file or other sources.
    """

    def __init__(self, prompt_file: Optional[str] = None):
        """Initializes the PromptManager.

        Args:
            prompt_file: Path to the YAML file containing prompts. If None,
                defaults to the package's internal prompts.yaml.
        """
        if prompt_file is None:
            # Default to the package's data directory
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            prompt_file = os.path.join(base_dir, "data", "prompts.yaml")

        self._prompts = self._load_prompts(prompt_file)

    def _load_prompts(self, prompt_file: str) -> Dict[str, Any]:
        """Loads prompts from a YAML file.

        Args:
            prompt_file: Path to the YAML file.

        Returns:
            A dictionary containing the prompts.
        """
        try:
            with open(prompt_file, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            # Fallback or log warning
            return {}

    def get_template(self, metric_name: str) -> Optional[str]:
        """Retrieves the description template for a given metric.

        Args:
            metric_name: The name of the metric (snake_case).

        Returns:
            The description template string, or None if not found.
        """
        metric_data = self._prompts.get("metrics", {}).get(metric_name)
        if metric_data:
            return metric_data.get("description_template")
        return None
