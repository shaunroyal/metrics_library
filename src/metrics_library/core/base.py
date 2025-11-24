"""Base classes for metrics library.

This module defines the abstract base class `Metric` and the `MetricResult`
dataclass that all metrics in the library should use.
"""

import abc
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np


@dataclass
class MetricResult:
    """Container for the result of a metric calculation.

    Attributes:
        value: The calculated metric value.
        name: The name of the metric.
        metadata: Additional metadata about the calculation.
    """

    value: float
    name: str
    metadata: Optional[Dict[str, Any]] = None


class Metric(abc.ABC):
    """Abstract base class for all metrics.

    All custom metrics should inherit from this class and implement the
    `calculate` method.
    """

    def __init__(self, name: str):
        """Initializes the Metric.

        Args:
            name: The name of the metric.
        """
        self._name = name

    @property
    def name(self) -> str:
        """Returns the name of the metric."""
        return self._name

    @abc.abstractmethod
    def calculate(
        self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
    ) -> MetricResult:
        """Calculates the metric.

        Args:
            y_true: Ground truth (correct) target values.
            y_pred: Estimated targets as returned by a classifier or regressor.

        Returns:
            A MetricResult object containing the calculated value.
        """
        pass

    def get_description(self, result: MetricResult) -> str:
        """Generates a description for the metric result.

        This method is intended to be overridden or used in conjunction with
        the Gen AI components to produce a natural language description.

        Args:
            result: The result of the metric calculation.

        Returns:
            A string description of the result.
        """
        return f"The {self.name} is {result.value}."
