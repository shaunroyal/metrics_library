"""Accuracy metric implementation.

Provides the :class:`Accuracy` class implementing the ``Metric`` interface for classification tasks.
"""

from typing import Union
import numpy as np

from ..core.base import Metric, MetricResult


class Accuracy(Metric):
    """Accuracy metric.

    Accuracy is the proportion of correct predictions.

    Args:
        None.

    Returns:
        MetricResult: An object containing the accuracy value and metric name.

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` have mismatched shapes.

    Example:
        >>> from metrics_library.metrics import Accuracy
        >>> acc = Accuracy()
        >>> result = acc.calculate([1, 0, 1, 1], [1, 0, 0, 1])
        >>> result.value
        0.75
    """

    def __init__(self) -> None:
        super().__init__(name="accuracy")

    def calculate(
        self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
    ) -> MetricResult:
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        if y_true_arr.shape != y_pred_arr.shape:
            raise ValueError(
                "y_true and y_pred must have the same shape for Accuracy calculation."
            )
        correct = np.sum(y_true_arr == y_pred_arr)
        accuracy = float(correct) / y_true_arr.size
        return MetricResult(value=accuracy, name=self.name)
