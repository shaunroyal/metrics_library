"""Log Loss (cross-entropy) metric implementation.

Provides the `LogLoss` class implementing the Metric interface for binary classification.
"""

from typing import Union
import numpy as np

from ..core.base import Metric, MetricResult


class LogLoss(Metric):
    """Log Loss metric.

    Computes the binary cross-entropy loss given true labels and predicted probabilities.
    """

    def __init__(self) -> None:
        super().__init__(name="log_loss")

    def calculate(
        self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
    ) -> MetricResult:
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        if y_true_arr.shape != y_pred_arr.shape:
            raise ValueError("y_true and y_pred must have the same shape for LogLoss calculation.")
        # Clip predictions to avoid log(0)
        eps = np.finfo(float).eps
        y_pred_clipped = np.clip(y_pred_arr, eps, 1 - eps)
        # Binary cross-entropy
        loss = -np.mean(y_true_arr * np.log(y_pred_clipped) + (1 - y_true_arr) * np.log(1 - y_pred_clipped))
        return MetricResult(value=float(loss), name=self.name)
