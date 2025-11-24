"""Mean Squared Error metric implementation.

This module provides the `MeanSquaredError` class, a concrete implementation of
the :class:`~metrics_library.core.base.Metric` abstract base class.
"""

from typing import Union
import numpy as np

from ..core.base import Metric, MetricResult


class MeanSquaredError(Metric):
    """Mean Squared Error (MSE) metric.

    The MSE is defined as the average of the squared differences between the
    true values and the predicted values.
    """

    def __init__(self) -> None:
        super().__init__(name="mean_squared_error")

    def calculate(
        self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
    ) -> MetricResult:
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        if y_true_arr.shape != y_pred_arr.shape:
            raise ValueError("y_true and y_pred must have the same shape for MSE calculation.")
        mse = float(np.mean((y_true_arr - y_pred_arr) ** 2))
        return MetricResult(value=mse, name=self.name)
