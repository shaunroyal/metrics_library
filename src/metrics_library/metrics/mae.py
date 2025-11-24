"""Mean Absolute Error metric implementation.

Provides the `MeanAbsoluteError` class implementing the Metric interface.
"""

from typing import Union
import numpy as np

from ..core.base import Metric, MetricResult


class MeanAbsoluteError(Metric):
    """Mean Absolute Error (MAE) metric.

    MAE is the average of absolute differences between true and predicted values.
    """

    def __init__(self) -> None:
        super().__init__(name="mean_absolute_error")

    def calculate(
        self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
    ) -> MetricResult:
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        if y_true_arr.shape != y_pred_arr.shape:
            raise ValueError("y_true and y_pred must have the same shape for MAE calculation.")
        mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
        return MetricResult(value=mae, name=self.name)
