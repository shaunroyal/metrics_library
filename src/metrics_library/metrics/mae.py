"""Mean Absolute Error metric implementation.

Provides the :class:`MeanAbsoluteError` class implementing the ``Metric`` interface.
"""

from typing import Union
import numpy as np

from ..core.base import Metric, MetricResult


class MeanAbsoluteError(Metric):
    """Mean Absolute Error (MAE) metric.

    MAE is the average of absolute differences between true and predicted values.

    Args:
        None.

    Returns:
        MetricResult: An object containing the MAE value and metric name.

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` have mismatched shapes.

    Example:
        >>> from metrics_library.metrics import MeanAbsoluteError
        >>> mae = MeanAbsoluteError()
        >>> result = mae.calculate([3, -0.5, 2, 7], [2.5, 0.0, 2, 8])
        >>> result.value
        0.5
    """

    def __init__(self) -> None:
        super().__init__(name="mean_absolute_error")

    def calculate(
        self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
    ) -> MetricResult:
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        if y_true_arr.shape != y_pred_arr.shape:
            raise ValueError(
                "y_true and y_pred must have the same shape for MAE calculation."
            )
        mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
        return MetricResult(value=mae, name=self.name)
