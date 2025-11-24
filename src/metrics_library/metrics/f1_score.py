"""F1 Score metric implementation.

Provides the :class:`F1Score` class implementing the ``Metric`` interface for binary classification.
"""

from typing import Union
import numpy as np

from ..core.base import Metric, MetricResult


class F1Score(Metric):
    """F1 Score metric.

    The F1 Score is the harmonic mean of precision and recall.

    Args:
        None.

    Returns:
        MetricResult: An object containing the F1 score value and metric name.

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` have mismatched shapes.

    Example:
        >>> from metrics_library.metrics import F1Score
        >>> f1 = F1Score()
        >>> result = f1.calculate([1, 0, 1, 1, 0, 0], [1, 0, 0, 1, 0, 1])
        >>> result.value
        0.6666666666666666
    """

    def __init__(self) -> None:
        super().__init__(name="f1_score")

    def _precision_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
        """Calculate precision and recall for binary classification.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.

        Returns:
            A tuple of (precision, recall).
        """
        # Assuming binary classification with labels 0 and 1
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return precision, recall

    def calculate(
        self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
    ) -> MetricResult:
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        if y_true_arr.shape != y_pred_arr.shape:
            raise ValueError(
                "y_true and y_pred must have the same shape for F1Score calculation."
            )
        precision, recall = self._precision_recall(y_true_arr, y_pred_arr)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return MetricResult(value=f1, name=self.name)
