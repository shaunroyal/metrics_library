import numpy as np
import pytest

from metrics_library.metrics import (
    MeanSquaredError,
    MeanAbsoluteError,
    Accuracy,
    F1Score,
    LogLoss,
)
from metrics_library.genai.description_generator import DescriptionGenerator
from metrics_library.core.base import MetricResult


@pytest.fixture
def y_true_regression():
    return np.array([3.0, -0.5, 2.0, 7.0])

@pytest.fixture
def y_pred_regression():
    return np.array([2.5, 0.0, 2.0, 8.0])

@pytest.fixture
def y_true_classification():
    return np.array([1, 0, 1, 1, 0, 0])

@pytest.fixture
def y_pred_classification():
    return np.array([1, 0, 0, 1, 0, 1])

@pytest.fixture
def y_pred_proba():
    # Probabilities for positive class
    return np.array([0.9, 0.2, 0.8, 0.7, 0.1, 0.3])


def test_mean_squared_error(y_true_regression, y_pred_regression):
    mse = MeanSquaredError()
    result = mse.calculate(y_true_regression, y_pred_regression)
    expected = np.mean((y_true_regression - y_pred_regression) ** 2)
    assert isinstance(result, MetricResult)
    assert result.name == "mean_squared_error"
    assert pytest.approx(result.value, rel=1e-6) == expected


def test_mean_absolute_error(y_true_regression, y_pred_regression):
    mae = MeanAbsoluteError()
    result = mae.calculate(y_true_regression, y_pred_regression)
    expected = np.mean(np.abs(y_true_regression - y_pred_regression))
    assert result.name == "mean_absolute_error"
    assert pytest.approx(result.value, rel=1e-6) == expected


def test_accuracy(y_true_classification, y_pred_classification):
    acc = Accuracy()
    result = acc.calculate(y_true_classification, y_pred_classification)
    expected = np.mean(y_true_classification == y_pred_classification)
    assert result.name == "accuracy"
    assert pytest.approx(result.value, rel=1e-6) == expected


def test_f1_score(y_true_classification, y_pred_classification):
    f1 = F1Score()
    result = f1.calculate(y_true_classification, y_pred_classification)
    tp = np.sum((y_true_classification == 1) & (y_pred_classification == 1))
    fp = np.sum((y_true_classification == 0) & (y_pred_classification == 1))
    fn = np.sum((y_true_classification == 1) & (y_pred_classification == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    expected = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    assert result.name == "f1_score"
    assert pytest.approx(result.value, rel=1e-6) == expected


def test_log_loss(y_true_classification, y_pred_proba):
    logloss = LogLoss()
    result = logloss.calculate(y_true_classification, y_pred_proba)
    eps = np.finfo(float).eps
    y_pred_clipped = np.clip(y_pred_proba, eps, 1 - eps)
    expected = -np.mean(
        y_true_classification * np.log(y_pred_clipped)
        + (1 - y_true_classification) * np.log(1 - y_pred_clipped)
    )
    assert result.name == "log_loss"
    assert pytest.approx(result.value, rel=1e-6) == expected


def test_description_generator(y_true_regression, y_pred_regression):
    mse = MeanSquaredError()
    result = mse.calculate(y_true_regression, y_pred_regression)
    generator = DescriptionGenerator()
    description = generator.generate(result)
    # Should contain the metric name and the numeric value
    assert "Mean Squared Error" in description or "mean_squared_error" in description
    assert f"{result.value:.4f}" in description
