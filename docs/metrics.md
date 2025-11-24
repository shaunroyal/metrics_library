# Metrics

This library provides the following metric implementations:

## Mean Squared Error (`MeanSquaredError`)
Calculates the average of the squared differences between true and predicted values.

```python
from metrics_library.metrics import MeanSquaredError
mse = MeanSquaredError()
result = mse.calculate(y_true, y_pred)
```

## Mean Absolute Error (`MeanAbsoluteError`)
Calculates the average absolute difference between true and predicted values.

```python
from metrics_library.metrics import MeanAbsoluteError
mae = MeanAbsoluteError()
result = mae.calculate(y_true, y_pred)
```

## Accuracy (`Accuracy`)
Proportion of correct predictions for classification tasks.

```python
from metrics_library.metrics import Accuracy
acc = Accuracy()
result = acc.calculate(y_true, y_pred)
```

## F1 Score (`F1Score`)
Harmonic mean of precision and recall for binary classification.

```python
from metrics_library.metrics import F1Score
f1 = F1Score()
result = f1.calculate(y_true, y_pred)
```

## Log Loss (`LogLoss`)
Binary cross‑entropy loss for classification probabilities.

```python
from metrics_library.metrics import LogLoss
logloss = LogLoss()
result = logloss.calculate(y_true, y_pred_proba)
```

Each metric returns a `MetricResult` object containing the `value`, `name`, and optional `metadata`.
