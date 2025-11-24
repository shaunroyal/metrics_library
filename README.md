# Metrics Library

A suite of modelling metric functions designed for scalability and ease of use, featuring Gen AI integration for automated metric descriptions.

## Features

- **Standard Metrics**: Implementations of common regression and classification metrics.
- **Gen AI Integration**: Generate natural language descriptions of your metric results using LLMs.
- **Prompt Management**: Manage prompts via local YAML or MLflow.
- **Google Style Guide**: Codebase adheres to Google's Python style guide.

## Installation

```bash
pip install .
```

## Usage

```python
from metrics_library.metrics.regression import MeanSquaredError

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

mse = MeanSquaredError()
result = mse.calculate(y_true, y_pred)
print(f"MSE: {result.value}")

# Generate description (requires configuration)
# description = mse.get_description(result)
# print(description)
```
