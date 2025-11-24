# Metrics Library

Welcome to the **Metrics Library**! This package provides a suite of modelling metric functions with optional Gen AI generated descriptions.

## Features

- **Metric implementations**: MSE, MAE, Accuracy, F1 Score, Log Loss.
- **Gen AI description**: Generate natural‑language explanations for metric results.
- **Scalable design**: Easy to extend with new metrics and prompts.
- **Fully tested**: 90%+ test coverage.
- **Documentation**: Hosted on GitHub Pages via MkDocs.

## Installation

```bash
pip install metrics_library
```

## Quick Start

```python
from metrics_library.metrics import MeanSquaredError
from metrics_library.genai.description_generator import DescriptionGenerator

mse = MeanSquaredError()
result = mse.calculate([3, -0.5, 2, 7], [2.5, 0.0, 2.0, 8.0])

desc = DescriptionGenerator().generate(result)
print(desc)
```
