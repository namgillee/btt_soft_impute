# Data Analysis Tool

A lightweight helper tool for data analysts using pandas.

## Features

- Quick dataframe summaries
- Missing value checks
- Data type overview

## Installation

```bash
pip install git+https://github.com/namgillee/BTT-SCD```

## Usage

```python
from data_analysis_tool import core
import pandas as pd

df = pd.read_csv('your_data.csv')
report = core.summarize_dataframe(df)
print(report)
```
