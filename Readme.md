# mtcars Linear Regression Analysis

This repository contains an automated workflow that mirrors the requested R-based analysis while providing a Python fallback that can be executed in restricted environments (such as this container, which does not have R installed).

## Contents

- `analysis.R` – primary R script that performs data preparation, linear regression modelling, metric calculation, and visualization export using the **mtcars** dataset.
- `run_analysis.py` – pure-Python implementation of the same workflow (no third-party dependencies). It generates all KPI outputs when `Rscript` is unavailable.
- `generate_reports.py` – builds formatted English and German reports (`.doc` via RTF) containing KPI metrics, accuracy results, visualizations references, and model details.
- `config/train_indices.csv` – deterministic training indices shared between the R and Python implementations to keep results reproducible.
- `output/` – generated artifacts (metrics, KPI summary, predictions, model summary, visualization SVG).
- `reports/` – generated report documents.

## Running the workflow

1. **Preferred (R) path**
   ```bash
   Rscript analysis.R
   ```
   > Installs `ggplot2` if required, then writes artefacts to the `output/` directory.

2. **Fallback (Python) path**
   ```bash
   python run_analysis.py
   python generate_reports.py
   ```
   > Produces the same metrics/artefacts without requiring external Python packages.

## Generated outputs

- `output/metrics.csv` – RMSE, MAE, and R-squared on the hold-out set.
- `output/kpi_summary.csv` – overall dataset KPI summary.
- `output/predictions.csv` – comparison of actual vs. predicted MPG for the test split.
- `output/model_summary.txt` – model coefficients and evaluation statistics.
- `output/actual_vs_predicted.svg` – scatter plot visualization.
- `reports/mtcars_regression_report_en.doc` – English report.
- `reports/mtcars_regressionsbericht_de.doc` – German report.

Both reports contain the computed KPIs, model accuracy, and references to the generated visualization.
