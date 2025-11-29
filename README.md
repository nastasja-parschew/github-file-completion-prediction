# Prediction of Files' Completion Times of GitHub Open-Source Repositories
This repository contains the research code for predicting file completion times in open source projects. It was developed 
as part of the master's thesis named _"Event and Data Analytics for Predicting Development Task Completion through File
Evolution Trends"_ by Nastasja Stephanie Parschew.

The goal of this project is to estimate how long a source code file will remain active before it is 
considered completed (stable, deleted, or abandoned). It features an end-to-end machine learning pipeline 
that fetches raw commit history from GitHub, engineers temporal and code-metric features, and trains 
five machine learning models with optional hyperparameter optimization.

## Key Features
* **Automated Data Mining:** Asynchronously fetches commit history and file metadata via the GitHub API with 
rate-limit handling.
* **Dynamic Labeling:** Algorithms to automatically label files as "completed" based on line-change stability, 
deletion events, or project inactivity.
* **Feature Engineering:** Generates features based on file size, path depth, commit frequency, and author history.
* **Hyperparameter Tuning:** Uses Optuna for hyperparameter tuning on models like XGBoost, LightGBM, and Random Forest.
* **Explainability:** built-in generation of SHAP plots and Partial Dependence Plots (PDP) to analyze model decisions.

## Repository layout
- **src/** – implementation of data collection, feature engineering and modelling
- **config/** – configuration files with project definitions and credentials
- **tests/** – unit tests for some of the components
- **environment.yml** – conda environment specification with all required dependencies

## Prerequisites
This project requires:
* **Python 3.11+**
* **MongoDB:** A running instance is required to store commit histories and engineered features.

## Installation
Create a conda environment using the provided environment file:

```bash
conda env create -f environment.yml
conda activate completiontimes
```

The configuration in `config/config.yml` requires a GitHub access token for downloading repository data. 
Provide your token via the `GITHUB_TOKEN` environment variable, for example by creating a `.env` file based on 
`config/example.env` and setting `GITHUB_TOKEN` accordingly.

## Running the pipeline

The main entry point is `src/main.py`. It processes the repositories defined in `config/config.yml`, performs feature 
engineering and trains the configured models:

```bash
python src/main.py
```

Results (trained models and visualisations) are written to timestamped folders under `runs/`.

## Logging
Logging is configured via `src/logging_config.py` and uses the following defaults:

- **Level**: `DEBUG` (can be overridden when calling `setup_logging`)
- **Format**: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

Noise from third-party libraries such as `matplotlib` and `pymongo` is suppressed by
setting their log levels to `WARNING`.

When creating new entry points or standalone scripts, initialise logging by calling:

```python
from src.logging_config import setup_logging

setup_logging()
```