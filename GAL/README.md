# GAL-TIR: Guided Active Learning for TIR Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Overview
This repository provides the official implementation of the **Guided Active Learning (GAL)** framework. GAL is designed to navigate complex, high-dimensional fitness landscapes in synthetic biology, specifically for optimizing **Translation Initiation Regions (TIR)** with minimal experimental samples. 

By integrating biophysical constraints (e.g., mRNA folding energy via ViennaRNA) with advanced machine learning architectures (XGBoost, GPR, and DNN Ensembles), GAL significantly accelerates the Design-Build-Test-Learn (DBTL) cycle in microbial engineering.

## Directory Structure
```text
spacer/
├── features/           # Feature engineering 
│   ├── NT-v2.py
│   ├── one-hot.py
│   └── ViennaRNA.py
├── scripts/            # Model training & Active Learning strategies
│   ├── DNN ensemble.py
│   ├── Xgboost ensemble.py
│   ├── GPR.py
│   └── Exploration.py  # Main entry for GAL exploration
├── visualization/      # Scripts to reproduce manuscript figures
│   ├── sampling_figure.py
│   ├── fluorescence.py
│   └── ...
├── data/              
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation