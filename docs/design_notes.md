# Design Notes for Probabilistic Stock Trading Agent

## Overview
This project aims to build a goal-based probabilistic stock trading agent using historical market data. The agent employs a Bayesian network to estimate market conditions and inform trading decisions.

## Data Exploration & Preprocessing
- **Data Source:** Stock Time Series 20050101 to 20171231 from Kaggle.
- **EDA:** Performed using descriptive statistics and visualizations (histograms, box plots, time series plots).
- **Preprocessing:**  
  - Missing data imputation via mean/mode.
  - Scaling with StandardScaler.
  - Logarithmic transformation to reduce skewness.
  - Discretization into quantile-based bins.
- **CPT Generation:**  
  - For each numerical feature, frequency counts of binned values are computed and normalized to produce CPTs.

## Baseline Model
- **Model Structure:**  
  A Bayesian network where selected binned features serve as parent nodes to the target variable ("Trend"). The target indicates whether the market (as approximated by the 'Close' price) will go up (1) or not (0) the following day.
- **Training & Evaluation:**  
  - Data is split into training (80%) and testing (20%) sets.
  - The model is trained using Maximum Likelihood Estimation.
  - Evaluation is performed using accuracy as the primary metric.
  
## Future Work
- Extend the model to incorporate HMMs and RL for improved decision-making.
- Enhance feature engineering and explore alternative discretization techniques.
- Evaluate additional performance metrics, including risk-adjusted returns.
