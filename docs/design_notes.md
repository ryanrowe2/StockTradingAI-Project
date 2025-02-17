# Design Notes for Probabilistic Stock Trading Agent

## Overview
This project aims to build a goal-based probabilistic stock trading agent using historical market data. The agent leverages a Bayesian network to estimate market conditions and inform trading decisions. The ultimate goal is to develop an agent that can help guide trading actions by predicting whether the next day's stock "Close" price will be higher than today's.

## Data Exploration & Preprocessing
- **Data Source:**  
  The data comes from the "Stock Time Series 20050101 to 20171231" dataset on Kaggle. This dataset contains over a decade of daily stock market data for various stocks, including fields such as Date, Open, High, Low, Close, Volume, and the stock Name.
  
- **EDA:**  
  Exploratory data analysis was performed using descriptive statistics and visualizations:
  - **Descriptive Statistics:** Summary statistics (mean, median, standard deviation) were computed to understand the distribution of key features.
  - **Visualizations:** Histograms, box plots, and time-series plots were generated to identify trends, seasonal patterns, and outliers.
  
- **Preprocessing Steps:**  
  To prepare the data for probabilistic modeling, the following steps were applied:
  - **Missing Data Imputation:** Missing numerical values were filled using the mean of each column.
  - **Normalization:** Features were scaled using StandardScaler to standardize the data (mean=0, std=1).
  - **Logarithmic Transformation:** Log transformation was applied to reduce skewness in features such as stock prices and volumes.
  - **Discretization:** Continuous numerical features were discretized into quantile-based bins (e.g., generating new columns like `Open_binned`, `High_binned`, etc.).
  - **CPT Generation:** Frequency counts of the binned values were normalized to create Conditional Probability Tables (CPTs), summarizing the distribution of each feature in probabilistic terms.

## Baseline Model
- **Model Structure:**  
  The baseline model is a Bayesian network in which each selected binned feature serves as a parent node to the target variable, "Trend" (indicating whether the next day's Close price will be higher than today's). The simple structure is defined as:
[Feature 1] -> Trend [Feature 2] -> Trend [Feature 3] -> Trend

## Training & Evaluation:
- The dataset is split into training (80%) and testing (20%) sets.
- The model is trained on the training set using Maximum Likelihood Estimation.
- For evaluation, the model uses Variable Elimination to infer the probability that the "Trend" is 1 for each test instance.
- The accuracy is computed as the proportion of test instances where the predicted trend (using a threshold of 0.5) matches the actual trend.
- The current baseline model achieves an accuracy of approximately 51.1%, which is only marginally better than random guessing. This is expected for a simple baseline on noisy financial data.

## Future Work
- **Incorporate Hidden Markov Models (HMMs):**  
To better capture latent market regimes and temporal dependencies, HMMs could be integrated into the agent.

- **Integrate Reinforcement Learning (RL):**  
Future iterations may involve using RL to learn dynamic trading policies that optimize returns over time.

- **Enhance Feature Engineering:**  
Further improvements could involve exploring additional features, alternative discretization techniques, or other transformation methods to better capture the complexities of market data.

- **Alternative Modeling Approaches:**  
Experiment with more advanced probabilistic models or hybrid approaches to improve predictive performance.

## Final Thoughts
The baseline Bayesian network model provides a promising starting point for our probabilistic stock trading agent. While its performance is modest at around 51.1% accuracy, this is expected given the complexity and noise inherent in financial data. Future work will focus on integrating additional models (such as HMMs and RL) and refining the feature set and preprocessing methods to improve overall predictive performance and robustness.