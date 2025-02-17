# Probabilistic Stock Trading Agent

## Project Overview & Agent Description (PEAS)
This project aims to build a **goal-based probabilistic stock trading agent** using historical stock market data. Our agent leverages a Bayesian network to estimate market conditions and guide trading decisions. In PEAS terms:

- **Performance:** Maximize profit and control risk by predicting whether the next day's stock Close price will be higher than today's.
- **Environment:** The dynamic, uncertain world of the stock market, represented by historical data (Date, Open, High, Low, Close, Volume, etc.) from the "Stock Time Series 20050101 to 20171231" Kaggle dataset.
- **Actuators:** Trading actions such as buy, sell, or hold.
- **Sensors:** Historical market indicators and technical signals derived from the dataset.

The agent is goal-based, making decisions to achieve a specific objective (profitable trading) based on probabilistic predictions.

## Data Exploration & Preprocessing
### Data Source
- **Dataset:** [Stock Time Series 20050101 to 20171231](https://www.kaggle.com/datasets/szrlee/stock-time-series-20050101-to-20171231)
- **Description:** Over a decade of daily stock market data for various stocks, including fields for Date, Open, High, Low, Close, Volume, and the stock Name.
- **Observations:** The dataset contains approximately 93,612 observations.

### Exploratory Data Analysis (EDA)
- **Descriptive Statistics:**  
  Summary statistics (mean, median, standard deviation) were calculated for key features to understand their distributions.
- **Visualizations:**  
  Histograms, box plots, and time-series plots were generated to identify trends, seasonal patterns, outliers, and missing data.
- **Missing Data:**  
  Missing numerical values were imputed using the mean of each column.

### Preprocessing Steps
- **Normalization & Transformation:**  
  - Features were scaled using StandardScaler (to have mean=0 and standard deviation=1).
  - A logarithmic transformation was applied to reduce skewness in stock prices and volumes.
- **Discretization:**  
  Continuous numerical features (e.g., Open, High, Low) were discretized into quantile-based bins. For instance, the "Open" price was divided into 5 bins to create a new feature, `Open_binned`. Similar processes produced `High_binned` and `Low_binned`.  
  Mathematically, if \( X \) is a continuous variable (like the Open price), then the binned version \( X_{binned} \) is calculated by partitioning the range of \( X \) into quantile-based intervals. If 5 bins are used, each bin represents approximately 20% of the data, and the conditional probabilities for each bin (as generated in the CPTs) may be close to 0.2 if the distribution is uniform.
- **Conditional Probability Tables (CPTs):**  
  The frequency counts of observations in each bin were normalized to create CPTs. These tables summarize the probability distribution for each discretized feature, which can be used in the Bayesian network to model the uncertainty of each feature.

## Baseline Model
### Model Structure
Our baseline model is a Bayesian network where each selected binned feature serves as a parent node to the target variable, `Trend`. The structure is as follows:
[Open_binned] -> Trend, [High_binned] -> Trend, [Low_binned] -> Trend.
The network learns the conditional probability P(Trend|Open_binned, High_binned, Low_binned).

### Training & Evaluation
- **Training:**  
  The dataset was split into training (80%) and testing (20%) sets. The Bayesian network is trained using Maximum Likelihood Estimation (MLE) on the training data, which estimates the Conditional Probability Distributions (CPDs) for each node.
- **Inference:**  
  Inference is performed via Variable Elimination (VE) to compute
  P(Trend=1|features) for each test instance. A threshold of 0.5 is applied to generate binary predictions.
- **Evaluation Metrics:**  
  We evaluated the model using multiple metrics:
  - Accuracy, Precision, Recall, and F1 Score.
  - Confusion Matrix, ROC Curve, and ROC AUC.
  
  The current baseline model achieves an accuracy of approximately 51.1%, which is only marginally better than random guessing—this is expected given the inherent noise and complexity in financial data.

## Repository Structure & Collaboration
- **data/** – Contains raw and processed stock market data.
- **notebooks/** – Jupyter notebooks for EDA, preprocessing, and model training/evaluation.
- **scripts/** – Python scripts for data processing, model training (this file), and generating CPTs.
- **docs/** – Additional documentation and design notes (including this project's design rationale).
- **README.md** – This document, summarizing our work and providing links to code and notebooks.

## Conclusion & Future Work
### Conclusion
The baseline Bayesian network model demonstrates a fundamental approach to predicting whether the stock's Close price will increase the next day (Trend = 1). With an accuracy of around 51.1%, the model provides a starting point for understanding market dynamics probabilistically. The discretized features (Open_binned, High_binned, Low_binned) are used to capture key patterns in the data, though their current uniform distributions suggest that further feature engineering may be beneficial.

### Future Enhancements
- **Advanced Probabilistic Models:**  
  Integrate Hidden Markov Models (HMMs) to capture temporal dependencies and latent market regimes.
- **Reinforcement Learning (RL):**  
  Develop RL-based approaches to optimize trading decisions based on model predictions.
- **Improved Decision Making:**  
  Combine probabilistic predictions with a utility-based decision-making to directly maximize expected profit while controlling for risk.