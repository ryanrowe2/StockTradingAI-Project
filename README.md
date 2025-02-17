# Probabilistic Stock Trading Agent

## Project Overview & Agent Description (PEAS)
This project develops a **goal-based probabilistic stock trading agent** designed to maximize profit while controlling risk. The agent operates in the dynamic world of the stock market, where its **sensors** are historical market indicators (e.g., prices, trading volumes, technical signals), and its **actuators** are trading actions (buy, sell, hold). **Performance** is measured via risk-adjusted returns and overall profit. Initially, our agent employs a Bayesian network-based model to estimate market conditions, forming the foundation for later integration of Hidden Markov Models (HMMs) and Reinforcement Learning (RL) to further refine trading decisions.

## Data Exploration & Preprocessing
### Dataset Overview
- **Dataset:** Stock Time Series 20050101 to 20171231 (Kaggle)
- **Link:** [https://www.kaggle.com/datasets/szrlee/stock-time-series-20050101-to-20171231](https://www.kaggle.com/datasets/szrlee/stock-time-series-20050101-to-20171231)
- **Description:** Contains over a decade of historical stock market data including daily open, high, low, close prices, and volumes.
- **Observations & Features:**  
  - *Observations:* Approximately [INSERT NUMBER]
  - *Key Features:* Open, High, Low, Close, Volume, and technical indicators.
- **Data Characteristics:**  
  - Varied scales and distributions.
  - Some missing data (imputed via mean/mode).
  - Outliers and skewness addressed via log transformations and scaling.

### Exploratory Data Analysis (EDA)
- **Summary Statistics:** Generated descriptive statistics (mean, median, std. deviation) for key features.
- **Visualizations:** Histograms, box plots, and time-series plots were created to reveal trends and distribution patterns.
- **Insights:**  
  - Seasonal trends and high-volatility periods observed.
  - High correlation among certain technical indicators, guiding feature expansion efforts.

### Preprocessing Steps
- **Missing Data:** Imputed missing values using mean/mode as appropriate.
- **Scaling & Transformation:**  
  - Normalization performed using StandardScaler.
  - Logarithmic transformations and polynomial feature expansions applied to select features.
- **Encoding:** One-hot encoding for any categorical variables (if applicable).
- **Probability Conversion & CPT Generation:**  
  - Continuous features discretized into bins.
  - Frequency counts computed and normalized to create Conditional Probability Tables (CPTs) for our Bayesian network.

## Model 1: Training and Evaluation
### Model Description
- **Baseline Model:**  
  - A Bayesian network model estimating market conditions, which informs our trading decisions.
  - Serves as the first component in our probabilistic modeling framework, paving the way for HMMs and RL integration.
  
### Training Process
- **Data Splitting:** Training (80%) and Validation (20%).
- **Hyperparameter Tuning:** Adjusted discretization thresholds and smoothing parameters.
- **Implementation:** Training executed via Python scripts and Jupyter notebooks (see links below).

### Evaluation
- **Metrics & Analysis:**  
  - Learning curves, training vs. validation error plots, and additional performance metrics are generated.
  - The model currently exhibits [describe if it’s underfitting/overfitting, e.g., “a slight underfitting trend, suggesting further feature engineering may be beneficial”].
- **Code & Notebooks:**  
  - Detailed EDA, preprocessing, and model training can be reviewed in the following notebooks:
    - [EDA & Preprocessing Notebook](#)
    - [Model Training & Evaluation Notebook](#)

## Conclusion & Future Improvements
- **Conclusion:**  
  - The baseline Bayesian network provides a solid initial estimation of market conditions, though there is potential for improvement.
- **Future Enhancements:**  
  - Integrate HMMs to better capture latent market regimes.
  - Incorporate Reinforcement Learning for dynamic policy optimization.
  - Further refine feature engineering and experiment with alternative discretization methods.

## Repository Structure & Collaboration
- **Repository Structure:**
  - `data/` – Contains raw and processed stock market data.
  - `notebooks/` – Jupyter notebooks for EDA, preprocessing, and model training/evaluation.
  - `scripts/` – Python scripts for data processing, CPT generation, and model implementation.
  - `docs/` – Additional documentation and design notes.
  - `README.md` – This submission document.
- **Collaboration Certification:**  
  - I certify that I have added **atong28** to our group project GitHub repository with at least read access.

## Milestone2 Branch URL
The Milestone2 branch for this submission can be found at:  
[https://github.com/your-username/your-repo-name/tree/Milestone2](https://github.com/your-username/your-repo-name/tree/Milestone2)
