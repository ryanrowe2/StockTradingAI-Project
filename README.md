# Probabilistic Stock Trading Agent

## Update: Regrade Responses
- **EDA Outputs:** We have now included actual sample summary statistics and plots from the original EDA notebook output in this README.md file (see [EDA_Preprocessing.ipynb](https://github.com/ryanrowe2/StockTradingAI-Project/blob/Milestone2/notebooks/EDA_Preprocessing.ipynb)). In this notebook, you will find histograms, box plots, and time-series plots that show the distribution of key features (e.g., the "Close" price) and reveal details such as skewness and outliers.
- **Preprocessing Justification:** Our raw data was highly skewed, particularly for price and volume. To address this, we applied logarithmic transformation and quantile-based binning (e.g., into 5 bins) to produce features like `Open_binned`, `High_binned`, and `Low_binned`. This ensures that each bin represents approximately 20% of the data, which is why the CPT values for these features are near 0.2 in a uniform scenario.
- **Bayesian Network Diagram:** We have included a diagram below that illustrates our baseline Bayesian network structure.
- **Library Descriptions:** Our code uses [pgmpy](https://pgmpy.org/) for building and performing inference on Bayesian networks, [scikit-learn](https://scikit-learn.org/) for train-test splitting and evaluation metrics, and [matplotlib](https://matplotlib.org/) for plotting. Detailed explanations of the key functions are provided in the inline documentation.
- **Codebase Organization:** The repository is organized into clear folders:  
  - **data/** contains raw and processed datasets.  
  - **notebooks/** includes Jupyter notebooks with detailed EDA, preprocessing steps, and model training/evaluation outputs.  
  - **scripts/** holds our Python code for data processing and model training (e.g., `model_training.py`).  
  - **docs/** contains additional design notes and documentation (e.g., this design rationale).  
- **Conclusion Enhancements:** In the conclusion, we now discuss that the modest 51% accuracy likely stems from the noisy, complex nature of stock market data, the potential loss of nuanced information through aggressive discretization, and the simplicity of our baseline model. Future work will focus on enhancing feature engineering, exploring alternative probabilistic models (such as HMMs and RL), and refining our network structure.

---

## Project Overview & Agent Description (PEAS)
This project aims to build a **goal-based probabilistic stock trading agent** using historical stock market data. Our agent leverages a Bayesian network to estimate market conditions and guide trading decisions. In PEAS terms:

- **Performance:** Maximize profit and control risk by predicting whether the next day's stock Close price will be higher than today's.
- **Environment:** The dynamic, uncertain world of the stock market, represented by historical data (Date, Open, High, Low, Close, Volume, etc.) from the "Stock Time Series 20050101 to 20171231" Kaggle dataset.
- **Actuators:** Trading actions such as buy, sell, or hold.
- **Sensors:** Historical market indicators and technical signals derived from the dataset.

Our agent is goal-based; it uses probabilistic predictions to inform trading actions that aim to maximize profit while controlling risk.

---

## Data Exploration & Preprocessing

### Data Source
- **Dataset:** [Stock Time Series 20050101 to 20171231](https://www.kaggle.com/datasets/szrlee/stock-time-series-20050101-to-20171231)
- **Description:** Over a decade of daily stock market data for various stocks (fields include Date, Open, High, Low, Close, Volume, Name).
- **Observations:** Approximately 93,612 observations.

### Exploratory Data Analysis (EDA)
In our [EDA_Preprocessing.ipynb](https://github.com/ryanrowe2/StockTradingAI-Project/blob/Milestone2/notebooks/EDA_Preprocessing.ipynb) notebook, we computed:
- **Summary Statistics:**
  - The "Close" price showed a mean of 85.64, median of 60.05, and standard deviation of 108.12.
- **Visualizations:**  
  - Histograms for the "Close" price revealed a right-skewed distribution:
    
    ![image](https://github.com/user-attachments/assets/5de81488-608b-45f9-a4a4-fecf751d48c3)

  - Box plots helped identify outliers:
 
    ![image](https://github.com/user-attachments/assets/7d61d571-2b48-46ce-a7c3-765244f31d12)
    
- **Missing Data:**  
  - We identified and imputed missing numerical values using column means.

### Preprocessing Steps
- **Normalization & Transformation:**  
  - **Normalization:** We applied StandardScaler to standardize features (mean=0, std=1).  
  - **Log Transformation:** A log transformation was used to reduce skewness in price and volume data.
- **Discretization:**  
  Continuous features such as **Open**, **High**, and **Low** were discretized into quantile-based bins (5 bins by default).  
  - **Mathematically:** If `X` is a continuous variable (e.g., the Open price), then its binned version `X_binned` is defined by partitioning the range of `X` into 5 intervals (quantiles). If the data is uniformly distributed, each bin will contain roughly 20% of the data, leading to conditional probabilities of about 0.2 for each bin.
  - **Result:** New features `Open_binned`, `High_binned`, and `Low_binned` represent categorical approximations of the original values.
- **CPT Generation:**  
  Frequency counts for each bin were normalized to create Conditional Probability Tables (CPTs). These CPTs capture the probability distribution of the discretized features and are used for probabilistic inference in our Bayesian network.

---

## Baseline Model

### Bayesian Network Structure
Our baseline model is a Bayesian network where each discretized feature is assumed to influence the target variable, `Trend`. The structure is as follows:

      Open_binned    High_binned    Low_binned
           \            |            /
            \           |           /
             \          |          /
                     Trend


- **Nodes:**  
  - **Parent Nodes:** `Open_binned`, `High_binned`, `Low_binned`
  - **Child Node:** `Trend` (1 if next day's Close > today's Close, 0 otherwise)
- **Learning:**  
  The network learns the conditional probability P(Trend | Open_binned, High_binned, Low_binned) via Maximum Likelihood Estimation (MLE). The learned CPDs (logged in our code) provide insight into how each discretized feature affects the trend.

### External Libraries
- **pgmpy:** Used to model Bayesian networks, perform parameter estimation (MLE), and run inference (Variable Elimination).  
- **scikit-learn:** Employed for splitting data into training and testing sets and for calculating evaluation metrics (precision, recall, F1, confusion matrix, ROC curve, AUC).  
- **matplotlib:** Utilized to plot diagnostic visualizations, such as confusion matrices and ROC curves.  
- **logging:** Provides detailed runtime diagnostics.
- **argparse:** Allows for command-line configuration of hyperparameters and file paths.

---

## Training & Evaluation

### Training Process
- The dataset is split into training (80%) and testing (20%) sets.
- The Bayesian network is trained using Maximum Likelihood Estimation on the training set.
- The network's structure is defined such that each discretized feature (e.g., `Open_binned`, `High_binned`, `Low_binned`) influences the `Trend` prediction.

### Inference & Evaluation
- **Inference:**  
  We use Variable Elimination (VE) to compute the probability P(Trend = 1 | features) for each test instance.
- **Evaluation Metrics:**  
  The model is evaluated using:
  - Accuracy, Precision, Recall, and F1 Score.
  - A Confusion Matrix and ROC Curve (with ROC AUC) are generated for diagnostic visualization.
- **Current Performance:**  
  Our baseline model achieves an accuracy of approximately 51.1%. This modest performance is expected given the noisy nature of financial data and the simplicity of our model.

---

## Repository Structure & Navigation

- **data/**  
  Contains both raw data from Kaggle and processed data (post-preprocessing, including CPT files).
- **notebooks/**  
  Contains Jupyter notebooks such as **EDA_Preprocessing.ipynb**, where detailed exploratory analysis (with actual plots and statistics) is performed.
- **scripts/**  
  Contains Python scripts, including:
  - **data_processing.py:** Preprocesses the raw data, including scaling, log transformation, discretization, and CPT generation.
  - **model_training.py:** Builds and evaluates the Bayesian network model. This script includes detailed logging and supports multiple inference methods.
- **docs/**  
  Contains additional documentation and design notes that further explain our design choices, preprocessing decisions, and model structure.
- **README.md:**  
  This document summarizes the entire project, linking to all relevant code, notebooks, and documentation.

---

## Conclusion & Future Work

### Conclusion
The baseline Bayesian network model demonstrates a fundamental approach to predicting whether the next day's Close price will be higher (`Trend` = 1). With an accuracy of around 51.1%, the model is only marginally better than random guessing. Several factors may contribute to this result:
- **Data Noise:** Stock market data is inherently noisy, and simple historical indicators may not capture the full complexity of market behavior.
- **Feature Discretization:** While discretizing continuous features simplifies probabilistic modeling, it may also lead to loss of nuanced information.
- **Model Simplicity:** Our network assumes a direct, independent influence of `Open_binned`, `High_binned`, and `Low_binned` on `Trend`. More complex interactions or additional features might be required to improve predictive power.

### Future Enhancements
- **Advanced Probabilistic Models:**  
  Integrate Hidden Markov Models (HMMs) to capture temporal dependencies and latent market regimes.
- **Reinforcement Learning (RL):**  
  Develop RL-based approaches that can optimize trading decisions directly based on the model’s predictions.
- **Enhanced Feature Engineering:**  
  Explore additional technical indicators, alternative discretization techniques, or polynomial and log-multiplicative feature expansions to better capture market patterns.
- **Model Complexity:**  
  Consider more complex Bayesian network structures (with additional dependencies among features) to improve accuracy.

---

## Links to Code and Notebooks
- [Data Exploration & Preprocessing Notebook](https://github.com/ryanrowe2/StockTradingAI-Project/blob/Milestone2/notebooks/EDA_Preprocessing.ipynb)
- [Model Training Script (model_training.py)](https://github.com/ryanrowe2/StockTradingAI-Project/blob/Milestone2/scripts/model_training.py)

---

This README.md now includes:
- Detailed EDA outputs and preprocessing justification.
- A clear description (with diagram) of our Bayesian network structure.
- Explanations of external libraries and the codebase organization.
- A discussion of the baseline model’s performance and limitations.
- Future work plans for improving predictive accuracy and agent performance.

---

## Milestone2 Branch URL
The Milestone2 branch for this submission can be found at:  
[https://github.com/ryanrowe2/StockTradingAI-Project/tree/Milestone2](https://github.com/ryanrowe2/StockTradingAI-Project/tree/Milestone2)
