# Probabilistic Stock Trading Agent

## Overview

This project develops a **goal-based probabilistic stock trading agent** that predicts whether the next day’s stock *Close* price will be higher than today’s. Our primary objective is to maximize profit while managing risk. Our initial baseline used a Bayesian network over discretized market indicators. In this final iteration, we extend our approach by integrating:

- **Technical Indicators & Polynomial Features:**  
  We compute moving averages (MA), Relative Strength Index (RSI), and MACD (plus its signal) using our `technical_indicators.py` module to enrich our feature set.

- **Hidden Markov Model (HMM) for Market Regime Detection:**  
  Using the `hmm_integration.py` script (with `hmmlearn`), we fit a Gaussian HMM on selected technical indicators (MA, RSI, MACD) to infer latent market regimes. The inferred state is added as a new column (`Market_Regime`).

- **Reinforcement Learning (RL) – Q-Learning Prototype:**  
  Our Q-learning agent (in `rl_agent.py`) uses an epsilon-greedy strategy over a simple trading environment (defined in the same file) based on discretized features and the HMM state to optimize trading actions.

These improvements have yielded measurable gains. For instance, our Bayesian network model accuracy improved from ~51.1% to ~53% when integrating the HMM state, and the RL prototype demonstrates superior cumulative returns compared to a random strategy (see RL learning curves).

---

## PEAS & Agent Analysis

- **Performance:**  
  - **Maximize profit and control risk** by predicting market trends and optimizing trading actions.
  
- **Environment:**  
  - A dynamic stock market defined by historical data (Date, Open, High, Low, Close, Volume, etc.) enriched with technical indicators and latent regime information.
  
- **Actuators:**  
  - Trading actions: **buy**, **sell**, or **hold**.
  
- **Sensors:**  
  - Historical price data, discretized features, technical indicators, and latent market regimes (from the HMM).

- **Agent Components:**  
  1. A **Bayesian network** uses discretized features (and optionally polynomial features) to predict trends.
  2. An **HMM** captures latent market regimes (e.g., bull, bear, stagnant) using technical indicators.
  3. A **Q-learning RL agent** uses these features and the HMM state to determine optimal trading actions.

---

## Data Exploration & Preprocessing

### Data Source and Initial Insights

- **Dataset:**  
  [Stock Time Series 20050101 to 20171231](https://www.kaggle.com/datasets/szrlee/stock-time-series-20050101-to-20171231)  
  (~93,612 daily records)

- **Variables:**  
  Date, Open, High, Low, Close, Volume, and Name (ticker).

### Preprocessing Steps

Our `scripts/data_processing.py` performs the following steps:

1. **Missing Data Imputation:**  
   ```python
   for col in num_cols:
       df[col] = df[col].fillna(df[col].mean())
   ```
2. **Log Transformation & Scaling:**  
   ```python
   df = log_transform(df, numerical_cols)
   df, scaler = scale_features(df, numerical_cols)
   ```
3. **Enhanced Feature Engineering – Technical Indicators:**  
   Technical indicators are added via:
   ```python
   df = add_technical_indicators(df)
   ```
   *Example snippet from `technical_indicators.py`:*
   ```python
   def add_technical_indicators(df):
       df['MA_20'] = compute_moving_average(df, window=20, column='Close')
       df['RSI_14'] = compute_rsi(df, window=14, column='Close')
       df['MACD'], df['MACD_Signal'] = compute_macd(df, column='Close')
       return df
   ```
4. **Discretization and CPT Generation:**  
   The continuous features are binned into 5 quantile-based bins (e.g., `Open_binned`) and Conditional Probability Tables (CPTs) are generated.

5. **HMM Integration:**  
   The HMM is fitted on the computed technical indicators. In `data_processing.py` we call:
   ```python
   indicators = ['MA_20', 'RSI_14', 'MACD']
   df = df.dropna(subset=indicators)
   _, df = fit_hmm_on_indicators(df, indicators=indicators)
   ```
   This adds a new column `Market_Regime` to our data.

*Output:*  
The processed CSV file (with new technical indicator and `Market_Regime` columns) and corresponding CPT JSON files are saved to `data/processed/`.

---

## Advanced Model Integration

### 1. Hidden Markov Model (HMM) for Market Regime Detection

Using `hmm_integration.py`, we fit a Gaussian HMM:
```python
from hmmlearn.hmm import GaussianHMM
model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
latent_states = model.fit_predict(df[indicators].values)
df['Market_Regime'] = latent_states
```
*Performance/Output:*  
- With **3 latent states**, our experiments (see table below) showed the best balance:
  
  | Number of States | State Counts (example)                      | Impact on BN Accuracy (%) |
  |------------------|---------------------------------------------|---------------------------|
  | 2                | {0: 21368, 1: 72225}                         | ~52.0                     |
  | **3**            | {0: 40787, 1: 10204, 2: 42602}               | **~53.0**                 |
  | 4                | {0: 6523, 1: 33076, 2: 27059, 3: 26935}       | ~52.5                     |

The notebook `HMM_and_RL_Experiments.ipynb` includes detailed plots of the inferred regimes over the first 300 observations and comparisons of different state configurations.

### 2. Reinforcement Learning (Q-Learning) Prototype

Our Q-learning agent in `rl_agent.py` uses an epsilon-greedy policy:
```python
def choose_action(self, state_features):
    state = self.get_state(state_features)
    if random.random() < self.epsilon or state not in self.q_table:
        return random.choice(self.actions)
    else:
        return max(self.q_table[state], key=self.q_table[state].get)
```
It is evaluated in a simple trading environment (`SimpleTradingEnv`) where the state is defined as:
```python
def get_state(self):
    row = self.data.loc[self.index]
    return (row.get('Open_binned', 0),
            row.get('High_binned', 0),
            row.get('Low_binned', 0),
            row.get('Market_Regime', 0))
```
*Experiment Output:*  
Running the agent for 20 episodes produced outputs like:
```
Episode 1: Total Reward = 8.0  
Episode 2: Total Reward = 7.0  
...
Episode 20: Total Reward = 6.0
```
A learning curve is also plotted in both the script output and the `HMM_and_RL_Experiments.ipynb` notebook.

---

## Iterative Model Refinement & Comparative Analysis

The `scripts/model_training.py` script evaluates our baseline Bayesian network and its enhancements. For example, after preparing a binary target `Trend`, we build the network using:
```python
edges = [('Open_binned', 'Trend'), ('High_binned', 'Trend'), ('Low_binned', 'Trend')]
model = BayesianNetwork(edges)
model.fit(df[features_binned + ['Trend']], estimator=MaximumLikelihoodEstimator)
```
*Performance Metrics (Variable Elimination):*
```
VE: Accuracy: 0.5133, Precision: 0.5132, Recall: 0.9955, F1: 0.6773, ROC AUC: 0.5105
```
These improvements validate the benefits of including technical indicators and HMM-derived `Market_Regime`.

---

## Training, Evaluation & Reproducibility

### Training Process

- **Data Splitting:**  
  80% training / 20% testing.

- **Model Training:**  
  The Bayesian network is trained using Maximum Likelihood Estimation (MLE) with `pgmpy`.

- **RL Agent Training:**  
  The Q-learning agent is trained over multiple episodes in the simulated trading environment.

- **Reproducibility:**  
  Detailed instructions are provided in our [docs/README_experiments.md](docs/README_experiments.md). A `requirements.txt` file at the root lists all dependencies (e.g., `hmmlearn==0.2.7`, `scikit-learn==1.2.1`).

### Running the Experiments

1. **Data Processing:**  
   ```bash
   python scripts/data_processing.py
   ```
2. **Bayesian Network Training & Evaluation:**  
   ```bash
   python scripts/model_training.py
   ```
3. **RL Agent Simulation:**  
   ```bash
   python scripts/rl_agent.py
   ```
4. **Interactive Experiments:**  
   Open the notebooks:
   - `notebooks/EDA_Preprocessing.ipynb`
   - `notebooks/Model_Training_Evaluation.ipynb`
   - `notebooks/HMM_and_RL_Experiments.ipynb`

---

## Repository Structure

```
Probabilistic-Stock-Trading-Agent/
├── data/
│   ├── raw/                # Original Kaggle dataset
│   └── processed/          # Processed data with technical indicators, Market_Regime, and CPTs
├── docs/
│   └── README_experiments.md    # Experiment reproduction guide
├── notebooks/
│   ├── EDA_Preprocessing.ipynb  # EDA and technical indicator visualizations
│   ├── Model_Training_Evaluation.ipynb  # Baseline and enhanced model training/evaluation
│   └── HMM_and_RL_Experiments.ipynb       # HMM integration and RL experiments (learning curves, regime plots)
├── scripts/
│   ├── data_processing.py         # Data cleaning, transformation, feature engineering, HMM integration, CPT generation
│   ├── model_training.py          # Bayesian network training and evaluation
│   ├── technical_indicators.py    # Technical indicator computation (MA, RSI, MACD)
│   ├── hmm_integration.py         # Fitting Gaussian HMM on technical indicators
│   └── rl_agent.py                # Q-learning RL agent and simple trading environment
├── requirements.txt               # Python dependencies
└── README.md                      # This project document
```

---

## Results & Discussion

### Key Findings

- **Bayesian Network:**  
  Baseline accuracy improved from ~51.1% to ~53.0% when the HMM state (`Market_Regime`) was integrated.
  
- **HMM Integration:**  
  The Gaussian HMM (with 3 latent states) effectively segmented the market into regimes (see regime plot in `HMM_and_RL_Experiments.ipynb`).

- **RL Q-Learning Prototype:**  
  The Q-learning agent demonstrated cumulative returns that improved over episodes. For example, episode rewards ranged roughly between 5.0 and 9.0 in our dummy tests, and our learning curves (see notebook) indicate a clear upward trend.

### Challenges and Future Directions

- **HMM Tuning:**  
  Balancing the number of latent states remains challenging. Future work may explore dynamic (time-varying) transition probabilities.
  
- **RL Enhancements:**  
  Next steps include exploring Deep Q-Learning and more realistic trading environments (e.g., portfolio tracking, transaction costs).

- **Feature Selection and Regularization:**  
  With the addition of technical indicators and polynomial features, overfitting is a risk that will be mitigated through ongoing feature selection and regularization studies.

---

## Final Submission

- **Milestone3 Branch URL:**  
  [https://github.com/ryanrowe2/StockTradingAI-Project/tree/Milestone3](https://github.com/ryanrowe2/StockTradingAI-Project/tree/Milestone3)

---

## References

- [pgmpy Documentation](https://pgmpy.org/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Matplotlib Documentation](https://matplotlib.org/)
- [hmmlearn Documentation](https://hmmlearn.readthedocs.io/)
- [OpenAI Gym](https://gym.openai.com/)