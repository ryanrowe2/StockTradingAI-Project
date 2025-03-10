# Experiment Reproduction Guide

This document provides detailed instructions for reproducing the experiments in the **Probabilistic Stock Trading Agent** project, including advanced techniques introduced in Milestone 3. You will find instructions on running the following experiments:

- **Enhanced Feature Engineering** (Technical Indicators & Polynomial Features)
- **HMM Integration for Market Regime Detection**
- **Reinforcement Learning (RL) – Q-Learning Prototype**

Follow the steps below to set up your environment, run the experiments, and review the results.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Running the Data Processing Pipeline](#running-the-data-processing-pipeline)
- [HMM Integration Experiment](#hmm-integration-experiment)
- [RL Q-Learning Experiment](#rl-q-learning-experiment)
- [Evaluation and Results](#evaluation-and-results)
- [Troubleshooting](#troubleshooting)
- [Future Work](#future-work)

---

## Prerequisites

Before running the experiments, ensure you have the following installed:

- Python 3.8 or later
- Git

The required Python packages are listed in the `requirements.txt` file at the root of the repository. Key dependencies include:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `pgmpy`
- `hmmlearn`

---

## Repository Structure

Your repository should have the following structure:

```
Probabilistic-Stock-Trading-Agent/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
|   ├── design_nodes.md
│   └── README_experiments.md    <-- (This file)
├── notebooks/
│   ├── EDA_Preprocessing.ipynb
│   ├── Model_Training_Evaluation.ipynb
│   └── HMM_and_RL_Experiments.ipynb
├── scripts/
│   ├── data_processing.py
│   ├── model_training.py
│   ├── technical_indicators.py
│   ├── hmm_integration.py
│   └── rl_agent.py
├── requirements.txt
└── README.md
```

---

## Environment Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/ryanrowe2/StockTradingAI-Project.git
   cd StockTradingAI-Project
   ```

2. **Install Dependencies:**

   Ensure you are in the root directory (where `requirements.txt` is located) and run:

   ```bash
   pip install -r requirements.txt
   ```

3. **Verify the Directory Structure:**

   Confirm that the `docs/`, `notebooks/`, and `scripts/` folders are present and contain the respective files.

---

## Running the Data Processing Pipeline

The data processing pipeline is responsible for:
- Loading raw data from `data/raw/`
- Imputing missing values, applying log transformation, and scaling
- Computing technical indicators (MA, RSI, MACD)
- (Optionally) Generating polynomial features
- Discretizing numerical features and generating CPTs
- Integrating the HMM to infer market regimes

To run the pipeline:

```bash
python scripts/data_processing.py
```

This script will:
- Save the processed data (with new indicator and `Market_Regime` columns) to `data/processed/`
- Save the corresponding CPT JSON files in the same folder

---

## HMM Integration Experiment

The HMM integration experiment is designed to assess how latent market regimes (inferred via technical indicators) enhance our feature set.

### Steps:
1. **Ensure Technical Indicators Are Added:**
   - The `data_processing.py` script automatically calls `add_technical_indicators` from `technical_indicators.py`.

2. **Fit the HMM:**
   - The script then calls `fit_hmm_on_indicators` from `hmm_integration.py` to compute and add the `Market_Regime` column.
   - You can adjust the number of latent states by editing the parameter in `fit_hmm_on_indicators`.

3. **Review Output:**
   - Open the processed CSV file in `data/processed/` and verify that the `Market_Regime` column is present.
   - For further analysis, open the `HMM_and_RL_Experiments.ipynb` notebook and inspect the plots showing the regime assignments and a summary table of different state experiments.

---

## RL Q-Learning Experiment

The Q-learning experiment uses a simple trading environment to test our reinforcement learning agent.

### Steps:
1. **Setup the RL Environment:**
   - The `SimpleTradingEnv` is defined in `rl_agent.py`. This environment uses key features such as `Open_binned`, `High_binned`, `Low_binned`, and `Market_Regime`.

2. **Run the RL Agent:**
   - You can run the RL experiment by executing the module directly:

     ```bash
     python scripts/rl_agent.py
     ```

   - The script simulates a specified number of episodes and prints the total reward for each episode.

3. **Visualize Learning:**
   - The script will plot a learning curve (episodic rewards) that shows the agent's progress over the episodes.
   - Open the `HMM_and_RL_Experiments.ipynb` notebook to view additional analysis and comparisons of RL performance.

---

## Evaluation and Results

- **Bayesian Network Evaluation:**
  - The `model_training.py` script evaluates the baseline and enhanced Bayesian network models.
  - Use the command-line arguments (e.g., `--use_hmm`) to include or exclude the HMM state.
  - Evaluation metrics (accuracy, precision, recall, F1, ROC AUC) are logged and plotted.

- **RL Agent Evaluation:**
  - Review the learning curves (episodic rewards) generated by the RL agent.
  - Compare simulated cumulative returns from the RL agent with a baseline random strategy.

For a side-by-side comparison, refer to the summary tables and plots in the `Model_Training_Evaluation.ipynb` and `HMM_and_RL_Experiments.ipynb` notebooks.

---

## Troubleshooting

- **Missing Dependencies:**  
  Ensure that you have installed all dependencies listed in `requirements.txt`. If any module is missing (e.g., `hmmlearn`), run:
  
  ```bash
  pip install hmmlearn
  ```

- **Data File Paths:**  
  Make sure that the raw data files are located in `data/raw/` and that file paths in the scripts are correctly set.

- **Technical Indicator NaNs:**  
  If you encounter NaN values in technical indicator columns, verify that your raw data contains enough rows (especially for moving averages and RSI calculations).

- **RL Agent Not Learning:**  
  If the Q-learning agent’s performance is poor, consider adjusting hyperparameters (alpha, gamma, epsilon) in `rl_agent.py` or testing the environment with a simplified subset of data.

---

## Future Directions

Our next steps include:
- **Deep Q-Learning Implementation:** Explore neural network-based approaches for more sophisticated state representations.
- **Time-Varying HMM Transitions:** Investigate more complex HMM models with dynamic transition probabilities.
- **Systematic Hyperparameter Optimization:** Use grid search or Bayesian optimization to fine-tune model parameters.
- **Model Ensembling:** Combine predictions from multiple models (e.g., Bayesian network, HMM, RL) to improve overall performance.

By following the steps outlined in this document, you should be able to fully reproduce our advanced experiments for the Probabilistic Stock Trading Agent project. If you have any questions or need further assistance, please refer to the project’s main README.md or contact the development team.

Happy experimenting!

---
