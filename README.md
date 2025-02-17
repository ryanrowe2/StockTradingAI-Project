# Probabilistic Stock Trading Agent

## Project Overview & Agent Description (PEAS)
This project develops a goal-based probabilistic stock trading agent designed to maximize profit while controlling risk. The agent operates in the stock market environment using historical market data, where its **sensors** are market indicators (such as prices, volumes, and technical signals), and its **actuators** are trading decisions (buy, sell, hold). The performance is measured by profit maximization and risk-adjusted returns. Our agent initially employs a Bayesian network-based model to estimate market conditions, forming the basis for further probabilistic models (e.g., Hidden Markov Models and Reinforcement Learning) to be integrated in later milestones.

## Data Exploration & Preprocessing
We use the **Stock Time Series 20050101 to 20171231** dataset from Kaggle, which contains over a decade of historical stock market data. The dataset includes approximately *[X]* observations and *[Y]* features (e.g., open, high, low, close prices, trading volume, and technical indicators). Our exploratory data analysis (EDA) involved:
- Generating summary statistics and visualizing feature distributions
- Identifying and imputing missing data using mean/mode imputation
- Analyzing the scale of each feature and applying normalization via StandardScaler
- Expanding features through polynomial transformations and logarithmic scaling

## Converting Observations into Probabilities & CPT Generation
To prepare our Bayesian network, we discretized continuous features (e.g., price changes, volume) into bins and computed conditional probabilities based on frequency counts. These counts were then normalized to generate Conditional Probability Tables (CPTs) that capture the relationships among key market indicators. Detailed code for these processes is provided in our preprocessing notebook: [Preprocessing Notebook Link](#).

## Model 1: Training and Evaluation
Our first model is a baseline Bayesian network that estimates market states and informs trading decisions. Key steps include:
- **Training:** Using historical data from 2005–2017, we trained the Bayesian network by tuning hyperparameters (e.g., discretization thresholds, smoothing factors).
- **Evaluation:** We split the dataset into training and validation sets and plotted learning curves to assess performance. The model’s error metrics (e.g., training vs. validation error) indicate that it currently [describe: e.g., “exhibits moderate underfitting/overfitting”]. Evaluation plots and detailed metrics are available in our evaluation notebook: [Evaluation Notebook Link](#).

## Conclusion & Future Improvements
The initial baseline model provides a foundational probabilistic estimate of market conditions, yet there is room for improvement. Future work will focus on:
- Enhancing feature engineering and exploring alternative discretization methods
- Integrating additional probabilistic models (such as HMMs to capture market regime changes and RL to optimize trading policies)
- Fine-tuning hyperparameters to better balance the bias-variance trade-off and improve model generalization

## Repository Structure & Collaboration
The repository is organized as follows:
- **`data/`**: Contains raw and processed stock market data.
- **`notebooks/`**: Jupyter notebooks for EDA, preprocessing, and model training/evaluation.
- **`scripts/`**: Python scripts for data processing, model training, and CPT generation.
- **`docs/`**: Supplementary documentation and design notes.
- **`README.md`**: This submission document.

I certify that I have added **atong28** to our group project GitHub repository with at least read access.

## GitHub URL
The Milestone2 branch for this submission can be found at:  
[https://github.com/your-username/your-repo-name/tree/Milestone2](https://github.com/your-username/your-repo-name/tree/Milestone2)
