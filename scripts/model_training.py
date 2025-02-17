# scripts/model_training.py

import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt

def load_processed_data(filepath):
    """Load the processed data from CSV."""
    return pd.read_csv(filepath)

def load_cpts(filepath):
    """Load generated CPTs from JSON."""
    with open(filepath, 'r') as f:
        cpts = json.load(f)
    return cpts

def prepare_target(df):
    """
    Create a binary target 'Trend' based on the change in the 'Close' price.
    Trend = 1 if the next day's Close is higher than today's, else 0.
    """
    df['Trend'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    return df

def build_bayesian_network(df, features, target):
    """
    Build a simple Bayesian network where each feature influences the target.
    The model structure is defined as: each feature -> target.
    """
    # Define the network structure
    edges = [(feature, target) for feature in features]
    model = BayesianModel(edges)
    
    # Fit the model using Maximum Likelihood Estimation
    model.fit(df[features + [target]], estimator=MaximumLikelihoodEstimator)
    return model

def evaluate_model(model, df, features, target):
    """Evaluate the Bayesian network model using accuracy on a test set."""
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    
    # Fit the model on the training set
    model.fit(train[features + [target]], estimator=MaximumLikelihoodEstimator)
    
    # Set up inference
    inference = VariableElimination(model)
    
    predictions = []
    actuals = test[target].tolist()
    
    for index, row in test[features].iterrows():
        evidence = row.to_dict()
        q = inference.query(variables=[target], evidence=evidence, show_progress=False)
        prob = q.values[1]  # probability of target being 1
        prediction = 1 if prob >= 0.5 else 0
        predictions.append(prediction)
    
    accuracy = np.mean(np.array(predictions) == np.array(actuals))
    print("Model accuracy on test set:", accuracy)
    return predictions, actuals, accuracy

def main():
    processed_data_path = os.path.join('data', 'processed', 'processed_stock_data.csv')
    cpts_path = os.path.join('data', 'processed', 'CPTs.json')
    
    # Load the processed data and CPTs
    df = load_processed_data(processed_data_path)
    cpts = load_cpts(cpts_path)
    print("Processed data shape:", df.shape)
    
    # Prepare target variable 'Trend'
    df = prepare_target(df)
    
    # Define features for the model:
    # For demonstration, we assume that for each numerical column, a binned version exists.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Trend' in numeric_cols:
        numeric_cols.remove('Trend')
    # Choose a subset of features (e.g., first 3 numerical columns)
    features = numeric_cols[:3]
    # Use binned versions if they exist (e.g., 'Close_binned' instead of 'Close')
    features_binned = [f"{col}_binned" if f"{col}_binned" in df.columns else col for col in features]
    target = 'Trend'
    
    print("Features used for modeling:", features_binned)
    
    # Build and train the Bayesian network model
    model = build_bayesian_network(df, features_binned, target)
    
    # Evaluate the model
    predictions, actuals, accuracy = evaluate_model(model, df, features_binned, target)
    print("Baseline Bayesian Network Model Accuracy:", accuracy)
    
    # Optionally, you can plot learning curves if you run iterative training; for now, we print accuracy.

if __name__ == '__main__':
    main()
