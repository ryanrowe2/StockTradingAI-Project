# scripts/data_processing.py

import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load stock market data from a CSV file."""
    df = pd.read_csv(filepath)
    return df

def impute_missing(df):
    """Impute missing values using mean for numerical columns."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col].fillna(df[col].mean(), inplace=True)
    return df

def scale_features(df, feature_cols):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler

def log_transform(df, feature_cols):
    """Apply a logarithmic transformation to reduce skewness."""
    for col in feature_cols:
        # Add a small constant to avoid log(0)
        df[col] = np.log(df[col] + 1e-5)
    return df

def discretize_features(df, feature_cols, bins=5):
    """Discretize continuous features into bins using quantiles."""
    bin_edges = {}
    for col in feature_cols:
        df[col + '_binned'], bin_edges[col] = pd.qcut(df[col], q=bins, retbins=True, labels=False, duplicates='drop')
    return df, bin_edges

def generate_cpts(df, feature_cols):
    """Generate Conditional Probability Tables (CPTs) for binned features."""
    cpts = {}
    for col in feature_cols:
        binned_col = col + '_binned'
        counts = df[binned_col].value_counts().sort_index()
        total = counts.sum()
        probs = (counts / total).to_dict()
        cpts[col] = probs
    return cpts

def save_processed_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

def save_cpts(cpts, output_path):
    with open(output_path, 'w') as f:
        json.dump(cpts, f, indent=4)
    print(f"CPTs saved to {output_path}")

def main():
    # Define file paths
    raw_data_path = os.path.join('data', 'raw', 'stock_data.csv')
    processed_data_path = os.path.join('data', 'processed', 'processed_stock_data.csv')
    cpts_path = os.path.join('data', 'processed', 'CPTs.json')
    
    # Load raw data
    df = load_data(raw_data_path)
    print("Data loaded. Shape:", df.shape)
    
    # Impute missing values
    df = impute_missing(df)
    
    # Identify numerical columns (excluding non-numeric or date columns if necessary)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Apply log transformation to reduce skewness
    df = log_transform(df, numerical_cols)
    
    # Scale numerical features
    df, scaler = scale_features(df, numerical_cols)
    
    # Discretize numerical features for CPT generation
    df, bin_edges = discretize_features(df, numerical_cols, bins=5)
    
    # Generate CPTs based on the discretized features
    cpts = generate_cpts(df, numerical_cols)
    
    # Ensure output directory exists
    os.makedirs(os.path.join('data', 'processed'), exist_ok=True)
    
    # Save processed data and CPTs
    save_processed_data(df, processed_data_path)
    save_cpts(cpts, cpts_path)

if __name__ == '__main__':
    main()
