# scripts/data_processing.py

import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler
import kagglehub

def load_data(filepath):
    """
    Load stock market data from a CSV file.
    
    Parameters:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    df = pd.read_csv(filepath)
    return df

def impute_missing(df):
    """
    Impute missing values in numerical columns with the column mean.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        
    Returns:
        pd.DataFrame: DataFrame with imputed missing values.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col].fillna(df[col].mean(), inplace=True)
    return df

def log_transform(df, feature_cols):
    """
    Apply a logarithmic transformation to reduce skewness in numerical features.
    A small constant is added to avoid log(0).
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        feature_cols (list): List of feature column names to transform.
        
    Returns:
        pd.DataFrame: DataFrame with log-transformed features.
    """
    for col in feature_cols:
        df[col] = np.log(df[col] + 1e-5)
    return df

def scale_features(df, feature_cols):
    """
    Scale numerical features using StandardScaler.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        feature_cols (list): List of numerical feature column names to scale.
        
    Returns:
        tuple: (Scaled DataFrame, scaler object)
    """
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler

def discretize_features(df, feature_cols, bins=5):
    """
    Discretize continuous features into categorical bins using quantiles.
    Bin edges are saved for potential future reference.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        feature_cols (list): List of feature column names to discretize.
        bins (int): Number of quantile bins.
        
    Returns:
        tuple: (DataFrame with binned features, dictionary of bin edges)
    """
    bin_edges = {}
    for col in feature_cols:
        df[col + '_binned'], bin_edges[col] = pd.qcut(df[col], q=bins, retbins=True, labels=False, duplicates='drop')
    return df, bin_edges

def generate_cpts(df, feature_cols):
    """
    Generate Conditional Probability Tables (CPTs) for the binned features.
    For each feature, counts are normalized to form probability tables.
    
    Parameters:
        df (pd.DataFrame): DataFrame with binned features.
        feature_cols (list): List of original numerical feature column names.
        
    Returns:
        dict: A dictionary where keys are feature names and values are the CPTs.
    """
    cpts = {}
    for col in feature_cols:
        binned_col = col + '_binned'
        counts = df[binned_col].value_counts().sort_index()
        total = counts.sum()
        probs = (counts / total).to_dict()
        cpts[col] = probs
    return cpts

def save_processed_data(df, output_path):
    """
    Save the processed DataFrame to a CSV file.
    
    Parameters:
        df (pd.DataFrame): DataFrame to save.
        output_path (str): Path to the output CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

def save_cpts(cpts, output_path):
    """
    Save the Conditional Probability Tables (CPTs) to a JSON file.
    
    Parameters:
        cpts (dict): CPTs dictionary.
        output_path (str): Path to the output JSON file.
    """
    with open(output_path, 'w') as f:
        json.dump(cpts, f, indent=4)
    print(f"CPTs saved to {output_path}")

def main():
    # # Define file paths
    # raw_data_path = os.path.join('data', 'raw', 'stock_data.csv')
    # processed_data_path = os.path.join('data', 'processed', 'processed_stock_data.csv')
    # cpts_path = os.path.join('data', 'processed', 'CPTs.json')
    
    # # Load raw data
    # df = load_data(raw_data_path)
    # print("Data loaded. Shape:", df.shape)
    
    # # Impute missing values
    # df = impute_missing(df)
    
    # # Identify numerical columns for transformation and scaling
    # numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # # Apply logarithmic transformation to reduce skewness
    # df = log_transform(df, numerical_cols)
    
    # # Scale numerical features
    # df, scaler = scale_features(df, numerical_cols)
    
    # # Discretize numerical features for CPT generation
    # df, bin_edges = discretize_features(df, numerical_cols, bins=5)
    
    # # Generate Conditional Probability Tables (CPTs)
    # cpts = generate_cpts(df, numerical_cols)
    
    # # Ensure the output directory exists
    # os.makedirs(os.path.join('data', 'processed'), exist_ok=True)
    
    # # Save processed data and CPTs
    # save_processed_data(df, processed_data_path)
    # save_cpts(cpts, cpts_path)

    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("szrlee/stock-time-series-20050101-to-20171231")
    
    print("Path to dataset files:", path)

if __name__ == '__main__':
    main()
