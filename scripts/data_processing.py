import pandas as pd
import numpy as np
import os
import json
import glob
from sklearn.preprocessing import StandardScaler
from technical_indicators import add_technical_indicators
from hmm_integration import fit_hmm_on_indicators
from hmmlearn.hmm import GaussianHMM  # Updated HMM class import

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
        # Reassign the filled series to avoid chained assignment warnings
        df[col] = df[col].fillna(df[col].mean())
    return df

def log_transform(df, feature_cols):
    """
    Apply a logarithmic transformation to reduce skewness in numerical features.
    
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
    
    Parameters:
        df (pd.DataFrame): DataFrame with binned features.
        feature_cols (list): List of original numerical feature column names.
        
    Returns:
        dict: A dictionary of CPTs.
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
    raw_folder = os.path.join('..', 'data', 'raw')
    processed_folder = os.path.join('..', 'data', 'processed')
    os.makedirs(processed_folder, exist_ok=True)
    csv_files = glob.glob(os.path.join(raw_folder, "*.csv"))
    if not csv_files:
        print("No CSV files found in the raw folder.")
        return

    for file in csv_files:
        print(f"\nProcessing file: {file}")
        df = load_data(file)
        print("Data loaded. Shape:", df.shape)
        df = impute_missing(df)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df = log_transform(df, numerical_cols)
        df, scaler = scale_features(df, numerical_cols)
        
        # Compute technical indicators and add them as new columns.
        df = add_technical_indicators(df)
        
        # Define the indicators for HMM fitting and drop rows with NaN in these columns
        indicators = ['MA_20', 'RSI_14', 'MACD']
        df = df.dropna(subset=indicators)
        
        # Attempt to fit HMM using the provided integration function.
        try:
            _, df = fit_hmm_on_indicators(df, indicators=indicators)
        except AttributeError:
            print("fit_hmm_on_indicators encountered an AttributeError. Falling back to manual HMM fitting.")
            X = df[indicators].values
            model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000)
            model.fit(X)
            latent_states = model.predict(X)
            df['Market_Regime'] = latent_states
        
        df, bin_edges = discretize_features(df, numerical_cols, bins=5)
        cpts = generate_cpts(df, numerical_cols)
        filename = os.path.basename(file)
        processed_file_path = os.path.join(processed_folder, filename)
        cpts_file_path = os.path.join(processed_folder, filename.replace(".csv", "_CPTs.json"))
        save_processed_data(df, processed_file_path)
        save_cpts(cpts, cpts_file_path)

if __name__ == '__main__':
    main()
