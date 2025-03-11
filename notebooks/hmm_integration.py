import pandas as pd
from hmmlearn import hmm

def fit_hmm_on_indicators(df, indicators=['MA_20', 'RSI_14', 'MACD'], n_components=3):
    """
    Fit a Gaussian HMM on selected technical indicators to infer market regimes.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing technical indicator columns.
        indicators (list): List of indicator column names to use.
        n_components (int): Number of latent states (market regimes).
    
    Returns:
        model: The fitted HMM model.
        df (pd.DataFrame): The DataFrame with an added 'Market_Regime' column.
    """
    # Drop rows with missing values in the selected indicators
    df_ind = df[indicators].dropna()
    X = df_ind.values  # Data as numpy array
    
    # Initialize and fit the Gaussian HMM
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100, random_state=42)
    latent_states = model.fit_predict(X)
    
    # Create a Series for latent states and align with original index
    states_series = pd.Series(latent_states, index=df_ind.index, name='Market_Regime')
    df = df.join(states_series)
    
    # Forward-fill missing regime values (from rows dropped due to NaNs)
    df['Market_Regime'].fillna(method='ffill', inplace=True)
    return model, df
