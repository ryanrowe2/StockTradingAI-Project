import pandas as pd
import numpy as np

def compute_moving_average(df, window=20, column='Close'):
    """
    Compute the simple moving average (SMA) for a given window.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing price data.
        window (int): Window size for the moving average.
        column (str): Column name on which to compute the SMA.
    
    Returns:
        pd.Series: The SMA of the specified column.
    """
    return df[column].rolling(window=window).mean()

def compute_rsi(df, window=14, column='Close'):
    """
    Compute the Relative Strength Index (RSI) for a given window.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing price data.
        window (int): Window size for the RSI.
        column (str): Column name on which to compute the RSI.
    
    Returns:
        pd.Series: The RSI values.
    """
    delta = df[column].diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = -delta.clip(upper=0).rolling(window=window).mean()
    rs = gain / (loss + 1e-5)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(df, column='Close', fast=12, slow=26, signal=9):
    """
    Compute the MACD (Moving Average Convergence Divergence) and its signal line.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing price data.
        column (str): Column name on which to compute MACD.
        fast (int): Fast EMA span.
        slow (int): Slow EMA span.
        signal (int): Signal line EMA span.
    
    Returns:
        tuple: (MACD line, Signal line) as pd.Series.
    """
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def add_technical_indicators(df):
    """
    Add technical indicators (MA, RSI, MACD, and MACD Signal) to the DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame with at least a 'Close' column.
    
    Returns:
        pd.DataFrame: Updated DataFrame with new indicator columns.
    """
    df['MA_20'] = compute_moving_average(df, window=20, column='Close')
    df['RSI_14'] = compute_rsi(df, window=14, column='Close')
    df['MACD'], df['MACD_Signal'] = compute_macd(df, column='Close')
    return df
