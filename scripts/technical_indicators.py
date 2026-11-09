"""Causal, per-symbol technical indicators for daily equity data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_moving_average(df: pd.DataFrame, window: int = 20, column: str = "Close") -> pd.Series:
    return df[column].rolling(window=window, min_periods=window).mean()


def compute_rsi(df: pd.DataFrame, window: int = 14, column: str = "Close") -> pd.Series:
    change = df[column].diff()
    gains = change.clip(lower=0).rolling(window, min_periods=window).mean()
    losses = (-change.clip(upper=0)).rolling(window, min_periods=window).mean()
    rsi = 100 - 100 / (1 + gains / losses.replace(0, np.nan))
    rsi = rsi.where(losses.ne(0), np.where(gains.gt(0), 100.0, 50.0))
    return rsi.where(gains.notna() & losses.notna())


def compute_macd(
    df: pd.DataFrame, column: str = "Close", fast: int = 12,
    slow: int = 26, signal: int = 9,
) -> tuple[pd.Series, pd.Series]:
    close = df[column]
    fast_ema = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd, signal_line


def _one_symbol_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    previous_close = frame["Close"].shift(1)
    true_range = pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - previous_close).abs(),
            (frame["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["ATR"] = true_range.rolling(14, min_periods=14).mean()
    frame["MA_20"] = compute_moving_average(frame)
    frame["RSI_14"] = compute_rsi(frame)
    frame["MACD"], frame["MACD_Signal"] = compute_macd(frame)
    frame["BB_Middle"] = frame["MA_20"]
    std = frame["Close"].rolling(20, min_periods=20).std()
    frame["BB_Upper"] = frame["BB_Middle"] + 2 * std
    frame["BB_Lower"] = frame["BB_Middle"] - 2 * std
    frame["Close_Lag1"] = previous_close
    frame["Return"] = frame["Close"].pct_change(fill_method=None)
    frame["Intraday_Return"] = (frame["Close"] - frame["Open"]) / frame["Open"]
    frame["Daily_Range"] = (frame["High"] - frame["Low"]) / frame["Open"]
    frame["MACD_Pct"] = frame["MACD"] / frame["Close"]
    frame["Return_Lag1"] = frame["Return"].shift(1)
    return frame


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate indicators from each symbol's past and current observations only.

    Rows are sorted chronologically within each symbol. No rolling window,
    shift, or exponential average crosses a symbol boundary.
    """
    if "Name" not in df.columns:
        return _one_symbol_indicators(df.sort_values("Date") if "Date" in df else df)
    return pd.concat(
        [
            _one_symbol_indicators(group.sort_values("Date"))
            for _, group in df.groupby("Name", sort=True)
        ],
        axis=0,
    ).sort_values(["Date", "Name"]).reset_index(drop=True)
