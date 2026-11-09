"""Load historical equity data and create causal, per-ticker model features.

This stage never fits global scalers, quantile bins, or an HMM. Those learned
transformations belong inside training folds, after the chronological split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from technical_indicators import add_technical_indicators


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DATA = PROJECT_ROOT / "data/raw/all_stocks_2006-01-01_to_2018-01-01.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/generated/stock_features.csv"
REQUIRED_COLUMNS = ("Date", "Name", "Open", "High", "Low", "Close", "Volume")
MODEL_FEATURES = ("Open", "High", "Low", "ATR", "Return")


def load_data(filepath: str | Path) -> pd.DataFrame:
    """Validate raw observations; retain their actual, positive price units."""
    df = pd.read_csv(filepath, parse_dates=["Date"])
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df[["Date", "Name"]].isna().any().any():
        raise ValueError("Date and Name must be present for every observation")
    for name in ("Open", "High", "Low", "Close", "Volume"):
        df[name] = pd.to_numeric(df[name], errors="coerce")
    # Dropping a handful of incomplete bars is safer than imputing from future
    # observations or inventing a price for the current trading day.
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy()
    df = df.loc[
        (df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
        & df["Volume"].ge(0)
    ]
    if df.duplicated(["Name", "Date"]).any():
        raise ValueError("Duplicate ticker/date observations")
    return df.sort_values(["Name", "Date"]).reset_index(drop=True)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Produce one next-observed-day target per symbol, with causal indicators."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    source = df.copy()
    source["Date"] = pd.to_datetime(source["Date"])
    source = source.sort_values(["Name", "Date"]).reset_index(drop=True)
    if source.duplicated(["Name", "Date"]).any():
        raise ValueError("Duplicate ticker/date observations")
    source = add_technical_indicators(source)
    next_close = source.groupby("Name", sort=False)["Close"].shift(-1)
    source["Target_Date"] = source.groupby("Name", sort=False)["Date"].shift(-1)
    source["Trend"] = (next_close > source["Close"]).astype("int64")
    required = list(MODEL_FEATURES) + [
        "Target_Date", "MA_20", "RSI_14",
        "MACD", "MACD_Signal", "BB_Middle", "BB_Upper", "BB_Lower",
        "Close_Lag1", "Return_Lag1",
    ]
    source = source.replace([np.inf, -np.inf], np.nan)
    return source.dropna(subset=required).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare causal stock features")
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ticker", type=str, default=None)
    args = parser.parse_args()
    data = load_data(args.input)
    if args.ticker:
        data = data.loc[data["Name"] == args.ticker.upper()].copy()
        if data.empty:
            raise ValueError(f"Ticker not found: {args.ticker}")
    result = prepare_features(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Prepared {len(result):,} observations from {result.Name.nunique()} tickers: {args.output}")


if __name__ == "__main__":
    main()
