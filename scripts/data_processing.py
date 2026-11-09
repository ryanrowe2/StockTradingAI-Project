"""Load historical equity data and create causal, per-ticker model features.

This stage never fits global scalers, quantile bins, or an HMM. Those learned
transformations belong inside training folds, after the chronological split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .technical_indicators import add_technical_indicators
else:
    from technical_indicators import add_technical_indicators


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DATA = PROJECT_ROOT / "data/raw/all_stocks_2006-01-01_to_2018-01-01.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/generated/stock_features.csv"
REQUIRED_COLUMNS = ("Date", "Name", "Open", "High", "Low", "Close", "Volume")
MODEL_FEATURES = ("Return", "Intraday_Return", "Daily_Range", "MACD_Pct")


def validate_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Retain complete, coherent daily price bars with verified dates and symbols.

    A contradictory OHLC record cannot be repaired reliably without an
    independent price source, so it is excluded from reproducible experiments.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
    if df["Date"].isna().any():
        raise ValueError("Every Date must be present and parseable")
    if df["Name"].isna().any() or df["Name"].astype(str).str.strip().eq("").any():
        raise ValueError("Every observation must have a ticker symbol")
    df["Name"] = df["Name"].astype(str).str.strip().str.upper()
    if df.duplicated(["Name", "Date"]).any():
        raise ValueError("Duplicate ticker/date observations")
    numeric_columns = ("Open", "High", "Low", "Close", "Volume")
    for name in numeric_columns:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    df = df.dropna(subset=numeric_columns).copy()
    finite = pd.Series(
        np.isfinite(df[list(numeric_columns)].to_numpy(dtype=float)).all(axis=1),
        index=df.index,
    )
    positive = (
        (df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
        & df["Volume"].ge(0)
    )
    coherent = (
        df["High"].ge(df[["Open", "Low", "Close"]].max(axis=1))
        & df["Low"].le(df[["Open", "High", "Close"]].min(axis=1))
    )
    df = df.loc[finite & positive & coherent].copy()
    if df.empty:
        raise ValueError("No valid OHLCV observations remain")
    return df.sort_values(["Name", "Date"]).reset_index(drop=True)


def load_data(filepath: str | Path) -> pd.DataFrame:
    """Read and validate raw stock observations in their actual price units."""
    return validate_bars(pd.read_csv(filepath))


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Produce one next-observed-day target per symbol, with causal indicators."""
    source = validate_bars(df)
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
