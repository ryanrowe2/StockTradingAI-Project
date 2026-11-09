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
    from .date_validation import parse_daily_dates
    from .price_validation import valid_price_bars
    from .ticker_validation import MISSING_TICKER_MARKERS as MISSING_TICKER_MARKERS
    from .ticker_validation import normalize_ticker_symbols
else:
    from technical_indicators import add_technical_indicators
    from date_validation import parse_daily_dates
    from price_validation import valid_price_bars
    from ticker_validation import MISSING_TICKER_MARKERS as MISSING_TICKER_MARKERS
    from ticker_validation import normalize_ticker_symbols


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DATA = PROJECT_ROOT / "data/raw/all_stocks_2006-01-01_to_2018-01-01.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/generated/stock_features.csv"
REQUIRED_COLUMNS = ("Date", "Name", "Open", "High", "Low", "Close", "Volume")
MODEL_FEATURES = ("Return", "Intraday_Return", "Daily_Range", "MACD_Pct")
SESSION_COLUMN = "Session_Number"


def validate_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Retain complete, coherent daily price bars with verified dates and symbols.

    A contradictory OHLC record cannot be repaired reliably without an
    independent price source, so it is excluded from reproducible experiments.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.copy()
    df["Date"] = parse_daily_dates(df["Date"], context="Every Date")
    df["Name"] = normalize_ticker_symbols(
        df["Name"], context="Every observation must have a ticker symbol",
    )
    if df.duplicated(["Name", "Date"]).any():
        raise ValueError("Duplicate ticker/date observations")
    input_tickers = set(df["Name"])
    # Preserve the original market-session positions *before* excluding any
    # corrupt bar. A subsequent valid price must not become the missing bar's
    # fictional next trading session during return/HMM calculations.
    if SESSION_COLUMN in df:
        sessions = df[SESSION_COLUMN]
        if (sessions.isna().any()
                or not pd.api.types.is_integer_dtype(sessions.dtype)
                or isinstance(sessions.dtype, pd.CategoricalDtype)
                or (sessions < 0).any()
                or df.assign(_session=sessions).groupby("Date")["_session"].nunique().gt(1).any()):
            raise ValueError("Session_Number must contain consistent nonnegative integer market positions")
    else:
        calendar = pd.Index(df["Date"].unique()).sort_values()
        df[SESSION_COLUMN] = calendar.get_indexer(df["Date"])
    ordered = df.sort_values(["Name", "Date"])
    if ordered.groupby("Name")[SESSION_COLUMN].diff().dropna().le(0).any():
        raise ValueError("Session_Number must increase with the observation dates")
    numeric_columns = ("Open", "High", "Low", "Close", "Volume")
    for name in numeric_columns:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    df = df.loc[valid_price_bars(df, require_volume=True)].copy()
    if df.empty:
        raise ValueError("No valid OHLCV observations remain")
    missing_tickers = sorted(input_tickers - set(df["Name"]))
    if missing_tickers:
        raise ValueError(
            "No valid OHLCV observations remain for ticker(s) "
            f"{', '.join(missing_tickers)}"
        )
    return df.sort_values(["Name", "Date"]).reset_index(drop=True)


def load_data(filepath: str | Path, *, tickers: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Read and validate raw stock observations, optionally selecting symbols first."""
    # Stock symbols such as "NA" are valid identifiers. Pandas' default NA
    # tokens would silently turn them into missing ticker names.
    source = pd.read_csv(filepath, keep_default_na=False)
    if tickers is not None:
        if not tickers or any(not isinstance(ticker, str) or not ticker.strip()
                              for ticker in tickers):
            raise ValueError("Selected tickers must be nonempty strings")
        if "Name" not in source:
            raise ValueError("Missing required columns: ['Name']")
        selected = {ticker.strip().upper() for ticker in tickers}
        normalized_names = source["Name"].astype("string").str.strip().str.upper()
        not_found = selected - set(normalized_names.dropna())
        if not_found:
            raise ValueError(f"Ticker not found: {', '.join(sorted(not_found))}")
        source = source.loc[normalized_names.isin(selected)].copy()
    return validate_bars(source)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Produce one next-observed-day target per symbol, with causal indicators."""
    source = prepare_indicator_history(df)
    next_close = source.groupby("Name", sort=False)["Close"].shift(-1)
    source["Target_Date"] = source.groupby("Name", sort=False)["Date"].shift(-1)
    source["Trend"] = (next_close > source["Close"]).astype("int64")
    required = list(MODEL_FEATURES) + [
        "Target_Date", "MA_20", "RSI_14",
        "MACD", "MACD_Signal", "BB_Middle", "BB_Upper", "BB_Lower",
        "Close_Lag1", "Return_Lag1",
    ]
    source = source.replace([np.inf, -np.inf], np.nan)
    prepared = source.dropna(subset=required).reset_index(drop=True)
    if prepared.empty:
        raise ValueError(
            "No usable feature observations remain after indicator warm-up; "
            "provide a longer price history"
        )
    missing_tickers = sorted(set(source["Name"]) - set(prepared["Name"]))
    if missing_tickers:
        raise ValueError(
            "No usable feature observations remain for ticker(s) "
            f"{', '.join(missing_tickers)}; provide longer price histories"
        )
    return prepared


def prepare_indicator_history(df: pd.DataFrame) -> pd.DataFrame:
    """Retain validated, causally calculated price history, including its last bar.

    Unlike supervised feature preparation, this does not remove rows solely
    because their next-day prediction target is unavailable. It also retains
    missing-indicator rows so HMM filtering can advance through them.
    """
    return add_technical_indicators(validate_bars(df))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare causal stock features")
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ticker", type=str, default=None)
    args = parser.parse_args()
    data = load_data(args.input, tickers=(args.ticker,) if args.ticker is not None else None)
    result = prepare_features(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Prepared {len(result):,} observations from {result.Name.nunique()} tickers: {args.output}")


if __name__ == "__main__":
    main()
