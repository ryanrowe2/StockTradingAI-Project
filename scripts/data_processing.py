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
    from .session_validation import validate_session_positions
    from .ticker_validation import MISSING_TICKER_MARKERS as MISSING_TICKER_MARKERS
    from .ticker_validation import normalize_ticker_symbols
else:
    from technical_indicators import add_technical_indicators
    from date_validation import parse_daily_dates
    from price_validation import valid_price_bars
    from session_validation import validate_session_positions
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
    if SESSION_COLUMN not in df:
        calendar = pd.Index(df["Date"].unique()).sort_values()
        df[SESSION_COLUMN] = calendar.get_indexer(df["Date"])
    validate_session_positions(df)
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
    """Read prices, preserving the source-wide calendar when selecting symbols."""
    # Stock symbols such as "NA" are valid identifiers. Pandas' default NA
    # tokens would silently turn them into missing ticker names.
    source = pd.read_csv(filepath, keep_default_na=False)
    if tickers is not None:
        if not tickers or any(not isinstance(ticker, str) or not ticker.strip()
                              for ticker in tickers):
            raise ValueError("Selected tickers must be nonempty strings")
        if "Name" not in source:
            raise ValueError("Missing required columns: ['Name']")
        # Other tickers' malformed OHLCV rows or dates must not invalidate a
        # selected ticker. Their *valid* source dates nevertheless identify
        # real market sessions and must remain in its calendar.
        if "Date" in source and SESSION_COLUMN not in source:
            source_dates = source["Date"].astype("string")
            iso_dates = source_dates.str.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}").fillna(False)
            observed_dates = pd.to_datetime(
                source_dates.loc[iso_dates], format="mixed", errors="coerce",
            ).dropna()
            source_calendar = pd.DatetimeIndex(observed_dates.unique()).sort_values()
        else:
            source_calendar = None
        selected = {ticker.strip().upper() for ticker in tickers}
        normalized_names = source["Name"].astype("string").str.strip().str.upper()
        not_found = selected - set(normalized_names.dropna())
        if not_found:
            raise ValueError(f"Ticker not found: {', '.join(sorted(not_found))}")
        source = source.loc[normalized_names.isin(selected)].copy()
        if source_calendar is not None:
            selected_dates = parse_daily_dates(source["Date"], context="Every Date")
            source[SESSION_COLUMN] = source_calendar.get_indexer(selected_dates)
    return validate_bars(source)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Produce one valid next-market-session target per symbol with causal indicators."""
    source = prepare_indicator_history(df)
    next_close = source.groupby("Name", sort=False)["Close"].shift(-1)
    next_session = source.groupby("Name", sort=False)[SESSION_COLUMN].shift(-1)
    adjacent_target = next_session.sub(source[SESSION_COLUMN]).eq(1)
    source["Target_Date"] = (source.groupby("Name", sort=False)["Date"]
                             .shift(-1).where(adjacent_target))
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
    because their next-session prediction target is unavailable. It also retains
    missing-indicator rows to preserve HMM sequence boundaries.
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
