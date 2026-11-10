"""Load historical equity data and create causal, per-ticker model features.

This stage never fits global scalers, quantile bins, or an HMM. Those learned
transformations belong inside training folds, after the chronological split.
"""

from __future__ import annotations

import argparse
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .technical_indicators import add_technical_indicators
    from .date_validation import parse_daily_dates
    from .price_validation import valid_price_bars
    from .session_validation import session_positions, validate_session_positions
    from .ticker_validation import MISSING_TICKER_MARKERS as MISSING_TICKER_MARKERS
    from .ticker_validation import normalize_ticker_symbols, valid_ticker_symbols
else:
    from technical_indicators import add_technical_indicators
    from date_validation import parse_daily_dates
    from price_validation import valid_price_bars
    from session_validation import session_positions, validate_session_positions
    from ticker_validation import MISSING_TICKER_MARKERS as MISSING_TICKER_MARKERS
    from ticker_validation import normalize_ticker_symbols, valid_ticker_symbols


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DATA = PROJECT_ROOT / "data/raw/all_stocks_2006-01-01_to_2018-01-01.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/generated/stock_features.csv"
REQUIRED_COLUMNS = ("Date", "Name", "Open", "High", "Low", "Close", "Volume")
MODEL_FEATURES = ("Return", "Intraday_Return", "Daily_Range", "MACD_Pct")
TARGET_DEFINITIONS = {
    "close_to_close": "Next session close is above the decision-session close",
    "next_open_to_close": "Next session close is above its opening price",
}
SESSION_COLUMN = "Session_Number"
_BUNDLED_SYMBOLS = (
    "AABA", "AAPL", "AMZN", "AXP", "BA", "CAT", "CSCO", "CVX", "DIS",
    "GE", "GOOGL", "GS", "HD", "IBM", "INTC", "JNJ", "JPM", "KO",
    "MCD", "MMM", "MRK", "MSFT", "NKE", "PFE", "PG", "TRV", "UNH",
    "UTX", "VZ", "WMT", "XOM",
)
_BUNDLED_RAW_NAMES = frozenset({
    DEFAULT_RAW_DATA.name,
    "all_stocks_2017-01-01_to_2018-01-01.csv",
    *(f"{ticker}_2006-01-01_to_2018-01-01.csv" for ticker in _BUNDLED_SYMBOLS),
})


@lru_cache(maxsize=4)
def _cached_bundled_session_calendar(path: Path, contents_hash: bytes) -> pd.DatetimeIndex:
    """Use source-observed market dates with valid ticker identities as authority.

    Individual CSVs contain ticker-specific holes, so their observed dates do
    not identify all trading sessions represented by the bundled dataset. A
    damaged price bar still identifies a potential session, whereas malformed
    dates and invalid ticker identifiers cannot be trusted for this purpose.
    """
    source = pd.read_csv(path, keep_default_na=False)
    usable_identifiers = valid_ticker_symbols(source["Name"])
    raw_dates = source.loc[usable_identifiers, "Date"].astype("string")
    explicit = raw_dates.str.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}").fillna(False)
    dates = pd.to_datetime(raw_dates.where(explicit), format="mixed", errors="coerce")
    dates = dates.loc[dates.notna() & dates.dt.dayofweek.lt(5)]
    return pd.DatetimeIndex(dates.unique()).sort_values()


def _bundled_session_calendar() -> pd.DatetimeIndex:
    # Key the cache to the bytes, not the filename or modification timestamp:
    # replacing the bundled source in a long-running interpreter cannot reuse
    # an obsolete calendar, even if its size and mtime are unchanged.
    return _cached_bundled_session_calendar(
        DEFAULT_RAW_DATA.resolve(), hashlib.sha256(DEFAULT_RAW_DATA.read_bytes()).digest(),
    )


def _is_bundled_raw_csv(filepath: str | Path) -> bool:
    if not isinstance(filepath, (str, Path)):
        return False
    path = Path(filepath).resolve()
    return (path.is_file() and path.parent == DEFAULT_RAW_DATA.parent.resolve()
            and path.name in _BUNDLED_RAW_NAMES)


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
        df[SESSION_COLUMN] = session_positions(df)
    validate_session_positions(df)
    valid_prices = valid_price_bars(df, require_volume=True)
    numeric_columns = ("Open", "High", "Low", "Close", "Volume")
    for name in numeric_columns:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    df = df.loc[valid_prices].copy()
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
        selected = {ticker.strip().upper() for ticker in tickers}
        normalized_names = source["Name"].astype("string").str.strip().str.upper()
        not_found = selected - set(normalized_names.dropna())
        if not_found:
            raise ValueError(f"Ticker not found: {', '.join(sorted(not_found))}")
        source = source.loc[normalized_names.isin(selected)].copy()
    if SESSION_COLUMN not in source and _is_bundled_raw_csv(filepath):
        # Both the complete source and every bundled per-symbol CSV must use
        # exactly the same reference calendar, even on dates absent for a symbol.
        dates = parse_daily_dates(source["Date"], context="Every Date")
        positions = _bundled_session_calendar().get_indexer(dates)
        if (positions < 0).any():
            raise ValueError("Bundled observations contain non-market dates")
        source[SESSION_COLUMN] = positions
    return validate_bars(source)


def prepare_features(df: pd.DataFrame, *,
                     features: tuple[str, ...] = MODEL_FEATURES,
                     target: str = "close_to_close") -> pd.DataFrame:
    """Produce causal next-session targets with only requested model features required.

    ``next_open_to_close`` aligns the target with an order submitted after the
    current close and filled at the next open. The historical close-to-close
    label remains the default for backward-compatible published experiments.
    """
    if target not in TARGET_DEFINITIONS:
        raise ValueError(f"target must be one of {sorted(TARGET_DEFINITIONS)}")
    source = prepare_indicator_history(df)
    next_close = source.groupby("Name", sort=False)["Close"].shift(-1)
    next_open = source.groupby("Name", sort=False)["Open"].shift(-1)
    next_session = source.groupby("Name", sort=False)[SESSION_COLUMN].shift(-1)
    adjacent_target = next_session.sub(source[SESSION_COLUMN]).eq(1)
    source["Target_Date"] = (source.groupby("Name", sort=False)["Date"]
                             .shift(-1).where(adjacent_target))
    reference_price = source["Close"] if target == "close_to_close" else next_open
    source["Trend"] = (next_close > reference_price).astype("int64")
    if not features or any(feature not in MODEL_FEATURES for feature in features):
        raise ValueError(f"Model features must be drawn from {MODEL_FEATURES}")
    required = list(features) + ["Target_Date"]
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
    count = result.Name.nunique()
    print(f"Prepared {len(result):,} observations from {count} {'ticker' if count == 1 else 'tickers'}: {args.output}")


if __name__ == "__main__":
    main()
