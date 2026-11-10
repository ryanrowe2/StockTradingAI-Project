"""Causal, per-symbol technical indicators for daily equity data."""

from __future__ import annotations

import numpy as np
import pandas as pd

if __package__:
    from .date_validation import parse_daily_dates
    from .price_validation import valid_price_bars
    from .session_validation import adjacent_session_flags, session_positions, validate_session_positions
    from .ticker_validation import normalize_ticker_symbols
else:
    from date_validation import parse_daily_dates
    from price_validation import valid_price_bars
    from session_validation import adjacent_session_flags, session_positions, validate_session_positions
    from ticker_validation import normalize_ticker_symbols


def compute_moving_average(df: pd.DataFrame, window: int = 20, column: str = "Close") -> pd.Series:
    return df[column].rolling(window=window, min_periods=window).mean()


def compute_rsi(df: pd.DataFrame, window: int = 14, column: str = "Close",
                *, changes: pd.Series | None = None) -> pd.Series:
    change = df[column].diff() if changes is None else changes
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
    adjacent = adjacent_session_flags(session_positions(frame))
    # The first valid bar after a discarded market session has no known
    # single-session close-to-close change. Do not fabricate one from two
    # nonadjacent observed closes.
    change = frame["Close"].diff().where(adjacent)
    true_range = pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - previous_close).abs(),
            (frame["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1).where(adjacent | previous_close.isna())
    frame["ATR"] = true_range.rolling(14, min_periods=14).mean()
    frame["MA_20"] = compute_moving_average(frame)
    # RSI must also avoid treating a multi-session price move as one day.
    frame["RSI_14"] = compute_rsi(frame, changes=change)
    frame["MACD"], frame["MACD_Signal"] = compute_macd(frame)
    frame["BB_Middle"] = frame["MA_20"]
    std = frame["Close"].rolling(20, min_periods=20).std()
    frame["BB_Upper"] = frame["BB_Middle"] + 2 * std
    frame["BB_Lower"] = frame["BB_Middle"] - 2 * std
    frame["Close_Lag1"] = previous_close
    frame["Return"] = frame["Close"].pct_change(fill_method=None).where(adjacent)
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
    if df.empty:
        raise ValueError("A nonempty indicator DataFrame is required")
    df = df.copy()
    if not valid_price_bars(df, require_volume="Volume" in df).all():
        raise ValueError("Indicator OHLC prices must be coherent and finite; supplied Volume must be nonnegative")
    # The shared validator accepts numeric CSV strings; calculate on the
    # normalized numbers rather than attempting subtraction on strings.
    for column in ("Open", "High", "Low", "Close", "Volume"):
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="raise")
    if "Date" in df:
        df["Date"] = parse_daily_dates(df["Date"], context="Indicator dates")
    if "Name" not in df.columns:
        if "Date" in df and df["Date"].duplicated().any():
            raise ValueError("Duplicate trading dates in indicator input")
        validate_session_positions(df)
        return _one_symbol_indicators(df.sort_values("Date") if "Date" in df else df)
    if "Date" not in df:
        raise ValueError("Date is required for multi-ticker indicators")
    df["Name"] = normalize_ticker_symbols(
        df["Name"], context="Every indicator observation must have a ticker name",
    )
    if df.duplicated(["Name", "Date"]).any():
        raise ValueError("Duplicate ticker/date observations in indicator input")
    validate_session_positions(df)
    # Infer on the complete input before grouping by ticker. A date observed
    # for a peer on the same Exchange can expose a missing session here.
    if "Session_Number" not in df:
        df["Session_Number"] = session_positions(df)
    return pd.concat(
        [
            _one_symbol_indicators(group.sort_values("Date"))
            for _, group in df.groupby("Name", sort=True)
        ],
        axis=0,
    ).sort_values(["Date", "Name"]).reset_index(drop=True)
