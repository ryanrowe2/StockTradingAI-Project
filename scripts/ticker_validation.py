"""Shared validation and normalization for stock ticker identifiers."""

from __future__ import annotations

import pandas as pd


MISSING_TICKER_MARKERS = frozenset({"", "N/A", "NAN", "NULL", "NONE", "<NA>"})


def valid_ticker_symbols(symbols: pd.Series) -> pd.Series:
    """Identify valid source identifiers without rejecting unrelated rows."""
    normalized = symbols.astype("string").str.strip().str.upper()
    valid_format = normalized.str.fullmatch(r"[A-Z][A-Z0-9]*(?:[.-][A-Z0-9]+)*")
    return (valid_format.fillna(False) & ~normalized.isin(MISSING_TICKER_MARKERS)).astype(bool)


def normalize_ticker_symbols(symbols: pd.Series, *, context: str) -> pd.Series:
    """Preserve valid tickers (including ``NA``) while rejecting placeholders."""
    if symbols.isna().any():
        raise ValueError(context)
    normalized = symbols.astype(str).str.strip().str.upper()
    if not valid_ticker_symbols(symbols).all():
        raise ValueError(context)
    return normalized
