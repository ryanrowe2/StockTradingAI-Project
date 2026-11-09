"""Shared validation and normalization for stock ticker identifiers."""

from __future__ import annotations

import pandas as pd


MISSING_TICKER_MARKERS = frozenset({"", "N/A", "NAN", "NULL", "NONE"})


def normalize_ticker_symbols(symbols: pd.Series, *, context: str) -> pd.Series:
    """Preserve valid tickers (including ``NA``) while rejecting placeholders."""
    if symbols.isna().any():
        raise ValueError(context)
    normalized = symbols.astype(str).str.strip().str.upper()
    if normalized.isin(MISSING_TICKER_MARKERS).any():
        raise ValueError(context)
    return normalized
