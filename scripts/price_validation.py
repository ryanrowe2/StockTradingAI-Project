"""Shared, non-mutating validation of observed daily OHLCV prices."""

from __future__ import annotations

import numpy as np
import pandas as pd


PRICE_COLUMNS = ("Open", "High", "Low", "Close")


def valid_price_bars(frame: pd.DataFrame, *, require_volume: bool = False) -> pd.Series:
    """Identify finite, positive and internally coherent price observations.

    A direct indicator calculation can omit Volume, whereas imported OHLCV
    records must include a finite, nonnegative Volume field.
    """
    required = (*PRICE_COLUMNS, "Volume") if require_volume else PRICE_COLUMNS
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required price columns: {sorted(missing)}")
    numeric = frame[list(required)].apply(pd.to_numeric, errors="coerce")
    # Python booleans are subclasses of integers, but ``True`` is not a
    # legitimate OHLC price or trading volume. Do not turn it into 1.0.
    booleans = (
        frame[list(required)]
        .apply(
            lambda column: column.map(lambda value: isinstance(value, (bool, np.bool_)))
        )
        .any(axis=1)
    )
    finite = pd.Series(
        np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1), index=frame.index
    )
    positive = numeric[list(PRICE_COLUMNS)].gt(0).all(axis=1)
    coherent = numeric["High"].ge(numeric[list(PRICE_COLUMNS)].max(axis=1)) & numeric[
        "Low"
    ].le(numeric[list(PRICE_COLUMNS)].min(axis=1))
    if require_volume:
        positive &= numeric["Volume"].ge(0)
    return finite & positive & coherent & ~booleans
