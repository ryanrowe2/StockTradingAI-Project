"""Consistent, overflow-safe validation of observed market-session positions."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd

if __package__:
    from .market_calendar import inferred_equity_sessions
else:
    from market_calendar import inferred_equity_sessions


SESSION_COLUMN = "Session_Number"


def session_positions(frame: pd.DataFrame) -> pd.Series:
    """Return supplied ordinals or exchange-aware inferred equity positions.

    In date-only U.S. equity inputs, genuine NYSE holidays do not create gaps.
    Missing regular exchange weekdays still do. Dates explicitly observed for
    each ticker are included to support other weekday-trading exchanges without
    contaminating another ticker's calendar. Exact exchange-specific adjacency
    may be supplied as ``Session_Number``.
    """
    if SESSION_COLUMN in frame:
        return frame[SESSION_COLUMN]
    if "Date" not in frame:
        return pd.Series(np.arange(len(frame), dtype=np.int64), index=frame.index)
    dates = pd.DatetimeIndex(frame["Date"])
    if (dates.dayofweek >= 5).any():
        raise ValueError("Trading dates must fall on weekdays; supply a verified market calendar")
    if dates.empty:
        return pd.Series(dtype=np.int64, index=frame.index)
    if "Name" not in frame:
        sessions = inferred_equity_sessions(dates)
        return pd.Series(sessions.get_indexer(dates), index=frame.index, dtype=np.int64)
    # Use a source-wide date range for comparable NYSE ordinals, but include
    # exchange-specific observed weekdays only for the ticker that traded them.
    positions = np.empty(len(frame), dtype=np.int64)
    for indices in frame.groupby("Name", sort=False, dropna=False).indices.values():
        symbol_dates = dates.take(indices)
        sessions = inferred_equity_sessions(
            symbol_dates, start=dates.min(), end=dates.max(),
        )
        positions[indices] = sessions.get_indexer(symbol_dates)
    return pd.Series(positions, index=frame.index, dtype=np.int64)


def validate_session_positions(frame: pd.DataFrame, *,
                               require_consecutive: bool = False) -> None:
    """Validate source-market ordinals without unsigned-integer subtraction.

    Within each ticker, a date identifies one market session and an ordinal
    identifies one date. Different exchanges may assign different ordinals to
    the same date. Subsets may omit sessions unless ``require_consecutive`` is
    requested.
    """
    if SESSION_COLUMN not in frame:
        return

    positions = frame[SESSION_COLUMN]
    if (positions.isna().any()
            or not pd.api.types.is_integer_dtype(positions.dtype)
            or pd.api.types.is_bool_dtype(positions.dtype)
            or (positions < 0).any()):
        raise ValueError("Session_Number must contain nonnegative integer market positions")

    if "Date" in frame:
        dates = pd.DatetimeIndex(frame["Date"])
        if (dates.dayofweek >= 5).any():
            raise ValueError("Trading dates must fall on weekdays even with Session_Number")
        groups = (group for _, group in frame.groupby("Name", sort=False, dropna=False)) \
            if "Name" in frame else (frame,)
        for group in groups:
            calendar = group[["Date", SESSION_COLUMN]].drop_duplicates()
            if (calendar["Date"].duplicated().any()
                    or calendar[SESSION_COLUMN].duplicated().any()):
                raise ValueError("Session_Number must identify exactly one market date per ticker")
            chronological = calendar.sort_values("Date")[SESSION_COLUMN].tolist()
            if any(later <= earlier for earlier, later in pairwise(chronological)):
                raise ValueError("Session_Number must increase with the market dates per ticker")

    # When Date is absent, the caller's row order is the chronological order.
    # With Date, verify adjacency inside each ticker, not across different names.
    if require_consecutive:
        if "Name" in frame:
            groups = (group for _, group in frame.groupby("Name", sort=False))
        else:
            groups = (frame,)
        for group in groups:
            ordered = group.sort_values("Date") if "Date" in group else group
            sequence = ordered[SESSION_COLUMN].tolist()
            if any(later != earlier + 1 for earlier, later in pairwise(sequence)):
                raise ValueError("Trading observations must occupy consecutive market sessions")
    elif "Date" not in frame:
        sequence = positions.tolist()
        if any(later <= earlier for earlier, later in pairwise(sequence)):
            raise ValueError("Session_Number must increase with the observation order")
