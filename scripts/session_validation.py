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


def _normalized_exchange_labels(frame: pd.DataFrame) -> pd.Series | None:
    """Validate optional exchange identities used to share market calendars."""
    if "Exchange" not in frame:
        return None
    values = frame["Exchange"]
    if (values.isna().any() or not values.map(lambda value: isinstance(value, str)).all()):
        raise ValueError("Exchange must contain nonempty exchange identifiers")
    normalized = values.str.strip().str.upper()
    if normalized.isin(("", "N/A", "NAN", "NULL", "NONE", "<NA>")).any():
        raise ValueError("Exchange must contain nonempty exchange identifiers")
    if "Name" in frame:
        assignments = pd.DataFrame({"Name": frame["Name"], "Exchange": normalized})
        if assignments.groupby("Name", dropna=False)["Exchange"].nunique().gt(1).any():
            raise ValueError("Each ticker must belong to exactly one Exchange")
    return normalized


def adjacent_session_flags(positions: pd.Series, *, forward: bool = False) -> pd.Series:
    """Test adjacent session ordinals using Python integers, without float conversion."""
    numbers = positions.tolist()
    adjacent = [int(later) == int(earlier) + 1
                for earlier, later in pairwise(numbers)]
    values = adjacent + [False] if forward else [False] + adjacent
    return pd.Series(values, index=positions.index, dtype=bool)


def session_positions(frame: pd.DataFrame, *,
                      calendar_start: pd.Timestamp | None = None,
                      calendar_end: pd.Timestamp | None = None) -> pd.Series:
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
    exchange_labels = _normalized_exchange_labels(frame)
    start = dates.min() if calendar_start is None else calendar_start
    end = dates.max() if calendar_end is None else calendar_end
    if "Name" not in frame and exchange_labels is None:
        sessions = inferred_equity_sessions(dates, start=start, end=end)
        return pd.Series(sessions.get_indexer(dates), index=frame.index, dtype=np.int64)
    # Use a source-wide date range for comparable NYSE ordinals. Additional
    # observed weekdays are shared within a declared exchange, or only within
    # each ticker when exchange membership is unspecified.
    positions = np.empty(len(frame), dtype=np.int64)
    grouping = exchange_labels if exchange_labels is not None else frame["Name"]
    for indices in frame.groupby(grouping, sort=False, dropna=False).indices.values():
        symbol_dates = dates.take(indices)
        sessions = inferred_equity_sessions(
            symbol_dates, start=start, end=end,
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
        exchange_labels = _normalized_exchange_labels(frame)
        if exchange_labels is not None:
            groups = (group for _, group in frame.groupby(exchange_labels, sort=False))
        elif "Name" in frame:
            groups = (group for _, group in frame.groupby("Name", sort=False, dropna=False))
        else:
            groups = (frame,)
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
        groups = (group for _, group in frame.groupby("Name", sort=False)) \
            if "Name" in frame else (frame,)
        for group in groups:
            sequence = group[SESSION_COLUMN].tolist()
            if any(later <= earlier for earlier, later in pairwise(sequence)):
                raise ValueError("Session_Number must increase with the observation order")
