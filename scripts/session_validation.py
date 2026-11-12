"""Consistent, overflow-safe validation of observed market-session positions."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd

if __package__:
    from .date_validation import (
        mark_verified_daily_dates,
        parse_daily_dates,
        require_verified_daily_date_source,
    )
    from .market_calendar import (
        _KNOWN_EXCHANGES,
        _calendar_bounds,
        inferred_equity_sessions,
        validate_exchange_open_dates,
    )
else:
    from date_validation import (
        mark_verified_daily_dates,
        parse_daily_dates,
        require_verified_daily_date_source,
    )
    from market_calendar import (
        _KNOWN_EXCHANGES,
        _calendar_bounds,
        inferred_equity_sessions,
        validate_exchange_open_dates,
    )


SESSION_COLUMN = "Session_Number"


def _normalized_exchange_labels(frame: pd.DataFrame) -> pd.Series | None:
    """Validate optional exchange identities used to share market calendars."""
    if "Exchange" not in frame:
        return None
    values = frame["Exchange"]
    if (
        values.isna().any()
        or not values.map(lambda value: isinstance(value, str)).all()
    ):
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
    if not numbers:
        return pd.Series(index=positions.index, dtype=bool)
    adjacent = [int(later) == int(earlier) + 1 for earlier, later in pairwise(numbers)]
    values = adjacent + [False] if forward else [False] + adjacent
    return pd.Series(values, index=positions.index, dtype=bool)


def session_positions(
    frame: pd.DataFrame,
    *,
    calendar_start: pd.Timestamp | None = None,
    calendar_end: pd.Timestamp | None = None,
    verified_datetime_dates: bool = False,
) -> pd.Series:
    """Return supplied ordinals or exchange-aware inferred equity positions.

    Known exchanges use authoritative schedules; unknown exchanges must supply
    verified Session_Number positions. Already-coerced pandas dates require
    explicit source-precision attestation. The requested calendar must include
    every observed trading date.
    """
    require_verified_daily_date_source(
        frame, verified_datetime_dates=verified_datetime_dates
    )
    if SESSION_COLUMN in frame:
        validate_session_positions(
            frame, verified_datetime_dates=verified_datetime_dates
        )
        if "Date" in frame and (calendar_start is not None or calendar_end is not None):
            supplied_dates = pd.DatetimeIndex(
                parse_daily_dates(frame["Date"], context="Trading dates")
            )
            if not supplied_dates.empty:
                _calendar_bounds(supplied_dates, calendar_start, calendar_end)
        return frame[SESSION_COLUMN]
    if "Date" not in frame:
        return pd.Series(np.arange(len(frame), dtype=np.int64), index=frame.index)
    dates = pd.DatetimeIndex(parse_daily_dates(frame["Date"], context="Trading dates"))
    if (dates.dayofweek >= 5).any():
        raise ValueError(
            "Trading dates must fall on weekdays; supply a verified market calendar"
        )
    if dates.empty:
        return pd.Series(dtype=np.int64, index=frame.index)
    exchange_labels = _normalized_exchange_labels(frame)
    start, end = _calendar_bounds(dates, calendar_start, calendar_end)
    if "Name" not in frame and exchange_labels is None:
        sessions = inferred_equity_sessions(dates, start=start, end=end)
        indices = sessions.get_indexer(dates)
        if (indices < 0).any():
            raise ValueError(
                "Observed trading dates are absent from the requested calendar"
            )
        return pd.Series(indices, index=frame.index, dtype=np.int64)
    # Use a source-wide date range for comparable NYSE ordinals. Additional
    # observed weekdays are shared within a declared exchange, or only within
    # each ticker when exchange membership is unspecified.
    positions = np.empty(len(frame), dtype=np.int64)
    grouping = exchange_labels if exchange_labels is not None else frame["Name"]
    for label, indices in frame.groupby(
        grouping, sort=False, dropna=False
    ).indices.items():
        symbol_dates = dates.take(indices)
        sessions = inferred_equity_sessions(
            symbol_dates,
            start=start,
            end=end,
            exchange=str(label) if exchange_labels is not None else None,
        )
        inferred = sessions.get_indexer(symbol_dates)
        if (inferred < 0).any():
            raise ValueError(
                "Observed trading dates are absent from the requested calendar"
            )
        positions[indices] = inferred
    return pd.Series(positions, index=frame.index, dtype=np.int64)


def validate_session_positions(
    frame: pd.DataFrame,
    *,
    require_consecutive: bool = False,
    verified_datetime_dates: bool = False,
) -> None:
    """Validate source-market ordinals without unsigned-integer subtraction.

    Dates already converted to pandas timestamps require source-precision
    attestation. Each ticker's dates and ordinals must identify valid sessions;
    subsets may omit sessions unless ``require_consecutive`` is requested.
    """
    require_verified_daily_date_source(
        frame, verified_datetime_dates=verified_datetime_dates
    )
    if "Date" in frame:
        # Check the daily identity before duplicate or ordinal comparisons.
        # Otherwise, two times (or differently formatted strings) on the same
        # calendar day can masquerade as separate trading sessions.
        frame = frame.copy()
        frame["Date"] = parse_daily_dates(frame["Date"], context="Trading dates")
        mark_verified_daily_dates(frame)
        identity = ["Name", "Date"] if "Name" in frame else ["Date"]
        if frame.duplicated(identity).any():
            raise ValueError("Duplicate ticker/date observations")

    if "Date" in frame:
        checked_dates = pd.DatetimeIndex(frame["Date"])
        exchanges = _normalized_exchange_labels(frame)
        if exchanges is None:
            validate_exchange_open_dates(checked_dates)
        else:
            for exchange, indices in frame.groupby(
                exchanges, sort=False
            ).indices.items():
                validate_exchange_open_dates(
                    checked_dates.take(indices),
                    str(exchange),
                    verified_session_ordinals=SESSION_COLUMN in frame,
                )

    if SESSION_COLUMN not in frame:
        return

    positions = frame[SESSION_COLUMN]
    if (
        positions.isna().any()
        or not pd.api.types.is_integer_dtype(positions.dtype)
        or pd.api.types.is_bool_dtype(positions.dtype)
        or (positions < 0).any()
    ):
        raise ValueError(
            "Session_Number must contain nonnegative integer market positions"
        )

    if "Date" in frame:
        exchange_labels = _normalized_exchange_labels(frame)
        if exchange_labels is not None:
            groups = (group for _, group in frame.groupby(exchange_labels, sort=False))
        elif "Name" in frame:
            groups = (
                group for _, group in frame.groupby("Name", sort=False, dropna=False)
            )
        else:
            groups = (frame,)
        for group in groups:
            calendar = group[["Date", SESSION_COLUMN]].drop_duplicates()
            if (
                calendar["Date"].duplicated().any()
                or calendar[SESSION_COLUMN].duplicated().any()
            ):
                raise ValueError(
                    "Session_Number must identify exactly one market date per ticker"
                )
            calendar = calendar.sort_values("Date")
            chronological = calendar[SESSION_COLUMN].tolist()
            if any(later <= earlier for earlier, later in pairwise(chronological)):
                raise ValueError(
                    "Session_Number must increase with the market dates per ticker"
                )
            exchange = (
                str(group["Exchange"].iloc[0]).strip().upper()
                if exchange_labels is not None
                else "NYSE"
            )
            if exchange in _KNOWN_EXCHANGES:
                # Declared ordinals can use any origin, but must preserve exact
                # spacing of every actual exchange session. This also applies
                # to unlabeled U.S. equities and to London/Toronto/Nasdaq.
                modern = calendar.loc[calendar["Date"] >= pd.Timestamp("1971-01-01")]
                if len(modern) > 1:
                    modern_dates = pd.DatetimeIndex(modern["Date"])
                    scheduled = inferred_equity_sessions(
                        modern_dates, exchange=exchange
                    )
                    true_ordinals = scheduled.get_indexer(modern_dates).tolist()
                    supplied = modern[SESSION_COLUMN].tolist()
                    if any(
                        int(later) - int(earlier) != real_later - real_earlier
                        for (earlier, later), (real_earlier, real_later) in zip(
                            pairwise(supplied), pairwise(true_ordinals), strict=True
                        )
                    ):
                        raise ValueError(
                            "Session_Number spacing must match scheduled "
                            f"{exchange} trading sessions"
                        )

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
                raise ValueError(
                    "Trading observations must occupy consecutive market sessions"
                )
    elif "Date" not in frame:
        groups = (
            (group for _, group in frame.groupby("Name", sort=False))
            if "Name" in frame
            else (frame,)
        )
        for group in groups:
            sequence = group[SESSION_COLUMN].tolist()
            if any(later <= earlier for earlier, later in pairwise(sequence)):
                raise ValueError(
                    "Session_Number must increase with the observation order"
                )
