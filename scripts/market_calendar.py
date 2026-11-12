"""Exchange-aware daily market-session validation and inference.

U.S., London, and Toronto equity dates use their respective market calendars.
Unknown exchanges must provide independently verified session numbers rather
than inheriting U.S. holidays or inventing additional trading sessions.
"""

from __future__ import annotations

from functools import lru_cache

import exchange_calendars
import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    MO,
    TH,
    nearest_workday,
    sunday_to_monday,
)
from pandas.tseries.offsets import CustomBusinessDay, DateOffset


LAST_NYSE_SATURDAY_SESSION = pd.Timestamp("1952-05-24")


class _NYSEHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        Holiday(
            "Martin Luther King Jr. Day",
            month=1,
            day=1,
            offset=DateOffset(weekday=MO(3)),
            start_date="1998-01-01",
        ),
        Holiday(
            "Washington's Birthday", month=2, day=1, offset=DateOffset(weekday=MO(3))
        ),
        GoodFriday,
        Holiday("Memorial Day", month=5, day=31, offset=DateOffset(weekday=MO(-1))),
        Holiday(
            "Juneteenth",
            month=6,
            day=19,
            start_date="2022-01-01",
            observance=nearest_workday,
        ),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        Holiday("Labor Day", month=9, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("Thanksgiving", month=11, day=1, offset=DateOffset(weekday=TH(4))),
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


# Special NYSE market-wide closures beyond the regular recurring holidays.
_EXCEPTIONAL_CLOSURES = pd.DatetimeIndex(
    [
        # Presidential Election Day was a full NYSE holiday in 1972, 1976, 1980.
        "1972-11-07",
        "1976-11-02",
        "1980-11-04",
        # Special full-day closures in the NYSE historical closing record.
        "1972-12-28",
        "1973-01-25",
        "1977-07-14",
        "1985-09-27",
        "1994-04-27",
        "2001-09-11",
        "2001-09-12",
        "2001-09-13",
        "2001-09-14",
        "2004-06-11",
        "2007-01-02",
        "2012-10-29",
        "2012-10-30",
        "2018-12-05",
        "2025-01-09",
    ]
)


@lru_cache(maxsize=32)
def _scheduled_sessions(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    recurring_holidays = _NYSEHolidays().holidays(start=start, end=end)
    exceptions = _EXCEPTIONAL_CLOSURES[
        (_EXCEPTIONAL_CLOSURES >= start) & (_EXCEPTIONAL_CLOSURES <= end)
    ]
    closure_days = recurring_holidays.union(exceptions)
    return pd.bdate_range(start, end, freq=CustomBusinessDay(holidays=closure_days))


def _valid_daily_mask(dates: pd.DatetimeIndex) -> np.ndarray:
    """Daily sessions are timezone-naive midnight dates, never intraday bars."""
    if dates.tz is not None:
        return np.zeros(len(dates), dtype=bool)
    return ~pd.isna(dates) & (dates == dates.normalize())


def nyse_open_mask(dates: pd.DatetimeIndex) -> np.ndarray:
    """Identify known NYSE sessions; pre-1971 dates require external ordinals."""
    daily = _valid_daily_mask(dates)
    if not daily.all():
        result = np.zeros(len(dates), dtype=bool)
        if daily.any():
            result[daily] = nyse_open_mask(dates[daily])
        return result
    modern = dates >= pd.Timestamp("1971-01-01")
    # Pre-1971 holiday schedules still require verified ordinals, but Sunday
    # was never a session, and Saturday trading stopped after May 24, 1952.
    historical_saturday = (dates.dayofweek == 5) & (dates <= LAST_NYSE_SATURDAY_SESSION)
    valid = ~pd.isna(dates) & ((dates.dayofweek < 5) | historical_saturday)
    if modern.any():
        observed = dates[modern]
        scheduled = _scheduled_sessions(observed.min(), observed.max())
        valid[modern] = observed.isin(scheduled)
    return valid


def validate_nyse_open_dates(dates: pd.DatetimeIndex) -> None:
    """Reject recorded NYSE trades on known closed dates from 1971 onward.

    Earlier verified sessions remain caller-supplied because the modern calendar
    deliberately cannot infer the NYSE's pre-1971 historical schedule.
    """
    impossible = dates[~nyse_open_mask(dates)]
    if not impossible.empty:
        raise ValueError(
            f"NYSE observations contain a closed-market date: {impossible[0].date()}"
        )


# Explicit aliases for exchanges whose actual scheduled sessions are known.
_US_EQUITY_EXCHANGES = frozenset({"NYSE", "XNYS", "NASDAQ", "XNAS"})
_LONDON_EXCHANGES = frozenset({"LSE", "XLON"})
_TORONTO_EXCHANGES = frozenset({"TSX", "XTSE"})
_KNOWN_EXCHANGES = _US_EQUITY_EXCHANGES | _LONDON_EXCHANGES | _TORONTO_EXCHANGES


@lru_cache(maxsize=64)
def _exchange_sessions(
    start: pd.Timestamp, end: pd.Timestamp, exchange: str
) -> pd.DatetimeIndex:
    """Official dated sessions for the London and Toronto stock exchanges."""
    code = "XLON" if exchange in _LONDON_EXCHANGES else "XTSE"
    # Explicit bounds prevent results from changing with today's date.
    calendar = exchange_calendars.get_calendar(
        code, start=start - pd.Timedelta(days=14), end=end + pd.Timedelta(days=14)
    )
    return pd.DatetimeIndex(calendar.sessions_in_range(start, end))


def _london_sessions(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Backward-compatible London calendar accessor."""
    return _exchange_sessions(start, end, "XLON")


def exchange_open_mask(
    dates: pd.DatetimeIndex,
    exchange: str | None = None,
    *,
    verified_session_ordinals: bool = False,
) -> np.ndarray:
    """Check session validity for known exchanges; unknown exchanges need ordinals.

    When Exchange is absent, the documented default is U.S. equities, not an
    arbitrary international exchange. Historical pre-1971 sessions with
    supplied ordinals retain the project’s existing partial calendar policy.
    """
    label = "NYSE" if exchange is None else exchange.strip().upper()
    daily = _valid_daily_mask(dates)
    if not daily.all():
        result = np.zeros(len(dates), dtype=bool)
        if daily.any():
            result[daily] = exchange_open_mask(
                dates[daily],
                exchange=exchange,
                verified_session_ordinals=verified_session_ordinals,
            )
        return result
    if label in _US_EQUITY_EXCHANGES:
        return nyse_open_mask(dates)
    if label in (_LONDON_EXCHANGES | _TORONTO_EXCHANGES):
        modern = dates >= pd.Timestamp("1971-01-01")
        if label in _LONDON_EXCHANGES:
            # London did not trade Saturdays in 1917-1931; Saturday
            # sessions returned 1931-09-19 and ended after 1934-05-26.
            # Earlier historical weekends need independently verified ordinals.
            historic_saturday = (
                (dates.dayofweek == 5)
                & verified_session_ordinals
                & (
                    (dates < pd.Timestamp("1917-01-01"))
                    | (
                        (dates >= pd.Timestamp("1931-09-19"))
                        & (dates <= pd.Timestamp("1934-05-26"))
                    )
                )
            )
        else:
            # The historical TSX schedule is not authoritatively inferred.
            # A caller-supplied, verified ordinal may attest old Saturdays.
            historic_saturday = (
                (dates.dayofweek == 5) & ~modern & verified_session_ordinals
            )
        valid = ~pd.isna(dates) & ((dates.dayofweek < 5) | historic_saturday)
        if modern.any():
            observed = dates[modern]
            sessions = _exchange_sessions(observed.min(), observed.max(), label)
            valid[modern] = observed.isin(sessions)
        return valid
    # Custom-market weekends require caller-verified ordinal metadata.
    if verified_session_ordinals:
        return ~pd.isna(dates)
    return ~pd.isna(dates) & (dates.dayofweek < 5)


def validate_exchange_open_dates(
    dates: pd.DatetimeIndex,
    exchange: str | None = None,
    *,
    verified_session_ordinals: bool = False,
) -> None:
    """Reject closed sessions; custom weekends require verified ordinals."""
    impossible = dates[
        ~exchange_open_mask(
            dates, exchange, verified_session_ordinals=verified_session_ordinals
        )
    ]
    if not impossible.empty:
        label = exchange or "NYSE (default U.S. equity calendar)"
        raise ValueError(
            f"{label} observations contain a closed-market date: {impossible[0].date()}"
        )


def _calendar_bounds(
    dates: pd.DatetimeIndex,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Ensure a requested session calendar covers every observed trading date."""
    try:
        first = dates.min() if start is None else pd.Timestamp(start)
        last = dates.max() if end is None else pd.Timestamp(end)
        invalid = (
            pd.isna(first)
            or pd.isna(last)
            or first > last
            or first > dates.min()
            or last < dates.max()
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Calendar bounds must be valid and include every observed trading date"
        ) from exc
    if invalid:
        raise ValueError(
            "Calendar bounds must be valid and include every observed trading date"
        )
    return first, last


def inferred_equity_sessions(
    dates: pd.DatetimeIndex,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    nyse_only: bool = False,
    exchange: str | None = None,
) -> pd.DatetimeIndex:
    """Infer authoritative sessions for known exchanges.

    Unlabeled equities use the U.S. market schedule. London and Toronto labels
    use their respective schedules. Unknown exchanges require independently
    verified Session_Number metadata instead of inferred trading days.
    Calendar bounds must include all observed dates.
    """
    if dates.empty:
        return pd.DatetimeIndex([])
    if pd.isna(dates).any() or (dates.dayofweek >= 5).any():
        raise ValueError("Trading dates must be nonmissing weekday market sessions")
    first, last = _calendar_bounds(dates, start, end)
    if first < pd.Timestamp("1971-01-01"):
        raise ValueError(
            "NYSE dates before 1971 require verified Session_Number metadata"
        )
    label = "NYSE" if exchange is None or nyse_only else exchange.strip().upper()
    if label in (_LONDON_EXCHANGES | _TORONTO_EXCHANGES):
        scheduled = _exchange_sessions(first, last, label)
        validate_exchange_open_dates(dates, label)
        return scheduled
    if label in _US_EQUITY_EXCHANGES:
        validate_exchange_open_dates(dates, label)
        return _scheduled_sessions(first, last)
    raise ValueError(
        f"Unknown exchange {label!r} requires verified Session_Number metadata; "
        "refusing to invent sessions from another market's calendar"
    )
