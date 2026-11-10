"""Exchange-aware daily session positions for date-only U.S. equity inputs.

NYSE regular holidays and exceptional full-day closures are omitted from the
reference calendar. Observed weekdays are included as well, since a custom
source may describe another exchange with a different holiday schedule.
"""

from __future__ import annotations

from functools import lru_cache

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
        Holiday("Martin Luther King Jr. Day", month=1, day=1,
                offset=DateOffset(weekday=MO(3)), start_date="1998-01-01"),
        Holiday("Washington's Birthday", month=2, day=1,
                offset=DateOffset(weekday=MO(3))),
        GoodFriday,
        Holiday("Memorial Day", month=5, day=31,
                offset=DateOffset(weekday=MO(-1))),
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01",
                observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        Holiday("Labor Day", month=9, day=1,
                offset=DateOffset(weekday=MO(1))),
        Holiday("Thanksgiving", month=11, day=1,
                offset=DateOffset(weekday=TH(4))),
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


# Special NYSE market-wide closures beyond the regular recurring holidays.
_EXCEPTIONAL_CLOSURES = pd.DatetimeIndex([
    # Presidential Election Day was a full NYSE holiday in 1972, 1976, 1980.
    "1972-11-07", "1976-11-02", "1980-11-04",
    # Special full-day closures in the NYSE historical closing record.
    "1972-12-28", "1973-01-25", "1977-07-14", "1985-09-27",
    "1994-04-27",
    "2001-09-11", "2001-09-12", "2001-09-13", "2001-09-14",
    "2004-06-11", "2007-01-02", "2012-10-29", "2012-10-30",
    "2018-12-05", "2025-01-09",
])


@lru_cache(maxsize=32)
def _scheduled_sessions(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    recurring_holidays = _NYSEHolidays().holidays(start=start, end=end)
    exceptions = _EXCEPTIONAL_CLOSURES[
        (_EXCEPTIONAL_CLOSURES >= start) & (_EXCEPTIONAL_CLOSURES <= end)
    ]
    closure_days = recurring_holidays.union(exceptions)
    return pd.bdate_range(start, end, freq=CustomBusinessDay(holidays=closure_days))


def nyse_open_mask(dates: pd.DatetimeIndex) -> np.ndarray:
    """Identify known NYSE sessions; pre-1971 dates require external ordinals."""
    modern = dates >= pd.Timestamp("1971-01-01")
    # Pre-1971 holiday schedules still require verified ordinals, but Sunday
    # was never a session, and Saturday trading stopped after May 24, 1952.
    historical_saturday = ((dates.dayofweek == 5)
                           & (dates <= LAST_NYSE_SATURDAY_SESSION))
    valid = (~pd.isna(dates) & ((dates.dayofweek < 5) | historical_saturday))
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


def inferred_equity_sessions(dates: pd.DatetimeIndex, *,
                             start: pd.Timestamp | None = None,
                             end: pd.Timestamp | None = None,
                             nyse_only: bool = False) -> pd.DatetimeIndex:
    """Include scheduled NYSE sessions plus dates explicitly observed by a caller.

    Thus an absent genuine holiday does not create a fictional gap, while an
    omitted ordinary market weekday still creates one. Observed weekdays stay
    eligible for non-NYSE sources unless the caller supplies exact ordinals.
    """
    if dates.empty:
        return pd.DatetimeIndex([])
    first = dates.min() if start is None else start
    if first < pd.Timestamp("1971-01-01"):
        # Earlier NYSE holidays and special schedules were materially different
        # (including the 1968 Wednesday closures). Inferring a modern calendar
        # would silently invent sessions. Historical callers must provide
        # independently verified market-session ordinals instead.
        raise ValueError("NYSE dates before 1971 require verified Session_Number metadata")
    if nyse_only:
        validate_nyse_open_dates(dates)
    scheduled = _scheduled_sessions(
        first,
        dates.max() if end is None else end,
    )
    return scheduled.union(dates.unique()).sort_values()
