"""Strict, reproducible calendar dates for daily market observations."""

from __future__ import annotations

import re
from datetime import date, datetime

import numpy as np
import pandas as pd


_ISO_DAY = re.compile(r"\d{4}-\d{1,2}-\d{1,2}\Z")


def parse_daily_dates(values: pd.Series, *, context: str = "Dates") -> pd.Series:
    """Accept ISO year-first dates or date objects, never timestamps or relative text.

    A daily bar represents a calendar day. Allowing an intraday time creates
    multiple observations for one session; allowing words such as ``today`` or
    numeric epoch values silently makes historical experiments nonreproducible.
    """
    message = f"{context} must be present, parseable ISO calendar dates without times or timezones"

    def is_explicit_day(value: object) -> bool:
        if isinstance(value, str):
            return bool(_ISO_DAY.fullmatch(value))
        if isinstance(value, (date, datetime, np.datetime64)):
            return getattr(value, "tzinfo", None) is None
        return False

    if values.isna().any() or not values.map(is_explicit_day).all():
        raise ValueError(message)
    try:
        parsed = pd.to_datetime(values, format="mixed", errors="coerce")
        if (parsed.isna().any() or not (parsed.dt.normalize() == parsed).all()
                or isinstance(parsed.dtype, pd.DatetimeTZDtype)):
            raise ValueError(message)
    except (TypeError, AttributeError, ValueError) as exc:
        raise ValueError(message) from exc
    return parsed
