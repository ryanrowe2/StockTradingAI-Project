"""Strict, reproducible calendar dates for daily market observations."""

from __future__ import annotations

import re
from datetime import date, datetime

import numpy as np
import pandas as pd


_ISO_DAY = re.compile(r"\d{4}-\d{1,2}-\d{1,2}\Z")


def parse_daily_dates(values: pd.Series, *, context: str = "Dates") -> pd.Series:
    """Accept ISO dates, date objects, and previously normalized pandas dates.

    A daily bar represents a calendar day. Allowing an intraday time creates
    multiple observations for one session; allowing words such as ``today`` or
    numeric epoch values silently makes historical experiments nonreproducible.
    For untrusted NumPy inputs, require day-precision scalars before pandas
    DataFrame construction: converted timestamps no longer retain precision.
    Pandas Timestamps support internal, already-validated date columns; public
    ingestion requires explicit source verification before accepting them.
    """
    message = f"{context} must be present, parseable ISO calendar dates without times or timezones"

    def is_explicit_day(value: object) -> bool:
        if isinstance(value, str):
            return bool(_ISO_DAY.fullmatch(value))
        if isinstance(value, np.datetime64):
            # Both coarse units and midnight intraday timestamps violate the
            # raw daily-date contract, despite normalizing to a midnight day.
            return np.datetime_data(value.dtype) == ("D", 1)
        if isinstance(value, pd.Timestamp):
            return value.tzinfo is None
        if isinstance(value, datetime):
            return False
        if isinstance(value, date):
            return True
        return False

    if values.isna().any() or not values.map(is_explicit_day).all():
        raise ValueError(message)
    try:
        parsed = pd.to_datetime(values, format="mixed", errors="coerce")
        if (
            parsed.isna().any()
            or not (parsed.dt.normalize() == parsed).all()
            or isinstance(parsed.dtype, pd.DatetimeTZDtype)
        ):
            raise ValueError(message)
    except (TypeError, AttributeError, ValueError) as exc:
        raise ValueError(message) from exc
    return parsed
