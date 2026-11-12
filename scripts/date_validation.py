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


# A validated DataFrame carries its source-date attestation into derived copies.
# This marker is also an explicit caller assertion, never proof by itself.
VERIFIED_DAILY_DATES_ATTR = "stocktrading_verified_daily_dates"
VERIFIED_TARGET_PAIRS_ATTR = "stocktrading_verified_target_pairs"


def _date_pairs(frame: pd.DataFrame) -> frozenset[tuple[int, int]]:
    """Capture verified date-to-target identities, surviving ordinary row subsets."""
    if "Date" not in frame or "Target_Date" not in frame:
        return frozenset()
    dates = parse_daily_dates(frame["Date"], context="Feature dates").astype("int64")
    targets = parse_daily_dates(frame["Target_Date"], context="Target dates").astype(
        "int64"
    )
    return frozenset(zip(dates, targets, strict=True))


def require_verified_daily_date_source(
    frame: pd.DataFrame,
    *,
    verified_datetime_dates: bool = False,
    column: str = "Date",
) -> None:
    """Reject previously coerced timestamps without a source-precision attestation.

    DataFrame construction erases whether NumPy dates originally identified a
    month or an actual day. Public APIs must not silently interpret the first
    day of a month as a daily observation. Validated frames retain their
    attestation in DataFrame.attrs; independent callers may explicitly attest.
    """
    if not isinstance(verified_datetime_dates, (bool, np.bool_)):
        raise ValueError("verified_datetime_dates must be a boolean")
    if column not in frame:
        return
    supplied = frame[column]
    ambiguous = (
        pd.api.types.is_datetime64_any_dtype(supplied.dtype)
        or supplied.map(lambda value: isinstance(value, pd.Timestamp)).any()
    )
    attested = (
        verified_datetime_dates or frame.attrs.get(VERIFIED_DAILY_DATES_ATTR) is True
    )
    if ambiguous and not attested:
        raise ValueError(
            "Pandas datetime dates have unknown original precision; use original "
            "ISO daily dates or explicitly set verified_datetime_dates=True "
            "after verifying their source precision"
        )
    if (
        column == "Target_Date"
        and ambiguous
        and not verified_datetime_dates
        and frame.attrs.get(VERIFIED_TARGET_PAIRS_ATTR) is not None
        and not _date_pairs(frame).issubset(frame.attrs[VERIFIED_TARGET_PAIRS_ATTR])
    ):
        raise ValueError(
            "Target_Date source verification is stale: target dates changed "
            "after their verified feature/target pairs were recorded"
        )


def mark_verified_daily_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """Mark a frame whose source dates have already passed daily verification."""
    frame.attrs[VERIFIED_DAILY_DATES_ATTR] = True
    if "Date" in frame and "Target_Date" in frame:
        frame.attrs[VERIFIED_TARGET_PAIRS_ATTR] = _date_pairs(frame)
    return frame
