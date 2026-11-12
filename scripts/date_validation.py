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
_VERIFIED_ROWS_ATTR = "stocktrading_verified_date_rows"


def _verified_row_tokens(frame: pd.DataFrame, column: str) -> frozenset[int]:
    """Bind date attestation to row identities, surviving filtering and reordering.

    An attestation for Date must not silently authorize a subsequently replaced
    Target_Date, even when pandas preserves DataFrame.attrs.
    """
    columns = []
    if "Name" in frame:
        columns.append("Name")
    if column == "Target_Date" and "Date" in frame:
        columns.append("Date")
    columns.append(column)
    identifying = frame[columns].copy()
    for date_column in ("Date", "Target_Date"):
        if date_column in identifying:
            identifying[date_column] = parse_daily_dates(
                identifying[date_column], context="Attested dates"
            )
    return frozenset(
        int(value)
        for value in pd.util.hash_pandas_object(identifying, index=False).to_numpy()
    )


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
    if not ambiguous or verified_datetime_dates:
        return
    attested = frame.attrs.get(_VERIFIED_ROWS_ATTR)
    if (
        frame.attrs.get(VERIFIED_DAILY_DATES_ATTR) is True
        and isinstance(attested, dict)
        and column in attested
        and _verified_row_tokens(frame, column).issubset(attested[column])
    ):
        return
    raise ValueError(
        "Pandas datetime dates have unknown original precision or changed since "
        "attestation; use original ISO daily dates or explicitly set "
        "verified_datetime_dates=True after verifying their source precision"
    )


def mark_verified_daily_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """Attest the current date values, including separate target-date identity.

    Immutable row tokens permit validated subsets, sorting, and index resets
    without trusting date columns that were replaced after validation.
    """
    attested = {
        column: _verified_row_tokens(frame, column)
        for column in ("Date", "Target_Date")
        if column in frame
    }
    frame.attrs[VERIFIED_DAILY_DATES_ATTR] = True
    frame.attrs[_VERIFIED_ROWS_ATTR] = attested
    return frame
