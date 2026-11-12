"""Load historical equity data and create causal, per-ticker model features.

This stage never fits global scalers, quantile bins, or an HMM. Those learned
transformations belong inside training folds, after the chronological split.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .technical_indicators import add_technical_indicators
    from .date_validation import (
        mark_verified_daily_dates,
        parse_daily_dates,
        require_verified_daily_date_source,
    )
    from .market_calendar import exchange_open_mask
    from .price_validation import valid_price_bars
    from .session_validation import (
        adjacent_session_flags,
        session_positions,
        validate_session_positions,
    )
    from .ticker_validation import MISSING_TICKER_MARKERS as MISSING_TICKER_MARKERS
    from .ticker_validation import normalize_ticker_symbols, valid_ticker_symbols
else:
    from technical_indicators import add_technical_indicators
    from date_validation import (
        mark_verified_daily_dates,
        parse_daily_dates,
        require_verified_daily_date_source,
    )
    from market_calendar import exchange_open_mask
    from price_validation import valid_price_bars
    from session_validation import (
        adjacent_session_flags,
        session_positions,
        validate_session_positions,
    )
    from ticker_validation import MISSING_TICKER_MARKERS as MISSING_TICKER_MARKERS
    from ticker_validation import normalize_ticker_symbols, valid_ticker_symbols


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DATA = PROJECT_ROOT / "data/raw/all_stocks_2006-01-01_to_2018-01-01.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/generated/stock_features.csv"
REQUIRED_COLUMNS = ("Date", "Name", "Open", "High", "Low", "Close", "Volume")
MODEL_FEATURES = ("Return", "Intraday_Return", "Daily_Range", "MACD_Pct")
TARGET_DEFINITIONS = {
    "close_to_close": "Next session close is above the decision-session close",
    "next_open_to_close": "Next session close is above its opening price",
}
SESSION_COLUMN = "Session_Number"
_BUNDLED_SYMBOLS = (
    "AABA",
    "AAPL",
    "AMZN",
    "AXP",
    "BA",
    "CAT",
    "CSCO",
    "CVX",
    "DIS",
    "GE",
    "GOOGL",
    "GS",
    "HD",
    "IBM",
    "INTC",
    "JNJ",
    "JPM",
    "KO",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "PFE",
    "PG",
    "TRV",
    "UNH",
    "UTX",
    "VZ",
    "WMT",
    "XOM",
)
# Nasdaq-listed members of the fixed third-party sample.
# Unknown symbols retain default NYSE validation.
_BUNDLED_NASDAQ_SYMBOLS = frozenset(
    {"AABA", "AAPL", "AMZN", "CSCO", "GOOGL", "INTC", "MSFT"}
)


_BUNDLED_RAW_NAMES = frozenset(
    {
        DEFAULT_RAW_DATA.name,
        "all_stocks_2017-01-01_to_2018-01-01.csv",
        *(f"{ticker}_2006-01-01_to_2018-01-01.csv" for ticker in _BUNDLED_SYMBOLS),
    }
)


@lru_cache(maxsize=4)
def _cached_bundled_session_calendar(
    path: Path, contents_hash: bytes
) -> pd.DatetimeIndex:
    """Use source-observed market dates with valid ticker identities as authority.

    Individual CSVs contain ticker-specific holes, so their observed dates do
    not identify all trading sessions represented by the bundled dataset. A
    damaged price bar still identifies a potential session, whereas malformed
    dates and invalid ticker identifiers cannot be trusted for this purpose.
    """
    source = pd.read_csv(path, keep_default_na=False)
    usable_identifiers = valid_ticker_symbols(source["Name"])
    raw_dates = source.loc[usable_identifiers, "Date"].astype("string")
    explicit = raw_dates.str.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}").fillna(False)
    dates = pd.to_datetime(raw_dates.where(explicit), format="mixed", errors="coerce")
    dates = dates.loc[dates.notna() & dates.dt.dayofweek.lt(5)]
    names = source.loc[dates.index, "Name"].astype("string").str.strip().str.upper()
    observed = pd.DatetimeIndex(dates)
    # The bundled universe spans two exchanges. Verify the source ticker's
    # own market before letting its date contribute a reference session.
    nasdaq = names.isin(_BUNDLED_NASDAQ_SYMBOLS).to_numpy()
    valid = (~nasdaq & exchange_open_mask(observed, "NYSE")) | (
        nasdaq & exchange_open_mask(observed, "NASDAQ")
    )
    return pd.DatetimeIndex(dates.loc[valid].unique()).sort_values()


def _bundled_session_calendar() -> pd.DatetimeIndex:
    # Key the cache to the bytes, not the filename or modification timestamp:
    # replacing the bundled source in a long-running interpreter cannot reuse
    # an obsolete calendar, even if its size and mtime are unchanged.
    return _cached_bundled_session_calendar(
        DEFAULT_RAW_DATA.resolve(),
        hashlib.sha256(DEFAULT_RAW_DATA.read_bytes()).digest(),
    )


def _is_bundled_raw_csv(filepath: str | Path) -> bool:
    if not isinstance(filepath, (str, Path)):
        return False
    path = Path(filepath).resolve()
    return (
        path.is_file()
        and path.parent == DEFAULT_RAW_DATA.parent.resolve()
        and path.name in _BUNDLED_RAW_NAMES
    )


def _custom_source_session_positions(
    source: pd.DataFrame, selected_mask: pd.Series
) -> pd.Series:
    """Infer selected sessions from the full source's valid market-date context.

    Malformed records for unrelated symbols must not invalidate a selected
    ticker. An exchange label, when supplied, allows other symbols on that
    exchange to establish a missing market session.
    """
    raw_dates = source["Date"].astype("string")
    explicit = raw_dates.str.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}").fillna(False)
    parsed = pd.to_datetime(raw_dates.where(explicit), format="mixed", errors="coerce")
    valid = (
        valid_ticker_symbols(source["Name"])
        & parsed.notna()
        & parsed.dt.dayofweek.lt(5)
    )
    reference = source.loc[valid].copy()
    if reference.empty:
        raise ValueError("No usable market dates remain for session inference")
    reference["Date"] = parsed.loc[valid]
    # Date attestation hashes the canonical ticker identity. Normalize Name
    # before recording that attestation, otherwise " aapl " and "AAPL" can
    # disagree with their own verified row identity during selected loads.
    reference["Name"] = reference["Name"].astype(str).str.strip().str.upper()
    mark_verified_daily_dates(reference, columns=("Date",))
    # An unrelated pre-1971 record must not make a selected modern ticker
    # depend on an historical calendar that requires verified metadata.
    if parsed.loc[selected_mask].min() >= pd.Timestamp("1971-01-01"):
        reference = reference.loc[reference["Date"] >= pd.Timestamp("1971-01-01")]
    start, end = reference["Date"].min(), reference["Date"].max()
    if "Exchange" in source:
        wanted = (
            source.loc[selected_mask, "Exchange"]
            .astype("string")
            .str.strip()
            .str.upper()
            .unique()
        )
        labels = reference["Exchange"].astype("string").str.strip().str.upper()
        reference = reference.loc[labels.isin(wanted)]
        # Invalid peer observations cannot establish trading sessions for a
        # selected ticker. Check each declared exchange independently, leaving
        # the selected ticker's own invalid records intact for strict rejection.
        labels = labels.loc[reference.index]
        peer_indices = reference.index[
            ~reference.index.isin(source.index[selected_mask])
        ]
        invalid_peers = []
        for exchange in labels.loc[peer_indices].unique():
            indices = peer_indices[labels.loc[peer_indices].eq(exchange)]
            dates = pd.DatetimeIndex(reference.loc[indices, "Date"])
            invalid_peers.extend(indices[~exchange_open_mask(dates, str(exchange))])
        reference = reference.drop(index=invalid_peers)
        # Only the requested symbol's records are part of the strict input
        # contract. Duplicate observations from unrelated peer symbols do not
        # create additional market sessions and must not invalidate a valid
        # ticker selection. Preserve duplicates for the requested ticker so
        # validate_bars() can still reject its corrupted source.
        unrelated = ~reference.index.isin(source.index[selected_mask])
        duplicate_peers = unrelated & reference.duplicated(
            ["Name", "Date"], keep="first"
        )
        reference = reference.loc[~duplicate_peers]
    else:
        reference = reference.loc[selected_mask]
    inferred = session_positions(reference, calendar_start=start, calendar_end=end)
    selected = inferred.reindex(source.index[selected_mask])
    if selected.isna().any():
        raise ValueError("Selected observations must have valid weekday market dates")
    return selected.astype("int64")


def validate_bars(
    df: pd.DataFrame, *, verified_datetime_dates: bool = False
) -> pd.DataFrame:
    """Retain complete, coherent daily price bars with verified dates and symbols.

    A contradictory OHLC record cannot be repaired reliably without an
    independent price source, so it is excluded from reproducible experiments.
    Pandas datetime64/Timestamp values have irretrievably lost their original
    precision; callers must explicitly attest to having verified source dates.
    """
    if not isinstance(verified_datetime_dates, (bool, np.bool_)):
        raise ValueError("verified_datetime_dates must be a boolean")
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    supplied_dates = df["Date"]
    ambiguous_precision = (
        pd.api.types.is_datetime64_any_dtype(supplied_dates.dtype)
        or supplied_dates.map(lambda value: isinstance(value, pd.Timestamp)).any()
    )
    if ambiguous_precision and not verified_datetime_dates:
        raise ValueError(
            "Pandas datetime dates have unknown original precision; supply explicit "
            "ISO daily dates, use validate_raw_bar_records on raw NumPy dates, or "
            "set verified_datetime_dates=True after checking the source precision"
        )
    # If a Target_Date survives input processing, its provenance must already
    # be valid before normalizing Date/Name changes the row-identity hashes.
    # Only then may those unchanged target values be rebound to canonical
    # row identities. This refuses tampered targets instead of laundering them.
    attested_columns = ("Date",)
    if "Target_Date" in df:
        require_verified_daily_date_source(df, column="Target_Date")
        attested_columns = ("Date", "Target_Date")
    df = df.copy()
    df["Date"] = parse_daily_dates(df["Date"], context="Every Date")
    mark_verified_daily_dates(df, columns=attested_columns)
    df["Name"] = normalize_ticker_symbols(
        df["Name"],
        context="Every observation must have a ticker symbol",
    )
    mark_verified_daily_dates(df, columns=attested_columns)
    if df.duplicated(["Name", "Date"]).any():
        raise ValueError("Duplicate ticker/date observations")
    input_tickers = set(df["Name"])
    # Preserve the original market-session positions *before* excluding any
    # corrupt bar. A subsequent valid price must not become the missing bar's
    # fictional next trading session during return/HMM calculations.
    if SESSION_COLUMN not in df:
        df[SESSION_COLUMN] = session_positions(df)
    validate_session_positions(df)
    valid_prices = valid_price_bars(df, require_volume=True)
    numeric_columns = ("Open", "High", "Low", "Close", "Volume")
    for name in numeric_columns:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    df = df.loc[valid_prices].copy()
    if df.empty:
        raise ValueError("No valid OHLCV observations remain")
    missing_tickers = sorted(input_tickers - set(df["Name"]))
    if missing_tickers:
        raise ValueError(
            "No valid OHLCV observations remain for ticker(s) "
            f"{', '.join(missing_tickers)}"
        )
    return mark_verified_daily_dates(
        df.sort_values(["Name", "Date"]).reset_index(drop=True),
        columns=attested_columns,
    )


def validate_raw_bar_records(records: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    """Validate raw records *before* pandas can erase date precision.

    A normal ``pd.DataFrame`` constructor coerces an array of month-only
    ``np.datetime64`` values into complete timestamps. The resulting values
    cannot be distinguished from genuinely day-precise timestamps. This entry
    point preserves each original date scalar until precision is verified.
    """
    rows = list(records)
    if not rows:
        raise ValueError("A nonempty collection of raw price records is required")
    if any(not isinstance(row, Mapping) or "Date" not in row for row in rows):
        raise ValueError("Every raw price record must provide a Date")
    raw_dates = [row["Date"] for row in rows]
    # A Timestamp has already discarded its original source precision. Raw
    # records must retain genuine date-only scalars; an independently verified
    # pandas timestamp belongs in validate_bars(..., verified_datetime_dates=True).
    if any(isinstance(value, pd.Timestamp) for value in raw_dates):
        raise ValueError(
            "Every Date must be an original date-only scalar, not a pandas Timestamp"
        )
    dates = parse_daily_dates(pd.Series(raw_dates, dtype=object), context="Every Date")
    frame = pd.DataFrame.from_records(rows)
    frame["Date"] = dates.to_numpy()
    return validate_bars(frame, verified_datetime_dates=True)


def load_data(
    filepath: str | Path, *, tickers: tuple[str, ...] | None = None
) -> pd.DataFrame:
    """Read prices, preserving the source-wide calendar when selecting symbols."""
    # Stock symbols such as "NA" are valid identifiers. Pandas' default NA
    # tokens would silently turn them into missing ticker names.
    source = pd.read_csv(filepath, keep_default_na=False)
    bundled = _is_bundled_raw_csv(filepath)
    if tickers is not None:
        if not tickers or any(
            not isinstance(ticker, str) or not ticker.strip() for ticker in tickers
        ):
            raise ValueError("Selected tickers must be nonempty strings")
        if "Name" not in source:
            raise ValueError("Missing required columns: ['Name']")
        selected = {ticker.strip().upper() for ticker in tickers}
        normalized_names = source["Name"].astype("string").str.strip().str.upper()
        not_found = selected - set(normalized_names.dropna())
        if not_found:
            raise ValueError(f"Ticker not found: {', '.join(sorted(not_found))}")
        selected_mask = normalized_names.isin(selected)
        if SESSION_COLUMN not in source and not bundled and "Date" in source:
            # Selected dates are validated first; malformed unrelated source
            # dates are ignored when establishing the reference calendar.
            parse_daily_dates(source.loc[selected_mask, "Date"], context="Every Date")
            selected_positions = _custom_source_session_positions(source, selected_mask)
            source = source.loc[selected_mask].copy()
            source[SESSION_COLUMN] = selected_positions.to_numpy(dtype="int64")
        else:
            source = source.loc[selected_mask].copy()
    if SESSION_COLUMN not in source and bundled:
        # Both the complete source and every bundled per-symbol CSV must use
        # exactly the same reference calendar, even on dates absent for a symbol.
        dates = parse_daily_dates(source["Date"], context="Every Date")
        calendar = _bundled_session_calendar()
        if not exchange_open_mask(calendar, "NYSE").all():
            # In older bundled sources, Nasdaq-only sessions require
            # exchange-specific ordinals instead of a fictional NYSE gap.
            # The published 2006-2017 data retain their original shared
            # source-observed ordinal positions and output schema.
            symbols = source["Name"].astype("string").str.strip().str.upper()
            source["Exchange"] = np.where(
                symbols.isin(_BUNDLED_NASDAQ_SYMBOLS), "NASDAQ", "NYSE"
            )
            source[SESSION_COLUMN] = session_positions(source)
        else:
            positions = calendar.get_indexer(dates)
            if (positions < 0).any():
                raise ValueError("Bundled observations contain non-market dates")
            source[SESSION_COLUMN] = positions
    return validate_bars(source)


def prepare_features(
    df: pd.DataFrame,
    *,
    features: tuple[str, ...] = MODEL_FEATURES,
    target: str = "close_to_close",
    verified_datetime_dates: bool = False,
) -> pd.DataFrame:
    """Produce causal next-session targets with only requested model features required.

    ``next_open_to_close`` aligns the target with an order submitted after the
    current close and filled at the next open. The historical close-to-close
    label remains the default for backward-compatible published experiments.
    Source-verified datetime DataFrames, including results from ``load_data``,
    require explicit ``verified_datetime_dates=True`` at this public boundary.
    """
    if target not in TARGET_DEFINITIONS:
        raise ValueError(f"target must be one of {sorted(TARGET_DEFINITIONS)}")
    source = prepare_indicator_history(
        df,
        verified_datetime_dates=verified_datetime_dates,
    )
    next_close = source.groupby("Name", sort=False)["Close"].shift(-1)
    next_open = source.groupby("Name", sort=False)["Open"].shift(-1)
    adjacent_target = source.groupby("Name", sort=False)[SESSION_COLUMN].transform(
        lambda positions: adjacent_session_flags(positions, forward=True)
    )
    source["Target_Date"] = (
        source.groupby("Name", sort=False)["Date"].shift(-1).where(adjacent_target)
    )
    # Shift object-precision ordinals before dropping targets; a float64
    # intermediate would corrupt exact uint64 ordinals above 2**53.
    source["Target_Session"] = (
        source.groupby("Name", sort=False)[SESSION_COLUMN]
        .transform(lambda positions: positions.astype(object).shift(-1))
        .where(adjacent_target)
    )
    reference_price = source["Close"] if target == "close_to_close" else next_open
    source["Trend"] = (next_close > reference_price).astype("int64")
    if not features or any(feature not in MODEL_FEATURES for feature in features):
        raise ValueError(f"Model features must be drawn from {MODEL_FEATURES}")
    required = list(features) + ["Target_Date"]
    # Only model features need nonfinite replacement. Applying DataFrame.replace
    # to object-precision target ordinals can silently downcast them and emits
    # a pandas FutureWarning that changes published notebook output.
    source[list(features)] = source[list(features)].replace([np.inf, -np.inf], np.nan)
    prepared = source.dropna(subset=required).reset_index(drop=True)
    # The target ordinal is exact source metadata, not a float-derived label.
    # Keep the integer dtype (including uint64 source positions) end-to-end.
    if prepared["Target_Session"].isna().any():
        raise ValueError("Prepared target sessions must be present")
    prepared["Target_Session"] = pd.Series(
        prepared["Target_Session"].tolist(),
        index=prepared.index,
        dtype=source[SESSION_COLUMN].dtype,
    )
    if prepared.empty:
        raise ValueError(
            "No usable feature observations remain after indicator warm-up; "
            "provide a longer price history"
        )
    missing_tickers = sorted(set(source["Name"]) - set(prepared["Name"]))
    if missing_tickers:
        raise ValueError(
            "No usable feature observations remain for ticker(s) "
            f"{', '.join(missing_tickers)}; provide longer price histories"
        )
    return mark_verified_daily_dates(prepared)


def prepare_indicator_history(
    df: pd.DataFrame, *, verified_datetime_dates: bool = False
) -> pd.DataFrame:
    """Retain validated, causally calculated price history, including its last bar.

    Unlike supervised feature preparation, this does not remove rows solely
    because their next-session prediction target is unavailable. It also retains
    missing-indicator rows to preserve HMM sequence boundaries. The caller must
    explicitly attest to the provenance of already-coerced pandas dates.
    """
    return add_technical_indicators(
        validate_bars(df, verified_datetime_dates=verified_datetime_dates)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare causal stock features")
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ticker", type=str, default=None)
    args = parser.parse_args()
    data = load_data(
        args.input, tickers=(args.ticker,) if args.ticker is not None else None
    )
    result = prepare_features(data, verified_datetime_dates=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    count = result.Name.nunique()
    print(
        f"Prepared {len(result):,} observations from {count} {'ticker' if count == 1 else 'tickers'}: {args.output}"
    )


if __name__ == "__main__":
    main()
