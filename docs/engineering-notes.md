# Advanced engineering implementation notes

This detailed engineering analysis was preserved verbatim from the repository's original top-level README to keep the first-pass overview concise. Return to the [project overview](../README.md#detailed-architecture-and-implementation-notes) or [experiments guide](README_experiments.md).

## Detailed Architecture and Implementation Notes

Each ticker's indicators and next-session labels are computed independently. Ticker
identifiers are normalized consistently, including for direct indicator calls;
literal symbols such as `NA` are preserved during CSV loading. Date boundaries
are shared across the Bayesian and AAPL trading experiments, so their partitions
correspond to the same historical market periods. All bundled ticker CSVs use
the observed-session calendar of the complete 31-stock source in the
published 2006–2017 period. The fixed sample contains both NYSE- and
Nasdaq-listed stocks. Older Nasdaq-only sessions instead use verified
exchange-specific ordinals, so a legitimate Nasdaq session never becomes
a fabricated missing NYSE session.
The reference calendar retains valid ticker dates even when prices are
unusable, but rejects dates closed on each observation's actual exchange.
Date-only U.S. equity data from 1971 onward use scheduled NYSE/Nasdaq sessions,
including effective-dated holidays, 1972–1980 presidential Election Day
closures, and exceptional full-day closures. The historical NYSE validator
also recognizes documented 1968 Wednesday paperwork-crisis closures and
other known special closures in 1968–1969. Standalone NYSE date validation
also requires explicit verified session ordinals for any pre-1971 observations;
that attestation remains required at training and ingestion boundaries. Observations on closed U.S. market
dates are rejected even when `Exchange` is omitted or session ordinals are
supplied. An optional `Exchange` column groups stocks on the same market:
`NYSE`/`XNYS` use the NYSE calendar; `NASDAQ`/`XNAS` use the U.S. calendar with independently verified Nasdaq differences. Nasdaq dates before its February 8, 1971 launch are rejected, and its shortened September 27, 1985 session is retained even though NYSE was closed;
`LSE` and `XLON` use the London Stock Exchange calendar, including Boxing Day.
Other explicitly named exchanges require independently verified
`Session_Number` metadata, even for weekdays; standalone calendar APIs also reject unverified unknown exchanges and blank identifiers.
Aliases (`NYSE`/`XNYS`, `NASDAQ`/`XNAS`, `LSE`/`XLON`,
`TSX`/`XTSE`) are canonicalized for duplicate detection and bootstrap
market-session gaps. The bootstrap consolidates equivalent market/date records after alias normalization, rejects conflicting session ordinals, and rejects duplicate scored ticker/date identities when tickers are provided. Their spelling never creates artificial missing sessions.
A peer's fabricated closed-market observation cannot contaminate a selected
stock's inferred calendar. When `Exchange` is
omitted, U.S. equity market rules apply rather than accepting arbitrary
observed holiday weekdays. Callers can supply verified `Session_Number`
metadata, required for pre-1971 NYSE dates where automatic inference cannot
safely reconstruct the historical schedule. Close-to-close daily
returns, RSI changes, and prediction targets do not bridge missing sessions.
Observation-window indicators (moving average, Bollinger Bands, and MACD)
retain earlier valid prices across market-session gaps, deliberately measuring
the most recent *observed* prices, not a guaranteed number of consecutive
sessions. Their windows never include future prices; returns, RSI changes,
and ATR's true range are invalidated at gaps because they require adjacent
sessions. The gap policy is covered by dedicated regression tests.
For recognized exchanges from 1971 onward, supplied ordinals must have
the same spacing as actual scheduled market sessions, so an omitted Tuesday
cannot be disguised as the next session after Monday. Verified session ordinals
also support historical NYSE Saturday trading through
its last Saturday session on May 24, 1952; later weekends are rejected.
Direct exchange-aware session utilities require daily dates **or supplied
verified session ordinals**; when no `Exchange` is present, undated synthetic
examples retain the existing sequential demonstration behavior. The standalone
`session_positions()` and `validate_session_positions()` helpers reject
previously coerced pandas datetime columns unless the caller verifies original
day-level precision and passes `verified_datetime_dates=True`, or supplies a
frame already marked by the validated pipeline. Explicit `calendar_start` and
`calendar_end` bounds must include every observed date; out-of-range bounds
are rejected instead of returning negative session indices.
For raw NumPy date scalars, use
`validate_raw_bar_records(records)` **before constructing a pandas DataFrame**:
it rejects year-, month-, or week-precision dates rather than inventing a day.
It also rejects finer-than-day NumPy timestamps and Python midnight datetime
values, since a midnight clock time does not attest to original daily precision.
Ordinary pandas construction may silently expand such scalars into a complete
timestamp, permanently discarding their original precision; a validator
receiving that already-converted DataFrame cannot reconstruct it. Therefore
`validate_bars()` rejects pandas datetime64/Timestamp dates by default;
`verified_datetime_dates=True` is an explicit attestation for callers who
independently checked their original date precision. Both
`prepare_indicator_history()` and `prepare_features()` enforce the same default:
they no longer bypass date provenance checks. When passing an already-validated
DataFrame returned by `load_data()`, explicitly set
`verified_datetime_dates=True` on either preparation function; unverified raw
pandas timestamps must not use that opt-in. ISO daily strings, Python
date objects, and explicit NumPy day-precision values are safe inputs to the
raw-record API. When selecting a ticker from a custom CSV, an unrelated stock's impossible
closed-market observation on a recognized exchange is excluded from shared
calendar inference, while the selected ticker's own invalid dates are rejected.
The standalone NYSE calendar validator also rejects missing (`NaT`) dates.
It rejects pre-1971 Sundays and Saturdays after the final May 1952 Saturday
session; other pre-1971 sessions still require verified market ordinals.
Direct multi-ticker indicator and trading-episode utilities infer shared
exchange calendars before splitting the source into individual stocks.
Trading episodes reject duplicate ticker/date records and sort parsed dates,
including valid non-zero-padded ISO dates.
HMM fitting uses separate complete contiguous sequences; forward filtering
restarts after incomplete observations or missing source sessions, so both use
the same boundary policy. Target dates must follow their
feature dates by exactly one recorded market session.
An experiment rejects a ticker with no usable validated history or without
observations in each chronological partition, and reports ticker coverage for
training, validation, and test separately.

