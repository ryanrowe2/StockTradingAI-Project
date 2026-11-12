"""Regressions for exchange-session spacing and date-source provenance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.data_processing import validate_bars
from scripts.date_validation import mark_verified_daily_dates
from scripts.hmm_integration import _validated_indicators
from scripts.market_calendar import inferred_equity_sessions
from scripts.rl_agent import SimpleTradingEnv, contiguous_trading_segments
from scripts.session_validation import session_positions
from scripts.technical_indicators import add_technical_indicators


def bars(dates, exchange=None, ordinals=None):
    length = len(dates)
    result = pd.DataFrame({
        "Date": dates,
        "Name": ["AAA"] * length,
        "Open": [10.0] * length,
        "High": [12.0] * length,
        "Low": [9.0] * length,
        "Close": [11.0] * length,
        "Volume": [100] * length,
    })
    if exchange is not None:
        result["Exchange"] = [exchange] * length
    if ordinals is not None:
        result["Session_Number"] = ordinals
    return result


@pytest.mark.parametrize("exchange", [None, "NASDAQ", "XNAS", "NYSE", "XNYS"])
def test_us_exchange_supplied_ordinals_cannot_skip_a_real_session(exchange):
    source = bars(["2024-06-03", "2024-06-05"], exchange, [100, 101])
    with pytest.raises(ValueError, match="Session_Number spacing"):
        validate_bars(source)
    correct = validate_bars(source.assign(Session_Number=[100, 102]))
    assert correct["Session_Number"].tolist() == [100, 102]


@pytest.mark.parametrize("exchange", ["LSE", "XLON", "XTSE", "TSX"])
def test_known_foreign_exchange_supplied_ordinals_cannot_skip_a_session(exchange):
    source = bars(["2024-06-03", "2024-06-05"], exchange, [100, 101])
    with pytest.raises(ValueError, match="Session_Number spacing"):
        validate_bars(source)
    assert validate_bars(source.assign(Session_Number=[100, 102])).shape[0] == 2


@pytest.mark.parametrize("exchange", ["XTSE", "TSX"])
def test_toronto_boxing_day_does_not_create_fictitious_session(exchange):
    source = bars(["2024-12-24", "2024-12-27"], exchange)
    assert session_positions(source).tolist() == [0, 1]
    assert validate_bars(source)["Session_Number"].tolist() == [0, 1]
    with pytest.raises(ValueError, match="closed-market date"):
        validate_bars(bars(["2024-12-24", "2024-12-26"], exchange))


def test_unknown_exchange_needs_verified_ordinals():
    source = bars(["2024-12-24", "2024-12-27"], "XTEST")
    with pytest.raises(ValueError, match="requires verified Session_Number"):
        validate_bars(source)
    assert len(validate_bars(source.assign(Session_Number=[100, 101]))) == 2
    with pytest.raises(ValueError, match="requires verified Session_Number"):
        inferred_equity_sessions(pd.DatetimeIndex(source["Date"]), exchange="XTEST")


def month_coerced():
    return bars(pd.Series([np.datetime64("2024-04"), np.datetime64("2024-05")]))


def test_direct_indicators_reject_unverified_coerced_months():
    source = month_coerced()
    assert str(source["Date"].dtype).startswith("datetime64")
    with pytest.raises(ValueError, match="unknown original precision"):
        add_technical_indicators(source)
    assert len(add_technical_indicators(source, verified_datetime_dates=True)) == 2


def test_direct_trading_apis_reject_unverified_coerced_months():
    source = month_coerced()
    with pytest.raises(ValueError, match="unknown original precision"):
        contiguous_trading_segments(source)
    with pytest.raises(ValueError, match="unknown original precision"):
        SimpleTradingEnv(source)


def test_direct_hmm_helper_rejects_unverified_coerced_months():
    source = month_coerced()
    source["MA_20"] = [10.0, 11.0]
    with pytest.raises(ValueError, match="unknown original precision"):
        _validated_indicators(source, ("MA_20",))
    assert len(
        _validated_indicators(
            source, ("MA_20",), verified_datetime_dates=True
        )
    ) == 2


def test_validated_frame_preserves_provenance_for_direct_apis():
    source = validate_bars(bars(["2024-04-01", "2024-04-02"]))
    assert source.attrs["stocktrading_verified_daily_dates"] is True
    assert len(add_technical_indicators(source)) == 2
    assert len(SimpleTradingEnv(source).data) == 2
    assert len(contiguous_trading_segments(source)) == 1


def test_attested_datetime_frame_accepted_by_direct_apis():
    source = mark_verified_daily_dates(
        bars(pd.to_datetime(["2024-04-01", "2024-04-02"]))
    )
    assert len(add_technical_indicators(source)) == 2
    assert len(SimpleTradingEnv(source).data) == 2
