"""Fifth audit: numeric validity, exchange identity, and public session guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.hmm_integration import fit_hmm_on_indicators
from scripts.market_calendar import (
    _exchange_sessions,
    _scheduled_sessions,
    exchange_open_mask,
    nyse_open_mask,
    validate_exchange_open_dates,
)
from scripts.model_training import QuantileBinner, classification_metrics
from scripts.price_validation import valid_price_bars
from scripts.session_validation import session_positions, validate_session_positions
from scripts.walk_forward_evaluation import (
    _market_date_block_ids,
    paired_block_accuracy_intervals,
)


def _nyse_days(count: int = 32) -> list[str]:
    return (
        _scheduled_sessions(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-04-30"))[
            :count
        ]
        .strftime("%Y-%m-%d")
        .tolist()
    )


def _scores(dates: list[str]) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame(
        {
            "Date": dates,
            "actual": [index % 2 for index in range(n)],
            "majority": [0] * n,
            "baseline_prob": [0.25, 0.75] * (n // 2),
            "enhanced_prob": [0.6] * n,
        }
    )


@pytest.mark.parametrize("bad", [10 + 2j, 10 + 0j, "10+2j"])
def test_complex_ohlcv_values_rejected_even_if_real_part_is_valid(bad):
    frame = pd.DataFrame(
        {
            "Open": [bad],
            "High": [12 + 2j if isinstance(bad, complex) else "12"],
            "Low": [9.0],
            "Close": [11.0],
            "Volume": [100],
        }
    )
    assert not valid_price_bars(frame, require_volume=True).iloc[0]


def test_valid_real_numeric_price_strings_remain_supported():
    frame = pd.DataFrame(
        {
            "Open": ["10.0"],
            "High": ["12"],
            "Low": ["9"],
            "Close": ["11"],
            "Volume": ["100"],
        }
    )
    assert valid_price_bars(frame, require_volume=True).iloc[0]


def test_complex_classification_probabilities_rejected_without_truncation():
    labels = pd.Series([0, 1, 0, 1])
    invalid = np.array([0.1 + 0.2j, 0.9 - 0.2j, 0.4 + 1j, 0.7 + 4j])
    with pytest.raises(ValueError, match="complex"):
        classification_metrics(labels, invalid)


def test_complex_bootstrap_probabilities_rejected_without_truncation():
    frame = _scores(_nyse_days())
    frame["baseline_prob"] = np.array([0.3 + 0j] * len(frame))
    with pytest.raises(ValueError, match="complex"):
        paired_block_accuracy_intervals(frame, bootstrap_samples=100)


def test_complex_feature_bins_rejected_on_fit_and_transform():
    source = pd.DataFrame({"X": [1 + 3j, 2 + 0j, 3 + 1j]})
    with pytest.raises(ValueError, match="complex"):
        QuantileBinner().fit(source, ("X",))
    binner = QuantileBinner().fit(
        pd.DataFrame({"X": [1.0, 2.0, 3.0]}), ("X",)
    )
    with pytest.raises(ValueError, match="complex"):
        binner.transform(source, ("X",))


def test_complex_hmm_emissions_rejected_before_fitting():
    dates = _nyse_days(20)
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Name": ["AAPL"] * len(dates),
            "MA_20": np.array([10 + 1j] * len(dates)),
            "RSI_14": [45.0] * len(dates),
            "MACD": [0.1] * len(dates),
        }
    )
    with pytest.raises(ValueError, match="complex"):
        fit_hmm_on_indicators(frame)


def test_london_aliases_are_one_market_without_false_session_gaps():
    dates = _exchange_sessions(
        pd.Timestamp("2024-01-02"), pd.Timestamp("2024-03-04"), "LSE"
    )[:20].strftime("%Y-%m-%d").tolist()
    frame = _scores(dates)
    frame["Exchange"] = ["LSE", "XLON"] * 10
    block_ids = _market_date_block_ids(frame, block_sessions=10)
    assert len(np.unique(block_ids)) == 2
    assert block_ids[:10].tolist() == [0] * 10
    assert block_ids[10:].tolist() == [1] * 10


def test_duplicate_ticker_identity_is_canonicalized_before_checking():
    frame = pd.DataFrame(
        {
            "Name": ["AAPL", " aapl "],
            "Date": ["2024-01-09", "2024-01-09"],
        }
    )
    with pytest.raises(ValueError, match="Duplicate"):
        validate_session_positions(frame)


def test_duplicate_exchange_aliases_and_spacing_are_canonicalized():
    frame = pd.DataFrame(
        {
            "Exchange": ["NYSE", " xnys "],
            "Date": ["2024-01-09", "2024-01-09"],
        }
    )
    with pytest.raises(ValueError, match="Duplicate"):
        validate_session_positions(frame)


def test_exchange_aware_undated_inputs_require_source_ordinals():
    frame = pd.DataFrame({"Exchange": ["XTAE", "XTAE"]})
    with pytest.raises(ValueError, match="require Date or verified"):
        session_positions(frame)
    with pytest.raises(ValueError, match="require Date or verified"):
        validate_session_positions(frame)


def test_datedless_exchange_ordinals_remain_valid():
    frame = pd.DataFrame(
        {"Exchange": ["XTAE", "XTAE"], "Session_Number": [200, 201]}
    )
    validate_session_positions(frame, require_consecutive=True)
    assert session_positions(frame).tolist() == [200, 201]


def test_undated_nonexchange_demo_retains_legacy_positions():
    assert session_positions(pd.DataFrame({"x": [1, 2]})).tolist() == [0, 1]


@pytest.mark.parametrize("exchange", ["NYSE", "LSE", "XTAE"])
def test_public_calendar_rejects_intraday_observations(exchange):
    observed = pd.DatetimeIndex(["2024-01-09 13:30:00"])
    assert not exchange_open_mask(
        observed, exchange, verified_session_ordinals=True
    ).any()
    with pytest.raises(ValueError, match="closed-market date"):
        validate_exchange_open_dates(
            observed, exchange, verified_session_ordinals=True
        )


def test_nyse_calendar_rejects_timezone_aware_and_intraday_dates():
    assert not nyse_open_mask(
        pd.DatetimeIndex(["2024-01-09 13:30:00"])
    ).any()
    assert not nyse_open_mask(
        pd.DatetimeIndex(["2024-01-09"], tz="UTC")
    ).any()


def test_walk_forward_rejects_bootstrap_workload_before_model_fitting(monkeypatch):
    from scripts.data_processing import validate_bars
    import scripts.walk_forward_evaluation as evaluation

    sessions = _scheduled_sessions(
        pd.Timestamp("2024-01-02"), pd.Timestamp("2024-12-31")
    )[:135]
    close = 100 + np.arange(len(sessions)) * 0.05
    prices = validate_bars(
        pd.DataFrame(
            {
                "Date": sessions,
                "Name": ["AAPL"] * len(sessions),
                "Open": close - 0.1,
                "High": close + 1.0,
                "Low": close - 1.0,
                "Close": close,
                "Volume": [1000] * len(sessions),
            }
        ),
        verified_datetime_dates=True,
    )
    monkeypatch.setattr(evaluation, "load_data", lambda *args, **kwargs: prices)
    monkeypatch.setattr(
        evaluation,
        "train_and_select",
        lambda *args, **kwargs: pytest.fail("Model fitting ran before workload check"),
    )
    monkeypatch.setattr(evaluation, "MAX_BOOTSTRAP_DRAW_CELLS", 50)
    with pytest.raises(ValueError, match="Bootstrap workload"):
        evaluation.walk_forward_experiment(
            "synthetic.csv", folds=2, bootstrap_samples=100, block_sessions=1
        )
