"""Seventh audit: training labels, canonical sessions, historical NYSE and priors."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.market_calendar import (
    exchange_open_mask,
    nyse_open_mask,
    validate_nyse_open_dates,
)
from scripts.model_training import (
    ChronologicalSplit,
    fit_network,
    train_and_select,
)
from scripts.session_validation import session_positions


def _split(labels=(0, 1, 0)) -> ChronologicalSplit:
    dates = pd.bdate_range("2024-01-02", periods=9)
    frames = []
    for i in range(3):
        frames.append(
            pd.DataFrame(
                {
                    "Date": dates[3 * i : 3 * i + 3],
                    "Return": [0.1, 0.2, 0.3],
                    "Trend": list(labels),
                }
            )
        )
    return ChronologicalSplit(
        frames[0],
        frames[1],
        frames[2],
        dates[2],
        dates[5],
    )


@pytest.mark.parametrize("bad", [0.5, 1.9, 0 + 0j, 1 + 0j])
def test_training_rejects_invalid_labels_before_binning(monkeypatch, bad):
    from scripts.model_training import QuantileBinner

    split = _split(labels=(0, bad, 1))

    def impossible_fit(*args, **kwargs):
        pytest.fail("Binning happened before training-label validation")

    monkeypatch.setattr(QuantileBinner, "fit", impossible_fit)
    with pytest.raises(ValueError, match="Training labels"):
        train_and_select(split, ("Return",), 2, sample_sizes=(5,))


@pytest.mark.parametrize("bad", [0.5, 1.9, 1 + 0j])
def test_fit_network_rejects_invalid_binary_label_directly(bad):
    frame = pd.DataFrame({"Return": [0, 1, 0], "Trend": [0, bad, 1]})
    with pytest.raises(ValueError, match="Training labels"):
        fit_network(frame, ("Return",), 2, 5)


@pytest.mark.parametrize(
    "candidates",
    [(), [], (0,), (-1,), (float("nan"),), (float("inf"),), (True,), (5, 0), None],
)
def test_invalid_prior_grid_fails_before_binning(monkeypatch, candidates):
    from scripts.model_training import QuantileBinner

    def impossible_fit(*args, **kwargs):
        pytest.fail("Binning happened before prior-grid validation")

    monkeypatch.setattr(QuantileBinner, "fit", impossible_fit)
    with pytest.raises(ValueError, match="sample_sizes"):
        train_and_select(_split(), ("Return",), 2, sample_sizes=candidates)


def test_alias_duplicate_ticker_dates_rejected_by_inference():
    frame = pd.DataFrame(
        {"Name": ["AAPL", " aapl "], "Date": ["2024-01-09", "2024-01-09"]}
    )
    with pytest.raises(ValueError, match="Duplicate ticker/date"):
        session_positions(frame)


def test_unlabeled_duplicate_dates_rejected_by_inference():
    frame = pd.DataFrame({"Date": ["2024-01-09", "2024-01-09"]})
    with pytest.raises(ValueError, match="Duplicate ticker/date"):
        session_positions(frame)


def test_valid_multi_ticker_market_date_remains_supported():
    frame = pd.DataFrame(
        {
            "Name": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "Date": ["2024-01-09", "2024-01-09", "2024-01-10", "2024-01-10"],
        }
    )
    assert session_positions(frame).tolist() == [0, 0, 1, 1]


@pytest.mark.parametrize(
    "closed",
    [
        "1968-06-12",
        "1968-07-05",
        "1968-07-10",
        "1968-11-05",
        "1968-12-18",
        "1969-02-10",
        "1969-03-31",
        "1969-07-21",
    ],
)
def test_historical_nyse_special_closures_are_not_sessions(closed):
    date = pd.DatetimeIndex([closed])
    assert not nyse_open_mask(date).any()
    with pytest.raises(ValueError, match="closed-market date"):
        validate_nyse_open_dates(date)


def test_historical_nearby_nyse_open_days_and_saturday_preserved():
    dates = pd.DatetimeIndex(["1968-07-09", "1968-07-11", "1951-01-06"])
    assert nyse_open_mask(dates).tolist() == [False, False, False]
    assert nyse_open_mask(
        dates, verified_session_ordinals=True
    ).tolist() == [True, True, True]


@pytest.mark.parametrize("alias", ["N/A", " NaN ", "NULL", "none", "<NA>"])
def test_placeholder_exchange_rejected_even_with_verified_ordinals(alias):
    with pytest.raises(ValueError, match="nonplaceholder"):
        exchange_open_mask(
            pd.DatetimeIndex(["2024-01-09"]),
            alias,
            verified_session_ordinals=True,
        )
