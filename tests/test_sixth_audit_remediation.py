"""Sixth audit: canonical bootstrap aliases, real-valued inputs, and early seeds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.market_calendar import (
    _exchange_sessions,
    exchange_open_mask,
    validate_exchange_open_dates,
)
from scripts.model_training import classification_metrics
from scripts.rl_agent import SimpleTradingEnv
from scripts.walk_forward_evaluation import (
    _market_date_block_ids,
    paired_block_accuracy_intervals,
    walk_forward_experiment,
)


def _london_scores():
    dates = _exchange_sessions(
        pd.Timestamp("2024-01-02"), pd.Timestamp("2024-03-01"), "LSE"
    )[:24]
    rows = []
    for number, date in enumerate(dates):
        for alias in ("LSE", "XLON"):
            rows.append(
                {
                    "Date": date.strftime("%Y-%m-%d"),
                    "Exchange": alias,
                    "actual": number % 2,
                    "majority": 0,
                    "baseline_prob": 0.6,
                    "enhanced_prob": 0.4,
                }
            )
    return pd.DataFrame(rows)


def test_london_aliases_with_same_date_are_one_bootstrap_market():
    scored = _london_scores()
    blocks = _market_date_block_ids(scored, block_sessions=10)
    assert blocks[:20].tolist() == [0] * 20
    assert len(np.unique(blocks)) == 3
    result = paired_block_accuracy_intervals(
        scored, block_sessions=10, bootstrap_samples=100
    )
    assert set(result) == {"baseline", "enhanced"}


def test_same_market_aliases_with_equal_ordinals_are_deduplicated():
    scored = _london_scores()
    scored["Session_Number"] = np.repeat(np.arange(24), 2)
    blocks = _market_date_block_ids(scored, block_sessions=10)
    assert len(np.unique(blocks)) == 3


def test_conflicting_alias_session_ordinals_are_rejected():
    scored = _london_scores()
    scored["Session_Number"] = np.repeat(np.arange(24), 2)
    scored.loc[1, "Session_Number"] += 5
    with pytest.raises(ValueError, match="Conflicting session ordinals"):
        _market_date_block_ids(scored, block_sessions=10)


@pytest.mark.parametrize("bad_price", [100 + 1j, 100 + 0j])
def test_standalone_trading_environment_rejects_complex_prices(bad_price):
    bars = pd.DataFrame(
        {
            "Open": [bad_price, 101.0],
            "Close": [100.0, 102.0],
        }
    )
    with pytest.raises(ValueError, match="complex"):
        SimpleTradingEnv(bars)


def test_trading_environment_keeps_numeric_string_prices():
    bars = pd.DataFrame(
        {"Open": ["100", "101"], "Close": ["100.5", "101.5"]}
    )
    env = SimpleTradingEnv(bars)
    assert env.get_state() == (0,)


def test_complex_classification_labels_are_rejected():
    labels = pd.Series(np.array([0 + 0j, 1 + 0j, 0 + 0j, 1 + 0j]))
    with pytest.raises(ValueError, match="complex"):
        classification_metrics(labels, np.array([0.1, 0.9, 0.4, 0.8]))


@pytest.mark.parametrize("column", ["actual", "majority"])
def test_complex_bootstrap_labels_are_rejected(column):
    scored = _london_scores()
    scored[column] = scored[column].astype(complex)
    with pytest.raises(ValueError, match="complex"):
        paired_block_accuracy_intervals(scored, bootstrap_samples=100)


def test_ordinary_integer_binary_labels_remain_supported():
    report = classification_metrics(
        pd.Series([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8])
    )
    assert report["accuracy"] == 1.0


def test_unknown_market_weekdays_need_verified_ordinals():
    dates = pd.DatetimeIndex(["2024-01-08"])
    with pytest.raises(ValueError, match="requires verified Session_Number"):
        exchange_open_mask(dates, "XTAE")
    with pytest.raises(ValueError, match="requires verified Session_Number"):
        validate_exchange_open_dates(dates, "XTAE")
    assert exchange_open_mask(
        dates, "XTAE", verified_session_ordinals=True
    ).tolist() == [True]


@pytest.mark.parametrize("invalid", ["", "  ", 123])
def test_blank_or_invalid_exchange_identifiers_are_rejected(invalid):
    with pytest.raises(ValueError, match="nonempty identifier"):
        exchange_open_mask(pd.DatetimeIndex(["2024-01-08"]), invalid)


@pytest.mark.parametrize("seed", [-1, True, "not-a-seed"])
def test_invalid_seed_rejected_before_loading_or_fitting(monkeypatch, seed):
    import scripts.walk_forward_evaluation as evaluation

    monkeypatch.setattr(
        evaluation,
        "load_data",
        lambda *args, **kwargs: pytest.fail("Data loaded before seed validation"),
    )
    with pytest.raises(ValueError, match="seed"):
        walk_forward_experiment("unused.csv", seed=seed)


def test_valid_seed_reproduces_identical_bootstrap_results():
    scored = _london_scores()
    one = paired_block_accuracy_intervals(scored, seed=17, bootstrap_samples=100)
    two = paired_block_accuracy_intervals(scored, seed=17, bootstrap_samples=100)
    assert one == two
