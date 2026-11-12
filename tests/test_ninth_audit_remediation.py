"""Ninth audit: historical session accuracy, source calendars, RL and bootstrap."""

from __future__ import annotations

import copy
import random

import numpy as np
import pandas as pd
import pytest

from scripts import data_processing
from scripts.market_calendar import (
    exchange_open_mask,
    inferred_equity_sessions,
    nyse_open_mask,
    validate_exchange_open_dates,
)
from scripts.rl_agent import QLearningAgent
from scripts.walk_forward_evaluation import paired_block_accuracy_intervals


@pytest.mark.parametrize(
    "closed",
    [
        "1954-12-24",
        "1963-11-25",
        "1964-07-03",
        "1965-12-24",
        "1966-02-22",
        "1966-05-30",
        "1968-07-10",
    ],
)
def test_verified_nyse_ordinal_cannot_attest_historic_closures(closed):
    dates = pd.DatetimeIndex([closed])
    assert not nyse_open_mask(dates, verified_session_ordinals=True).any()
    with pytest.raises(ValueError, match="closed-market date"):
        validate_exchange_open_dates(
            dates, "XNYS", verified_session_ordinals=True
        )


def test_historic_independence_day_friday_observed_from_1954():
    # In 1964 July 4 fell on Saturday, so the NYSE closed on Friday.
    dates = pd.DatetimeIndex(["1964-07-02", "1964-07-03", "1964-07-06"])
    assert nyse_open_mask(dates, verified_session_ordinals=True).tolist() == [
        True,
        False,
        True,
    ]


def test_nasdaq_launch_and_hurricane_gloria_half_session():
    old = pd.DatetimeIndex(["1971-02-05"])
    first = pd.DatetimeIndex(["1971-02-08"])
    gloria = pd.DatetimeIndex(["1985-09-27"])
    assert not exchange_open_mask(old, "NASDAQ", verified_session_ordinals=True).any()
    assert exchange_open_mask(first, "XNAS").all()
    assert exchange_open_mask(gloria, "NASDAQ").all()
    assert not exchange_open_mask(gloria, "XNYS").any()
    with pytest.raises(ValueError, match="closed-market date"):
        validate_exchange_open_dates(old, "XNAS")
    with pytest.raises(ValueError, match="closed-market date"):
        inferred_equity_sessions(old, exchange="XNAS")
    sessions = inferred_equity_sessions(
        pd.DatetimeIndex(["1985-09-26", "1985-09-30"]),
        exchange="XNAS",
    )
    assert sessions.strftime("%Y-%m-%d").tolist() == [
        "1985-09-26",
        "1985-09-27",
        "1985-09-30",
    ]
    nyse = inferred_equity_sessions(
        pd.DatetimeIndex(["1985-09-26", "1985-09-30"]),
        exchange="XNYS",
    )
    assert nyse.strftime("%Y-%m-%d").tolist() == [
        "1985-09-26",
        "1985-09-30",
    ]


def test_bundled_reference_calendar_ignores_closed_peer_market_dates(
    tmp_path, monkeypatch
):
    source = tmp_path / data_processing.DEFAULT_RAW_DATA.name
    source.write_text(
        "Date,Name,Open,High,Low,Close,Volume\n"
        "2024-01-12,AAA,10,12,9,11,100\n"
        "2024-01-15,BBB,10,12,9,11,100\n"
        "2024-01-16,AAA,11,13,10,12,100\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(data_processing, "DEFAULT_RAW_DATA", source)
    selected = data_processing.load_data(source, tickers=("AAA",))
    assert selected["Session_Number"].tolist() == [0, 1]
    assert data_processing._bundled_session_calendar().strftime(
        "%Y-%m-%d"
    ).tolist() == ["2024-01-12", "2024-01-16"]
    with pytest.raises(ValueError, match="market|closed|session"):
        data_processing.load_data(source, tickers=("BBB",))


@pytest.mark.parametrize(
    "invalid",
    [2 + 0j, 2 + 7j, np.complex64(2 + 0j), np.complex128(2 + 7j)],
)
def test_complex_q_reward_rejected_without_mutating_q_table(invalid):
    agent = QLearningAgent()
    agent.q_table[(0,)] = {"buy": 1.0, "sell": 0.0, "hold": 0.0}
    before = copy.deepcopy(agent.q_table)
    with pytest.raises(ValueError, match="Reward.*complex"):
        agent.update_q_value((0,), "buy", invalid, (1,))
    assert agent.q_table == before


@pytest.mark.parametrize(
    "invalid",
    [None, True, np.bool_(True), -1, 1.0, 1.5, "42", float("nan")],
)
def test_q_agent_rejects_unsupported_random_seeds(invalid):
    with pytest.raises(ValueError, match="Invalid Q-learning parameters"):
        QLearningAgent(seed=invalid)


def test_q_agent_accepts_numpy_int_seed_and_reproduces_random_stream():
    one = QLearningAgent(seed=np.int64(17), epsilon=1)
    two = QLearningAgent(seed=17, epsilon=1)
    assert [one.choose_action((0,)) for _ in range(20)] == [
        two.choose_action((0,)) for _ in range(20)
    ]
    assert one.rng.random() == two.rng.random() == random.Random(17).random() or (
        one.rng.random() != two.rng.random()
    ) is False


def _scores() -> pd.DataFrame:
    dates = pd.bdate_range("2024-05-01", periods=20)
    return pd.DataFrame(
        {
            "Date": dates.repeat(2),
            "Name": ["AAA", "BBB"] * len(dates),
            "actual": [0, 1] * len(dates),
            "majority": [0] * (2 * len(dates)),
            "baseline_prob": [0.2] * (2 * len(dates)),
            "enhanced_prob": [0.8] * (2 * len(dates)),
        }
    )


def test_bootstrap_accepts_two_different_tickers_on_same_dates():
    assert "baseline" in paired_block_accuracy_intervals(
        _scores(), verified_datetime_dates=True, bootstrap_samples=100
    )


@pytest.mark.parametrize("conflict", [False, True])
def test_bootstrap_rejects_duplicate_scored_ticker_dates(conflict):
    scored = _scores()
    extra = scored.iloc[[0]].copy()
    extra["Name"] = " aaa "
    if conflict:
        extra["baseline_prob"] = 0.95
    scored = pd.concat([scored, extra], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate scored ticker/date"):
        paired_block_accuracy_intervals(
            scored, verified_datetime_dates=True, bootstrap_samples=100
        )
