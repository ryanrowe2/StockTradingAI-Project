"""Regression tests for causality, data boundaries, inference, and simulation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data_processing import load_data, prepare_features  # noqa: E402
from hmm_integration import fit_hmm_on_indicators, infer_filtered_regimes  # noqa: E402
from model_training import (  # noqa: E402
    BASELINE_FEATURES, QuantileBinner, chronological_split,
    classification_metrics, fit_network, predict_up_probability, run_experiment,
)
from rl_agent import QLearningAgent, SimpleTradingEnv  # noqa: E402
from technical_indicators import add_technical_indicators  # noqa: E402


def synthetic_bars(days: int = 130) -> pd.DataFrame:
    dates = pd.bdate_range("2019-01-01", periods=days)
    frames = []
    for ticker, base in (("AAA", 30.0), ("BBB", 300.0)):
        index = np.arange(days)
        close = base + index * 0.15 + 2.5 * np.sin(index * 0.6)
        frames.append(pd.DataFrame({
            "Date": dates, "Name": ticker,
            "Open": close - 0.3, "High": close + 1,
            "Low": close - 1, "Close": close,
            "Volume": 100000 + index * 10,
        }))
    return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=7).reset_index(drop=True)


def test_target_and_indicators_never_cross_symbol_boundaries():
    original = synthetic_bars()
    features = prepare_features(original)
    for ticker, group in features.groupby("Name"):
        input_group = original.loc[original["Name"] == ticker].sort_values("Date").reset_index(drop=True)
        assert len(group) > 50
        assert set(group["Name"]) == {ticker}
        assert (group["Date"] < group["Target_Date"]).all()
        for _, row in group.iterrows():
            current_index = input_group.index[input_group["Date"] == row["Date"]][0]
            assert row["Trend"] == int(input_group.loc[current_index + 1, "Close"] > row["Close"])
            assert row["Close_Lag1"] == pytest.approx(input_group.loc[current_index - 1, "Close"])
        assert group["Target_Date"].max() <= input_group["Date"].max()


def test_technical_indicators_do_not_change_when_future_is_modified():
    data = synthetic_bars()
    original = add_technical_indicators(data)
    latest = data.loc[data["Name"] == "AAA", "Date"].max()
    edited = data.copy()
    edited.loc[(edited["Name"] == "AAA") & (edited["Date"] == latest), "Close"] += 1000
    changed = add_technical_indicators(edited)
    subset = (original["Name"] == "AAA") & (original["Date"] < latest)
    columns = ["MA_20", "RSI_14", "MACD", "ATR", "Return", "Close_Lag1", "BB_Upper"]
    pd.testing.assert_frame_equal(original.loc[subset, columns], changed.loc[subset, columns])


def test_load_data_validates_and_removes_incomplete_rows(tmp_path: Path):
    source = synthetic_bars(60)
    source.loc[0, "High"] = np.nan
    csv = tmp_path / "bars.csv"
    source.to_csv(csv, index=False)
    loaded = load_data(csv)
    assert len(loaded) == len(source) - 1
    assert (loaded["High"] > 0).all()
    assert not loaded.duplicated(["Name", "Date"]).any()


def test_chronological_split_purges_future_labels_and_symbols_remain_aligned():
    split = chronological_split(prepare_features(synthetic_bars()))
    assert split.train["Target_Date"].max() <= split.train_end
    assert split.validation["Date"].min() > split.train_end
    assert split.validation["Target_Date"].max() <= split.validation_end
    assert split.test["Date"].min() > split.validation_end
    assert set(split.train["Name"]) == {"AAA", "BBB"}
    assert set(split.validation["Name"]) == {"AAA", "BBB"}
    assert set(split.test["Name"]) == {"AAA", "BBB"}


def test_binner_learns_thresholds_only_from_training_data():
    train = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6]})
    binner = QuantileBinner(3).fit(train, ("A",))
    edges = binner.edges["A"].copy()
    scores = binner.transform(pd.DataFrame({"A": [-1000, 1000]}), ("A",))
    assert scores["A"].tolist() == [0, 2]
    np.testing.assert_array_equal(binner.edges["A"], edges)


def test_network_predictions_and_metrics_are_finite():
    examples = pd.DataFrame({
        "Open": [0, 0, 1, 1, 0, 1] * 10,
        "High": [0, 0, 1, 1, 0, 1] * 10,
        "Low": [0, 0, 1, 1, 0, 1] * 10,
        "Trend": [0, 0, 1, 1, 0, 1] * 10,
    })
    network = fit_network(examples, BASELINE_FEATURES, 2, 5)
    probs = predict_up_probability(network, examples)
    assert len(probs) == len(examples)
    assert np.isfinite(probs).all()
    assert probs[0] < 0.5 < probs[2]
    assert classification_metrics(examples["Trend"], probs)["accuracy"] > 0.9


def test_hmm_filtering_is_prefix_invariant():
    frame = prepare_features(synthetic_bars())
    ticker = frame.loc[frame["Name"] == "AAA"].reset_index(drop=True)
    train = ticker.iloc[:65]
    model, train_regimes = fit_hmm_on_indicators(train, n_components=2)
    full = infer_filtered_regimes(model, ticker)
    prefix = infer_filtered_regimes(model, ticker.iloc[:72].copy())
    assert train_regimes["Market_Regime"].notna().all()
    assert full["Market_Regime"].notna().all()
    assert prefix["Market_Regime"].tolist() == full["Market_Regime"].iloc[:72].tolist()


def test_rl_positions_reward_terminal_and_transaction_costs():
    data = pd.DataFrame({"Close": [100.0, 120.0, 110.0]})
    env = SimpleTradingEnv(data, transaction_cost=0.01)
    state = env.reset()
    assert state[-1] == 0
    state, reward, done, info = env.step("buy")
    assert reward == pytest.approx(19.0)
    assert state[-1] == 1 and not done
    state, reward, done, info = env.step("sell")
    assert reward == pytest.approx(-1.2)
    assert state[-1] == 0 and done
    with pytest.raises(RuntimeError):
        env.step("hold")


def test_rl_terminal_update_does_not_bootstrap():
    agent = QLearningAgent(alpha=1, gamma=0.95)
    agent.q_table[(1,)] = {"buy": 100.0, "sell": 20.0, "hold": 1.0}
    agent.update_q_value((0,), "buy", 7.0, (1,), done=True)
    assert agent.q_table[(0,)]["buy"] == 7.0


def test_hmm_and_rl_reject_multi_ticker_episodes():
    data = prepare_features(synthetic_bars())
    with pytest.raises(ValueError, match="one ticker"):
        fit_hmm_on_indicators(data)
    with pytest.raises(ValueError, match="one ticker"):
        SimpleTradingEnv(data)


def test_end_to_end_on_independent_dates_and_valid_notebooks(tmp_path: Path):
    csv = tmp_path / "synthetic.csv"
    synthetic_bars().to_csv(csv, index=False)
    output = run_experiment(csv, num_bins=3)
    assert output["tickers"] == 2
    assert min(output["observations"].values()) > 0
    for model in ("baseline", "enhanced"):
        assert 0 <= output[model]["test"]["accuracy"] <= 1
        assert 0 <= output[model]["validation"]["accuracy"] <= 1
    import nbformat
    for notebook in sorted((ROOT / "notebooks").glob("*.ipynb")):
        nbformat.validate(nbformat.read(notebook, as_version=4))
