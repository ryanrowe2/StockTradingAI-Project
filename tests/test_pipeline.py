"""Regression tests for causality, data boundaries, inference, and simulation."""

from __future__ import annotations

import copy
import base64
import io
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data_processing import DEFAULT_RAW_DATA, load_data, prepare_features  # noqa: E402
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


def test_direct_indicator_api_orders_parsed_dates():
    bars = pd.DataFrame({
        "Date": ["2020-1-10", "2020-1-2", "2020-1-3"],
        "Name": ["AAA", "AAA", "AAA"],
        "Open": [110.0, 90.0, 100.0], "High": [120.0, 95.0, 105.0],
        "Low": [100.0, 85.0, 95.0], "Close": [115.0, 90.0, 100.0],
    })
    result = add_technical_indicators(bars)
    assert result["Date"].dt.day.tolist() == [2, 3, 10]
    assert result["Close_Lag1"].iloc[1:].tolist() == [90.0, 100.0]
    with pytest.raises(ValueError, match="Date is required"):
        add_technical_indicators(bars.drop(columns="Date"))
    with pytest.raises(ValueError, match="parseable"):
        add_technical_indicators(bars.assign(Date=["invalid", "2020-1-2", "2020-1-3"]))


def test_model_features_are_invariant_to_ticker_price_scale():
    bars = synthetic_bars()
    original = prepare_features(bars).query("Name == 'AAA'")
    scaled = bars.copy()
    scaled.loc[scaled.Name == "AAA", ["Open", "High", "Low", "Close"]] *= 5
    transformed = prepare_features(scaled).query("Name == 'AAA'")
    features = ("Return", "Intraday_Return", "Daily_Range", "MACD_Pct")
    np.testing.assert_allclose(original[list(features)], transformed[list(features)], rtol=1e-10)


def test_load_data_validates_and_removes_incomplete_rows(tmp_path: Path):
    source = synthetic_bars(60)
    source.loc[0, "High"] = np.nan
    csv = tmp_path / "bars.csv"
    source.to_csv(csv, index=False)
    loaded = load_data(csv)
    assert len(loaded) == len(source) - 1
    assert (loaded["High"] > 0).all()
    assert not loaded.duplicated(["Name", "Date"]).any()


def test_loader_excludes_inconsistent_prices_and_rejects_invalid_dates(tmp_path: Path):
    data = synthetic_bars(70)
    data.loc[0, "Open"] = 2000.0
    csv = tmp_path / "invalid.csv"
    data.to_csv(csv, index=False)
    # The contradictory bar is excluded rather than being silently altered.
    remaining = load_data(csv)
    assert len(remaining) == len(data) - 1
    assert not ((remaining["High"] < remaining[["Open", "Close", "Low"]].max(axis=1)) |
                (remaining["Low"] > remaining[["Open", "Close", "High"]].min(axis=1))).any()
    with pytest.raises(ValueError, match="Date"):
        load_data(io.StringIO("Date,Name,Open,High,Low,Close,Volume\nnot-a-date,AAA,10,11,9,10,100\n"))


def test_loader_parses_valid_mixed_date_formats_without_lexical_ordering():
    raw = io.StringIO(
        "Date,Name,Open,High,Low,Close,Volume\n"
        "2020-1-10,AAA,10,11,9,10,100\n"
        "2020-01-02,AAA,10,11,9,10,100\n"
        "2020-1-3,AAA,10,11,9,10,100\n"
    )
    assert load_data(raw)["Date"].dt.day.tolist() == [2, 3, 10]


def test_loader_excludes_nonfinite_prices_and_volume(tmp_path: Path):
    data = synthetic_bars(70)
    data["Volume"] = data["Volume"].astype(float)
    data.loc[0, "High"] = np.inf
    data.loc[1, "Volume"] = np.inf
    csv = tmp_path / "nonfinite.csv"
    data.to_csv(csv, index=False)
    cleaned = load_data(csv)
    assert len(cleaned) == len(data) - 2
    assert np.isfinite(cleaned[["Open", "High", "Low", "Close", "Volume"]].to_numpy()).all()


def test_historical_source_contains_only_consistent_bars_after_validation():
    cleaned = load_data(DEFAULT_RAW_DATA)
    assert len(cleaned) == 93435
    assert (cleaned["High"] >= cleaned[["Open", "Close", "Low"]].max(axis=1)).all()
    assert (cleaned["Low"] <= cleaned[["Open", "Close", "High"]].min(axis=1)).all()


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


@pytest.mark.parametrize("value", [2.5, 4.0, True, "4", 0, 21])
def test_binner_rejects_invalid_bin_counts_at_construction(value):
    with pytest.raises(ValueError, match="integer between 2 and 20"):
        QuantileBinner(value)


def test_binner_accepts_numpy_integer_counts():
    assert QuantileBinner(np.int64(3)).num_bins == 3


def test_network_predictions_and_metrics_are_finite():
    examples = pd.DataFrame({
        **{name: [0, 0, 1, 1, 0, 1] * 10 for name in BASELINE_FEATURES},
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


def test_hmm_uses_chronological_dates_even_when_strings_are_not_zero_padded():
    frame = pd.DataFrame({
        "Date": ["2020-1-10", "2020-1-2", "2020-1-3"],
        "MA_20": [3.0, 1.0, 2.0], "RSI_14": [1.0, 2.0, 3.0],
        "MACD": [1.0, 2.0, 3.0],
    })
    from hmm_integration import _validated_indicators
    dates = _validated_indicators(frame, ("MA_20", "RSI_14", "MACD"))["Date"]
    assert dates.dt.day.tolist() == [2, 3, 10]
    with pytest.raises(ValueError, match="parseable"):
        _validated_indicators(frame.assign(Date=["invalid", "2020-1-2", "2020-1-3"]), ("MA_20",))


def test_hmm_fit_completes_without_exhausting_iteration_limit():
    data = prepare_features(synthetic_bars()).query("Name == 'AAA'")
    model, _ = fit_hmm_on_indicators(data.iloc[:65], n_components=2)
    assert model.monitor_.iter < model.n_iter


def test_historical_hmm_fit_reaches_configured_tolerance():
    data = load_data(DEFAULT_RAW_DATA)
    stock = prepare_features(data.loc[data["Name"] == "AAPL"])
    model, _ = fit_hmm_on_indicators(chronological_split(stock).train, n_components=3)
    assert model.monitor_.iter < model.n_iter
    assert abs(model.monitor_.history[-1] - model.monitor_.history[-2]) < model.tol


def test_rl_positions_reward_terminal_and_transaction_costs():
    data = pd.DataFrame({"Open": [99.0, 110.0, 100.0], "Close": [100.0, 120.0, 110.0]})
    env = SimpleTradingEnv(data, transaction_cost=0.01)
    state = env.reset()
    assert state[-1] == 0
    state, reward, done, info = env.step("buy")
    # Buy after day 0 closes; execute at the following 110 open.
    assert reward == pytest.approx(8.9)
    assert info["execution_price"] == 110
    assert state[-1] == 1 and not done
    state, reward, done, info = env.step("sell")
    # The overnight gap from 120 to 100 belongs to the existing long.
    assert reward == pytest.approx(-21.0)
    assert state[-1] == 0 and done
    with pytest.raises(RuntimeError):
        env.step("hold")


def test_trading_disallows_actions_that_cannot_execute():
    bars = pd.DataFrame({"Open": [99.0, 101.0, 102.0], "Close": [100.0, 102.0, 103.0]})
    env = SimpleTradingEnv(bars)
    with pytest.raises(ValueError, match="Cannot sell"):
        env.step("sell")
    env.step("buy")
    with pytest.raises(ValueError, match="Cannot buy"):
        env.step("buy")
    with pytest.raises(ValueError, match="Missing trading state columns"):
        SimpleTradingEnv(bars, state_columns=("Return_binned",))
    invalid_states = bars.assign(Return_binned=pd.Series([0, pd.NA, 1], dtype="Int64"))
    with pytest.raises(ValueError, match="finite integer states"):
        SimpleTradingEnv(invalid_states, state_columns=("Return_binned",))


def test_trading_sorts_parsed_dates_and_rejects_invalid_or_duplicate_dates():
    bars = pd.DataFrame({
        "Date": ["2020-1-10", "2020-1-2", "2020-1-3"],
        "Open": [110.0, 90.0, 100.0], "Close": [115.0, 95.0, 105.0],
    })
    original_dates = bars["Date"].tolist()
    env = SimpleTradingEnv(bars)
    assert env.data["Date"].dt.day.tolist() == [2, 3, 10]
    assert bars["Date"].tolist() == original_dates
    with pytest.raises(ValueError, match="parseable"):
        SimpleTradingEnv(bars.assign(Date=["invalid", "2020-1-2", "2020-1-3"]))
    with pytest.raises(ValueError, match="Duplicate trading dates"):
        SimpleTradingEnv(bars.assign(Date=["2020-1-2", "2020-01-02", "2020-1-3"]))


@pytest.mark.parametrize("actions", [
    ("dance",), ("buy", "hold", "dance"), ("buy", "buy", "hold"),
    ("buy",), ("sell",), (),
])
def test_qlearning_rejects_invalid_or_unusable_action_sets(actions):
    with pytest.raises(ValueError, match="Invalid Q-learning parameters"):
        QLearningAgent(actions=actions)


def test_qlearning_allows_legal_custom_action_sets():
    assert QLearningAgent(actions=("hold",)).choose_action((0,), explore=False) == "hold"
    agent = QLearningAgent(actions=("buy", "sell"), epsilon=0)
    assert agent.choose_action((0,), explore=False) == "buy"
    assert agent.choose_action((1,), explore=False) == "sell"


def test_policy_masks_invalid_actions_and_bootstrapping():
    agent = QLearningAgent(alpha=1, gamma=0.95, epsilon=0)
    agent.q_table[(0,)] = {"buy": 3.0, "sell": 1000.0, "hold": 1.0}
    agent.q_table[(1,)] = {"buy": 1000.0, "sell": 2.0, "hold": 1.0}
    assert agent.choose_action((0,), explore=False) == "buy"
    assert agent.choose_action((1,), explore=False) == "sell"
    with pytest.raises(ValueError, match="Unrecognized action"):
        agent.update_q_value((0,), "sell", 1.0, (1,))
    agent.update_q_value((0,), "buy", 0.0, (1,))
    assert agent.q_table[(0,)]["buy"] == pytest.approx(0.95 * 2.0)
    exploratory = QLearningAgent(epsilon=1.0)
    assert {exploratory.choose_action((0,)) for _ in range(50)} <= {"buy", "hold"}
    assert {exploratory.choose_action((1,)) for _ in range(50)} <= {"sell", "hold"}


def test_rl_terminal_update_does_not_bootstrap():
    agent = QLearningAgent(alpha=1, gamma=0.95)
    agent.q_table[(1,)] = {"buy": 100.0, "sell": 20.0, "hold": 1.0}
    agent.update_q_value((0,), "buy", 7.0, (1,), done=True)
    assert agent.q_table[(0,)]["buy"] == 7.0


def test_policy_evaluation_never_modifies_q_table():
    agent = QLearningAgent(epsilon=0.0)
    agent.q_table[(1, 2, 3, 4, 0)] = {"buy": 1.0, "sell": 0.0, "hold": 0.0}
    snapshot = copy.deepcopy(agent.q_table)
    unseen = (5, 5, 5, 5, 0)
    assert agent.choose_action(unseen, explore=False) == "hold"
    assert agent.q_table == snapshot
    env = SimpleTradingEnv(pd.DataFrame({"Open": [99, 100, 101], "Close": [100, 101, 102]}))
    agent.simulate_episode(env, train=False)
    assert agent.q_table == snapshot


def test_project_modules_support_package_imports():
    module = importlib.import_module("scripts.model_training")
    assert callable(module.run_experiment)
    preprocessing = importlib.import_module("scripts.data_processing")
    assert callable(preprocessing.load_data)


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
        document = nbformat.read(notebook, as_version=4)
        nbformat.validate(document)
        executable = [cell for cell in document.cells if cell.cell_type == "code"]
        assert executable and all(cell.execution_count is not None for cell in executable)
        assert all(cell.outputs for cell in executable[1:])
        for cell in executable:
            for output in cell.outputs:
                if "image/png" in output.get("data", {}):
                    assert base64.b64decode(output.data["image/png"], validate=True).startswith(b"\x89PNG")


def test_saved_experiment_notebooks_contain_visual_evidence():
    import nbformat
    visualization = nbformat.read(ROOT / "notebooks/HMM_and_RL_Experiments.ipynb", as_version=4)
    assert sum("image/png" in output.get("data", {})
               for cell in visualization.cells for output in cell.get("outputs", [])) >= 2
    evaluation = nbformat.read(ROOT / "notebooks/Model_Training_Evaluation.ipynb", as_version=4)
    assert any("accuracy" in output.get("text", "")
               for cell in evaluation.cells for output in cell.get("outputs", []))


def test_notebook_check_detects_stale_published_text():
    from scripts.refresh_notebooks import _verify_outputs

    expected = [dict(output_type="stream", name="stdout", text="Old result\n")]
    current = [dict(output_type="stream", name="stdout", text="New result\n")]
    with pytest.raises(RuntimeError, match="Published results differ"):
        _verify_outputs(expected, current, Path("example.ipynb"), 1)
    with pytest.raises(RuntimeError, match="Published chart count differs"):
        _verify_outputs([dict(output_type="display_data", data={"image/png": "unused"})],
                        [], Path("example.ipynb"), 1)
