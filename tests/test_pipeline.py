"""Regression tests for causality, data boundaries, inference, and simulation."""

from __future__ import annotations

import copy
import base64
import io
import importlib
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data_processing import DEFAULT_RAW_DATA, load_data, prepare_features, validate_bars  # noqa: E402
from date_validation import parse_daily_dates  # noqa: E402
from hmm_integration import (  # noqa: E402
    _gaussian_log_emissions, _validated_indicators,
    fit_hmm_on_indicators, infer_filtered_regimes,
)
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


def test_feature_preparation_rejects_history_shorter_than_indicator_warmup():
    with pytest.raises(ValueError, match="No usable feature observations remain"):
        prepare_features(synthetic_bars(25))


def test_feature_preparation_rejects_ticker_lost_during_warmup():
    bars = synthetic_bars()
    bbb_early_dates = bars.loc[bars["Name"] == "BBB", "Date"].nsmallest(25)
    mixed = bars.loc[(bars["Name"] == "AAA") |
                     ((bars["Name"] == "BBB") & bars["Date"].isin(bbb_early_dates))]
    with pytest.raises(ValueError, match=r"ticker\(s\) BBB"):
        prepare_features(mixed)


def test_validation_rejects_ticker_lost_to_invalid_bars(tmp_path: Path):
    bars = synthetic_bars()
    bars.loc[bars["Name"] == "BBB", "High"] = 0.0
    expected = r"No valid OHLCV observations remain for ticker\(s\) BBB"
    with pytest.raises(ValueError, match=expected):
        validate_bars(bars)
    with pytest.raises(ValueError, match=expected):
        prepare_features(bars)
    csv = tmp_path / "mixed_validity.csv"
    bars.to_csv(csv, index=False)
    with pytest.raises(ValueError, match=expected):
        load_data(csv)


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


def test_direct_indicator_api_rejects_missing_symbols_and_duplicate_dates():
    bars = synthetic_bars(3).loc[lambda frame: frame["Name"] == "AAA"].reset_index(drop=True)
    with pytest.raises(ValueError, match="ticker name"):
        add_technical_indicators(bars.assign(Name=["AAA", None, "AAA"]))
    with pytest.raises(ValueError, match="ticker name"):
        add_technical_indicators(bars.assign(Name=["AAA", " ", "AAA"]))
    with pytest.raises(ValueError, match="Duplicate ticker/date"):
        add_technical_indicators(bars.assign(Date=[bars.loc[0, "Date"]] * len(bars)))


def test_single_ticker_indicators_reject_duplicate_dates_without_name_column():
    bars = synthetic_bars(3).query("Name == 'AAA'").drop(columns="Name").reset_index(drop=True)
    bars.loc[1, "Date"] = bars.loc[0, "Date"]
    with pytest.raises(ValueError, match="Duplicate trading dates"):
        add_technical_indicators(bars)


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


def test_selected_ticker_ignores_unrelated_invalid_bars_and_dates(tmp_path: Path, monkeypatch):
    from data_processing import main as prepare_main

    bars = synthetic_bars()
    bars["Date"] = bars["Date"].astype(str)
    bars.loc[bars["Name"] == "BBB", "High"] = 0.0
    bars.loc[bars["Name"] == "BBB", "Date"] = "not-a-date"
    csv = tmp_path / "mixed.csv"
    bars.to_csv(csv, index=False)

    with pytest.raises(ValueError, match="parseable ISO calendar dates"):
        load_data(csv)
    selected = load_data(csv, tickers=(" aaa ",))
    assert len(selected) == 130 and set(selected["Name"]) == {"AAA"}
    experiment = run_experiment(csv, tickers=("aaa",), num_bins=3)
    assert experiment["tickers"] == 1
    assert experiment["tickers_per_partition"] == {
        "train": 1, "validation": 1, "test": 1,
    }

    output = tmp_path / "selected.csv"
    monkeypatch.setattr(sys, "argv", [
        "data_processing.py", "--input", str(csv), "--ticker", "AAA", "--output", str(output),
    ])
    prepare_main()
    assert set(pd.read_csv(output)["Name"]) == {"AAA"}

    with pytest.raises(ValueError, match="Ticker not found: MISSING"):
        load_data(csv, tickers=("MISSING",))
    with pytest.raises(ValueError, match="Selected tickers must be nonempty strings"):
        load_data(csv, tickers=())


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


@pytest.mark.parametrize("bad_date", [
    20200102, "today", "03/04/2020", "2020-01-02 12:00",
    pd.Timestamp("2020-01-02 12:00"), pd.Timestamp("2020-01-02", tz="UTC"),
])
def test_daily_calendar_dates_reject_ambiguous_or_non_daily_inputs(bad_date):
    with pytest.raises(ValueError, match="parseable ISO calendar dates"):
        parse_daily_dates(pd.Series([bad_date], dtype=object))


def test_daily_dates_accept_explicit_year_first_dates_and_date_objects():
    result = parse_daily_dates(pd.Series([
        "2020-1-2", "2020-01-03", date(2020, 1, 4), np.datetime64("2020-01-05"),
    ]))
    assert result.dt.strftime("%Y-%m-%d").tolist() == [
        "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05",
    ]


def test_all_daily_date_entry_points_reject_intraday_or_relative_dates():
    bars = pd.DataFrame({
        "Date": ["2020-01-02", "2020-01-02 12:00", "2020-01-03"],
        "Name": ["AAA"] * 3,
        "Open": [10., 10., 10.], "High": [12., 12., 12.],
        "Low": [9., 9., 9.], "Close": [11., 11., 11.],
        "Volume": [100, 100, 100],
    })
    with pytest.raises(ValueError, match="parseable ISO calendar dates"):
        validate_bars(bars)
    with pytest.raises(ValueError, match="parseable ISO calendar dates"):
        add_technical_indicators(bars)
    with pytest.raises(ValueError, match="parseable ISO calendar dates"):
        _validated_indicators(bars.assign(MA_20=[1., 2., 3.]), ("MA_20",))
    with pytest.raises(ValueError, match="parseable ISO calendar dates"):
        SimpleTradingEnv(bars)

    frame = prepare_features(synthetic_bars(70))
    frame["Target_Date"] = frame["Target_Date"].astype(object)
    frame.loc[frame.index[0], "Target_Date"] = "today"
    with pytest.raises(ValueError, match="parseable ISO calendar dates"):
        chronological_split(frame)


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


def test_chronological_split_rejects_missing_ticker_coverage():
    bars = synthetic_bars()
    early_bbb = bars.loc[bars["Name"] == "BBB", "Date"].nsmallest(55)
    short_bbb = bars.loc[(bars["Name"] == "AAA") |
                         ((bars["Name"] == "BBB") & bars["Date"].isin(early_bbb))]
    features = prepare_features(short_bbb)
    assert set(features["Name"]) == {"AAA", "BBB"}
    with pytest.raises(ValueError, match="Missing ticker coverage.*validation: BBB; test: BBB"):
        chronological_split(features)


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


def test_binner_refit_replaces_thresholds_atomically():
    binner = QuantileBinner(3).fit(pd.DataFrame({"A": [1, 2, 3, 4]}), ("A",))
    previous = binner.edges["A"].copy()
    with pytest.raises(ValueError, match="nonfinite"):
        binner.fit(pd.DataFrame({"B": [1., 2., 3., 4.], "C": [1., np.nan, 3., 4.]}), ("B", "C"))
    assert list(binner.edges) == ["A"]
    np.testing.assert_array_equal(binner.edges["A"], previous)
    binner.fit(pd.DataFrame({"B": [10, 20, 30, 40]}), ("B",))
    assert list(binner.edges) == ["B"]
    with pytest.raises(RuntimeError, match="not fitted"):
        binner.transform(pd.DataFrame({"A": [3]}), ("A",))


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


def test_hmm_notebook_keeps_causal_observations_purged_from_supervised_splits():
    import nbformat

    stock = prepare_features(synthetic_bars()).query("Name == 'AAA'").reset_index(drop=True)
    split = chronological_split(stock)
    labeled = pd.concat([split.train, split.validation, split.test], ignore_index=True)
    omitted_dates = stock.loc[~stock["Date"].isin(labeled["Date"]), "Date"]
    assert len(omitted_dates) == 2
    complete_training = stock.loc[stock["Date"] <= split.train_end]
    assert len(complete_training) == len(split.train) + 1

    notebook = nbformat.read(ROOT / "notebooks/HMM_and_RL_Experiments.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert "hmm_training = stock.loc[stock['Date'] <= split.train_end]" in source
    assert "fit_hmm_on_indicators(hmm_training" in source
    assert "infer_filtered_regimes(hmm_model, stock)" in source
    assert "train_regimes = history_regimes.loc[history_regimes['Date'].isin(split.train['Date'])]" in source
    assert "infer_filtered_regimes(hmm_model, history)" not in source


def test_gaussian_log_emissions_use_public_model_parameters():
    from scipy.stats import norm

    class GaussianParameters:
        covariance_type = "diag"
        means_ = np.array([[1., 2.], [4., 6.]])
        covars_ = np.array([np.diag([1., 4.]), np.diag([9., 16.])])

    observations = np.array([[1., 3.], [5., 8.]])
    expected = np.column_stack([
        norm.logpdf(observations, loc=GaussianParameters.means_[state],
                    scale=np.sqrt(np.diag(GaussianParameters.covars_[state]))).sum(axis=1)
        for state in range(2)
    ])
    np.testing.assert_allclose(
        _gaussian_log_emissions(GaussianParameters(), observations), expected, atol=1e-12
    )


def test_hmm_filtering_respects_impossible_initial_states_and_transitions():
    class DeterministicModel:
        covariance_type = "diag"
        means_ = np.array([[0.], [40.]])
        covars_ = np.array([[[1.]], [[1.]]])
        startprob_ = np.array([1., 0.])
        transmat_ = np.eye(2)

    # State 1 has an overwhelmingly better emission likelihood, but both
    # its initial probability and its transition from state 0 are zero.
    frame = pd.DataFrame({
        "Date": ["2020-01-02", "2020-01-03"],
        "MA_20": [40., 40.],
    })
    filtered = infer_filtered_regimes(DeterministicModel(), frame, ("MA_20",))
    assert filtered["Market_Regime"].tolist() == [0, 0]


def test_hmm_filtering_advances_through_missing_indicator_observations():
    class AlternatingModel:
        covariance_type = "diag"
        means_ = np.array([[0.], [0.]])
        covars_ = np.array([[[1.]], [[1.]]])
        startprob_ = np.array([0.99, 0.01])
        transmat_ = np.array([[0.1, 0.9], [0.9, 0.1]])

    frame = pd.DataFrame({
        "Date": ["2020-01-02", "2020-01-03", "2020-01-06"],
        "MA_20": [0., np.nan, 0.],
    })
    filtered = infer_filtered_regimes(AlternatingModel(), frame, ("MA_20",))
    assert filtered.loc[0, "Market_Regime"] == 0
    assert pd.isna(filtered.loc[1, "Market_Regime"])
    # Two transitions from the first day's prior yield [0.8136, 0.1864].
    assert filtered.loc[2, "Market_Regime"] == 0


def test_hmm_filtering_preserves_duplicate_input_index_labels():
    frame = prepare_features(synthetic_bars()).query("Name == 'AAA'").reset_index(drop=True)
    model, _ = fit_hmm_on_indicators(frame.iloc[:65], n_components=2)
    repeated = pd.concat([
        frame.iloc[:50].reset_index(drop=True),
        frame.iloc[50:].reset_index(drop=True),
    ]).sample(frac=1, random_state=13)
    assert repeated.index.has_duplicates

    actual = infer_filtered_regimes(model, repeated)
    expected = infer_filtered_regimes(model, repeated.reset_index(drop=True))
    assert actual.index.equals(repeated.index)
    assert actual["Market_Regime"].tolist() == expected["Market_Regime"].tolist()

    missing = repeated.copy()
    missing.iloc[4, missing.columns.get_loc("MA_20")] = np.nan
    filtered = infer_filtered_regimes(model, missing)
    comparison = infer_filtered_regimes(model, missing.reset_index(drop=True))
    assert filtered.index.equals(missing.index)
    pd.testing.assert_series_equal(
        filtered["Market_Regime"].reset_index(drop=True), comparison["Market_Regime"]
    )

    missing["MA_20"] = np.nan
    all_missing = infer_filtered_regimes(model, missing)
    assert all_missing.index.equals(missing.index)
    assert all_missing["Market_Regime"].isna().all()


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
    train_end = chronological_split(stock).train_end
    model, _ = fit_hmm_on_indicators(stock.loc[stock["Date"] <= train_end], n_components=3)
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


def test_terminal_open_position_is_marked_to_market_without_forced_sale():
    env = SimpleTradingEnv(pd.DataFrame({
        "Open": [10., 11., 12.], "Close": [10., 11., 14.],
    }), transaction_cost=0.01)
    _, first_reward, done, _ = env.step("buy")
    assert not done
    _, final_reward, done, info = env.step("hold")
    assert done and info["position"] == 1 and info["fee"] == 0
    assert first_reward + final_reward == pytest.approx(14 - 11 * 1.01)


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


def test_trading_rejects_incomplete_ticker_identity():
    bars = pd.DataFrame({
        "Date": pd.bdate_range("2020-01-01", periods=3),
        "Name": ["AAA", None, "AAA"],
        "Open": [10., 11., 12.], "Close": [11., 12., 13.],
    })
    with pytest.raises(ValueError, match="one ticker"):
        SimpleTradingEnv(bars)
    with pytest.raises(ValueError, match="one ticker"):
        SimpleTradingEnv(bars.assign(Name=["AAA", "", "AAA"]))


def test_hmm_rejects_incomplete_ticker_identity():
    from hmm_integration import _validated_indicators
    frame = pd.DataFrame({
        "Date": pd.bdate_range("2020-01-01", periods=3),
        "Name": ["AAA", None, "AAA"],
        "MA_20": [1., 2., 3.],
    })
    with pytest.raises(ValueError, match="one ticker"):
        _validated_indicators(frame, ("MA_20",))
    with pytest.raises(ValueError, match="one ticker"):
        _validated_indicators(frame.assign(Name=["AAA", " ", "AAA"]), ("MA_20",))


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


@pytest.mark.parametrize("invalid", [
    (1.2, 0), (1.9, 0), (1.0, 0), ("1", 0), (True, 0),
    (np.bool_(True), 0), (np.inf, 0), (None, 0),
])
def test_qlearning_state_rejects_noninteger_components_atomically(invalid):
    agent = QLearningAgent()
    with pytest.raises(ValueError, match="State components must be integers"):
        agent.get_state(invalid)
    with pytest.raises(ValueError, match="State components must be integers"):
        agent.choose_action(invalid)
    with pytest.raises(ValueError, match="State components must be integers"):
        agent.update_q_value((0, 0), "hold", 0.0, invalid)
    assert agent.q_table == {}


def test_qlearning_state_accepts_numpy_integer_components():
    assert QLearningAgent().get_state((np.int64(3), np.int64(0))) == (3, 0)


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


def test_rejected_q_update_is_atomic_and_cannot_change_frozen_policy():
    agent = QLearningAgent(epsilon=0)
    assert agent.choose_action((0,), explore=False) == "hold"
    with pytest.raises(ValueError, match="Unrecognized action"):
        agent.update_q_value((0,), "sell", 1.0, (0,))
    assert agent.q_table == {}
    assert agent.choose_action((0,), explore=False) == "hold"
    with pytest.raises(ValueError, match="current position"):
        agent.update_q_value((0,), "buy", 1.0, (2,))
    assert agent.q_table == {}
    with pytest.raises(ValueError, match="Reward must be finite"):
        agent.update_q_value((0,), "buy", float("nan"), (1,))
    assert agent.q_table == {}


def test_rl_terminal_update_does_not_bootstrap():
    agent = QLearningAgent(alpha=1, gamma=0.95)
    agent.q_table[(1,)] = {"buy": 100.0, "sell": 20.0, "hold": 1.0}
    agent.update_q_value((0,), "buy", 7.0, (1,), done=True)
    assert agent.q_table[(0,)]["buy"] == 7.0


@pytest.mark.parametrize("invalid", [-3, -1, 1.5, True, np.bool_(True), "2"])
def test_simulate_episode_rejects_invalid_step_limit_without_resetting(invalid):
    env = SimpleTradingEnv(pd.DataFrame({
        "Open": [10., 11., 12.], "Close": [10., 11., 12.],
    }))
    env.step("hold")
    previous_state = (env.index, env.position, env.done)
    agent = QLearningAgent()
    with pytest.raises(ValueError, match="max_steps must be a nonnegative integer"):
        agent.simulate_episode(env, max_steps=invalid)
    assert (env.index, env.position, env.done) == previous_state


def test_simulate_episode_allows_zero_and_numpy_integer_step_limits():
    bars = pd.DataFrame({"Open": [10., 11., 12.], "Close": [10., 11., 12.]})
    env = SimpleTradingEnv(bars)
    agent = QLearningAgent(epsilon=0)
    assert agent.simulate_episode(env, train=False, max_steps=0) == 0.0
    assert env.index == 0
    assert agent.simulate_episode(env, train=False, max_steps=np.int64(1)) == 0.0
    assert env.index == 1


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
    assert output["tickers_per_partition"] == {"train": 2, "validation": 2, "test": 2}
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


def test_readme_trading_result_matches_the_published_notebook():
    import nbformat

    notebook = nbformat.read(ROOT / "notebooks/HMM_and_RL_Experiments.ipynb", as_version=4)
    lines = "\n".join(str(output.get("text", "")) for cell in notebook.cells
                      for output in cell.get("outputs", [])).splitlines()
    prefix = "Frozen policy holdout P/L, one-share dollar units: "
    result = next(line for line in lines if line.startswith(prefix))
    pnl = float(result[len(prefix):])
    displayed = f"**{'+' if pnl >= 0 else '-'}${abs(pnl):.2f}**"
    assert displayed in (ROOT / "README.md").read_text(encoding="utf-8")


def test_notebook_check_detects_stale_published_text():
    from scripts.refresh_notebooks import _verify_outputs

    expected = [dict(output_type="stream", name="stdout", text="Old result\n")]
    current = [dict(output_type="stream", name="stdout", text="New result\n")]
    with pytest.raises(RuntimeError, match="Published results differ"):
        _verify_outputs(expected, current, Path("example.ipynb"), 1)
    with pytest.raises(RuntimeError, match="Published chart count differs"):
        _verify_outputs([dict(output_type="display_data", data={"image/png": "unused"})],
                        [], Path("example.ipynb"), 1)


def test_programmatic_notebook_verification_cannot_overwrite_files(tmp_path: Path):
    import nbformat
    from scripts.refresh_notebooks import reproduce_notebook

    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        source="print('verified')", execution_count=1,
        outputs=[nbformat.v4.new_output("stream", name="stdout", text="verified\n")],
    )])
    path = tmp_path / "check.ipynb"
    nbformat.write(notebook, path)
    original = path.read_bytes()

    reproduce_notebook(path, check=True)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="cannot save or overwrite"):
        reproduce_notebook(path, check=True, save=True)


def test_programmatic_notebook_execution_resolves_relative_paths_and_restores_cwd(tmp_path: Path):
    import nbformat
    from scripts.refresh_notebooks import reproduce_notebook

    original_cwd = Path.cwd()
    notebook_dir = tmp_path / "notebooks"
    notebook_dir.mkdir()
    (notebook_dir / "input.txt").write_text("verified", encoding="utf-8")
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        source="from pathlib import Path\nprint(Path('input.txt').read_text())",
        execution_count=1,
        outputs=[nbformat.v4.new_output("stream", name="stdout", text="verified\n")],
    )])
    path = notebook_dir / "relative.ipynb"
    nbformat.write(notebook, path)
    reproduce_notebook(path, check=True)
    assert Path.cwd() == original_cwd

    failed_path = notebook_dir / "failure.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        source="raise RuntimeError('expected failure')",
    )]), failed_path)
    with pytest.raises(RuntimeError, match="Notebook failure.ipynb, cell 1"):
        reproduce_notebook(failed_path, check=True)
    assert Path.cwd() == original_cwd


def test_notebook_failures_clean_own_figures_without_touching_caller_figures(tmp_path: Path):
    import matplotlib.pyplot as plt
    import nbformat
    from scripts.refresh_notebooks import reproduce_notebook

    initial_figures = set(plt.get_fignums())
    caller_figure = plt.figure()
    try:
        failed = tmp_path / "raises_after_plot.ipynb"
        nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
            "import matplotlib.pyplot as plt\nplt.figure()\nraise RuntimeError('failure')",
        )]), failed)
        with pytest.raises(RuntimeError, match="Notebook raises_after_plot.ipynb, cell 1"):
            reproduce_notebook(failed, check=True)
        assert set(plt.get_fignums()) == initial_figures | {caller_figure.number}

        succeeding = tmp_path / "still_valid.ipynb"
        nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
            source="print('untouched')", execution_count=1,
            outputs=[nbformat.v4.new_output("stream", name="stdout", text="untouched\n")],
        )]), succeeding)
        reproduce_notebook(succeeding, check=True)
        assert set(plt.get_fignums()) == initial_figures | {caller_figure.number}
    finally:
        plt.close(caller_figure)
    assert set(plt.get_fignums()) == initial_figures


def test_notebook_check_rejects_corrupted_or_materially_changed_pngs():
    from PIL import Image
    from scripts.refresh_notebooks import _verify_outputs

    def png_output(color):
        buffer = io.BytesIO()
        Image.new("RGB", (100, 60), color=color).save(buffer, format="PNG")
        return {"output_type": "display_data", "data": {
            "image/png": base64.b64encode(buffer.getvalue()).decode("ascii")
        }}

    white = png_output("white")
    black = png_output("black")
    _verify_outputs([white], [white], Path("example.ipynb"), 1)
    with pytest.raises(RuntimeError, match="Published chart differs"):
        _verify_outputs([white], [black], Path("example.ipynb"), 1)

    corrupt = {"output_type": "display_data", "data": {
        "image/png": base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii")
    }}
    with pytest.raises(RuntimeError, match="Invalid published chart"):
        _verify_outputs([white], [corrupt], Path("example.ipynb"), 1)


def test_notebook_check_catches_moved_thin_plot_lines():
    from PIL import Image, ImageDraw
    from scripts.refresh_notebooks import _verify_outputs

    def line_chart(y_position):
        chart = Image.new("RGB", (1000, 750), "white")
        ImageDraw.Draw(chart).line((30, y_position, 970, y_position),
                                   fill="black", width=3)
        buffer = io.BytesIO()
        chart.save(buffer, format="PNG")
        return {"output_type": "display_data", "data": {
            "image/png": base64.b64encode(buffer.getvalue()).decode("ascii")
        }}

    with pytest.raises(RuntimeError, match="Published chart differs"):
        _verify_outputs([line_chart(200)], [line_chart(500)], Path("chart.ipynb"), 2)


def test_notebook_check_catches_changed_rich_text_results():
    from scripts.refresh_notebooks import _verify_outputs

    before = [{"output_type": "execute_result", "data": {"text/plain": "accuracy: 0.70"}}]
    after = [{"output_type": "execute_result", "data": {"text/plain": "accuracy: 0.10"}}]
    with pytest.raises(RuntimeError, match="Published results differ"):
        _verify_outputs(before, after, Path("results.ipynb"), 2)
