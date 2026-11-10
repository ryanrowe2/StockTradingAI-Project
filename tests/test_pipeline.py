"""Regression tests for causality, data boundaries, inference, and simulation."""

from __future__ import annotations

import copy
import base64
import io
import importlib
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Make CPU-count probing deterministic in sandboxed CI/test environments.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data_processing import (  # noqa: E402
    DEFAULT_RAW_DATA, load_data, prepare_features, prepare_indicator_history, validate_bars,
)
from date_validation import parse_daily_dates  # noqa: E402
from hmm_integration import (  # noqa: E402
    _gaussian_log_emissions, _validated_indicators,
    fit_hmm_on_indicators, infer_filtered_regimes,
)
from model_training import (  # noqa: E402
    BASELINE_FEATURES, QuantileBinner, chronological_split,
    classification_metrics, fit_network, predict_up_probability, run_experiment,
)
from rl_agent import QLearningAgent, SimpleTradingEnv, contiguous_trading_segments  # noqa: E402
from session_validation import adjacent_session_flags  # noqa: E402
from technical_indicators import add_technical_indicators  # noqa: E402
from walk_forward_evaluation import (  # noqa: E402
    paired_block_accuracy_intervals, walk_forward_experiment,
)


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
    edited.loc[(edited["Name"] == "AAA") & (edited["Date"] == latest), "High"] += 1000
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


def test_direct_indicator_api_normalizes_tickers_before_grouping():
    bars = synthetic_bars(80).query("Name == 'AAA'").sort_values("Date").reset_index(drop=True)
    variants = bars.copy()
    variants.loc[40:, "Name"] = " aaa "
    result = add_technical_indicators(variants)
    canonical = add_technical_indicators(bars)
    assert set(result["Name"]) == {"AAA"}
    pd.testing.assert_frame_equal(result, canonical)
    duplicate = pd.concat([bars.iloc[:1], bars.iloc[:1].assign(Name=" aaa ")])
    with pytest.raises(ValueError, match="Duplicate ticker/date"):
        add_technical_indicators(duplicate)


def test_direct_indicator_api_rejects_empty_frame():
    with pytest.raises(ValueError, match="nonempty indicator DataFrame"):
        add_technical_indicators(pd.DataFrame(columns=["Name", "Date", "Close"]))


def test_direct_indicator_api_rejects_incoherent_and_nonfinite_prices():
    bars = synthetic_bars(50).query("Name == 'AAA'").reset_index(drop=True)
    for invalid in (
        bars.assign(High=bars["Low"] - 1),
        bars.assign(Low=bars["High"] + 1),
        bars.assign(Close=np.inf),
        bars.assign(Open=0.0),
    ):
        with pytest.raises(ValueError, match="Indicator OHLC prices"):
            add_technical_indicators(invalid)
    with pytest.raises(ValueError, match="Volume must be nonnegative"):
        add_technical_indicators(bars.assign(Volume=-1))


def test_direct_indicator_api_normalizes_numeric_strings():
    bars = synthetic_bars(60).query("Name == 'AAA'").reset_index(drop=True)
    string_prices = bars.copy()
    for field in ("Open", "High", "Low", "Close", "Volume"):
        string_prices[field] = string_prices[field].map(str)
    expected = add_technical_indicators(bars)
    actual = add_technical_indicators(string_prices)
    pd.testing.assert_frame_equal(actual, expected)


def test_direct_indicator_api_rejects_inconsistent_market_positions():
    bars = synthetic_bars(3).query("Name == 'AAA'").reset_index(drop=True)
    with pytest.raises(ValueError, match="Session_Number"):
        add_technical_indicators(bars.assign(Session_Number=[0, 0, 1]))
    with pytest.raises(ValueError, match="Session_Number"):
        add_technical_indicators(bars.assign(Session_Number=[2, 1, 0]))


def test_missing_raw_market_session_is_not_used_as_a_daily_return_or_rsi():
    bars = synthetic_bars(100).query("Name == 'AAA'").sort_values("Date").reset_index(drop=True)
    bad_day = bars.loc[45, "Date"]
    next_day = bars.loc[46, "Date"]
    bars.loc[45, "High"] = 0.0
    clean = validate_bars(bars)
    assert bad_day not in set(clean["Date"])
    before = clean.loc[clean["Date"] < bad_day, "Session_Number"].iloc[-1]
    after = clean.loc[clean["Date"] == next_day, "Session_Number"].iloc[0]
    assert after - before == 2
    history = prepare_indicator_history(clean)
    crossing = history.loc[history["Date"] == next_day].iloc[0]
    assert pd.isna(crossing["Return"])
    assert pd.isna(crossing["RSI_14"])
    assert pd.isna(crossing["ATR"])
    assert crossing["Close_Lag1"] == pytest.approx(bars.loc[44, "Close"])
    # After the warm-up, each daily return is again calculated from the true
    # immediately preceding market session, not from the next available bar.
    resumed = history.loc[history["Date"] == bars.loc[60, "Date"], "RSI_14"].iloc[0]
    assert np.isfinite(resumed)
    prepared = prepare_features(clean)
    # A target must be the next source-market session, not merely the next
    # available valid close after the discarded price bar.
    assert bars.loc[44, "Date"] not in set(prepared["Date"])
    assert next_day not in set(prepared["Date"])
    assert bars.iloc[-1]["Date"] == history.iloc[-1]["Date"]
    assert bars.iloc[-1]["Date"] not in set(prepared["Date"])


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


def test_selected_ticker_ignores_unrelated_invalid_bars_and_dates(tmp_path: Path, monkeypatch, capsys):
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
    assert "from 1 ticker:" in capsys.readouterr().out
    assert set(pd.read_csv(output)["Name"]) == {"AAA"}

    with pytest.raises(ValueError, match="Ticker not found: MISSING"):
        load_data(csv, tickers=("MISSING",))
    with pytest.raises(ValueError, match="Selected tickers must be nonempty strings"):
        load_data(csv, tickers=())


def test_selected_ticker_retains_source_wide_market_calendar(tmp_path: Path):
    source = synthetic_bars(100)
    missing = source.query("Name == 'AAA'")["Date"].sort_values().iloc[55]
    source = source.loc[~((source.Name == "AAA") & (source.Date == missing))].copy()
    csv = tmp_path / "source.csv"
    source.to_csv(csv, index=False)
    complete = load_data(csv)
    selected = load_data(csv, tickers=("AAA",))
    pd.testing.assert_frame_equal(
        complete.loc[complete.Name == "AAA"].reset_index(drop=True), selected,
    )
    full_history = prepare_indicator_history(complete.loc[complete.Name == "AAA"])
    selected_history = prepare_indicator_history(selected)
    pd.testing.assert_frame_equal(full_history, selected_history)
    next_date = source.query("Name == 'AAA' and Date > @missing")["Date"].min()
    assert pd.isna(selected_history.loc[selected_history.Date == next_date, "Return"].iloc[0])


def test_historical_aapl_subset_and_full_source_have_identical_features():
    complete = load_data(DEFAULT_RAW_DATA)
    selected = load_data(DEFAULT_RAW_DATA, tickers=("AAPL",))
    full_aapl = complete.loc[complete.Name == "AAPL"].reset_index(drop=True)
    pd.testing.assert_frame_equal(full_aapl, selected)
    pd.testing.assert_frame_equal(prepare_indicator_history(full_aapl),
                                  prepare_indicator_history(selected))
    pd.testing.assert_frame_equal(prepare_features(full_aapl), prepare_features(selected))


def test_every_bundled_individual_stock_file_uses_combined_market_calendar():
    complete = load_data(DEFAULT_RAW_DATA)
    for file in sorted(DEFAULT_RAW_DATA.parent.glob("*_2006-01-01_to_2018-01-01.csv")):
        if file == DEFAULT_RAW_DATA:
            continue
        symbol = file.name.split("_", 1)[0]
        individual = load_data(file)
        reference = complete.loc[complete.Name == symbol].reset_index(drop=True)
        pd.testing.assert_frame_equal(individual, reference)
        pd.testing.assert_frame_equal(
            prepare_features(individual), prepare_features(reference),
        )


def test_invalid_unrelated_weekend_does_not_shift_selected_stock_sessions(tmp_path: Path):
    source = synthetic_bars(100)
    clean_csv = tmp_path / "clean.csv"
    source.to_csv(clean_csv, index=False)
    expected = load_data(clean_csv, tickers=("AAA",))
    corrupted = pd.concat([source, source.iloc[:1].assign(
        Name="BBB", Date=pd.Timestamp("2019-01-05"), High=-10.0,
    )], ignore_index=True)
    corrupt_csv = tmp_path / "corrupted.csv"
    corrupted.to_csv(corrupt_csv, index=False)
    actual = load_data(corrupt_csv, tickers=("AAA",))
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(prepare_features(actual), prepare_features(expected))


def test_model_specific_feature_warmup_does_not_require_unused_indicators():
    clean = load_data(DEFAULT_RAW_DATA)
    assert len(prepare_features(clean)) == 92_284
    assert len(prepare_features(clean, features=BASELINE_FEATURES)) == 93_028
    default = prepare_features(clean)
    assert default[list(BASELINE_FEATURES)].notna().all().all()
    assert default["MACD_Pct"].notna().all()
    with pytest.raises(ValueError, match="Model features"):
        prepare_features(clean, features=("Not_A_Model_Feature",))


def test_validate_bars_rejects_conflicting_source_session_calendar():
    bars = synthetic_bars(3)
    dates = sorted(bars.Date.unique())
    bars["Session_Number"] = bars.Date.map({date: i for i, date in enumerate(dates)})
    collision = bars.copy()
    collision.loc[collision.Date == dates[2], "Session_Number"] = 1
    with pytest.raises(ValueError, match="Session_Number"):
        validate_bars(collision)
    descending = bars.copy()
    descending["Session_Number"] = descending.Session_Number.map({0: 2, 1: 1, 2: 0})
    with pytest.raises(ValueError, match="Session_Number"):
        validate_bars(descending)


def test_historical_supervised_targets_are_next_market_session_only():
    raw = load_data(DEFAULT_RAW_DATA)
    prepared = prepare_features(raw)
    calendar = raw[["Name", "Date", "Session_Number"]].rename(
        columns={"Date": "Target_Date", "Session_Number": "Target_Session"},
    )
    checked = prepared.merge(calendar, on=["Name", "Target_Date"], validate="many_to_one")
    assert (checked.Target_Session - checked.Session_Number).eq(1).all()


def test_csv_loader_preserves_literal_ticker_na():
    source = "Date,Name,Open,High,Low,Close,Volume\n2020-01-02,NA,10,11,9,10,100\n"
    for selected in (None, ("NA",), (" na ",)):
        kwargs = {} if selected is None else {"tickers": selected}
        loaded = load_data(io.StringIO(source), **kwargs)
        assert loaded["Name"].tolist() == ["NA"]
    with pytest.raises(ValueError, match="ticker symbol"):
        load_data(io.StringIO(source.replace(",NA,", ",,")))
    for marker in ("N/A", "NaN", "NULL", "None"):
        with pytest.raises(ValueError, match="ticker symbol"):
            load_data(io.StringIO(source.replace(",NA,", f",{marker},")))


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


def test_chronological_split_rejects_nonfuture_target_dates():
    frame = prepare_features(synthetic_bars())
    for bad_target in (frame.loc[0, "Date"], frame.loc[0, "Date"] - pd.Timedelta(days=1)):
        edited = frame.copy()
        edited.loc[0, "Target_Date"] = bad_target
        with pytest.raises(ValueError, match="Target_Date must follow"):
            chronological_split(edited)


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
    assert "hmm_training = history.loc[history['Date'] <= split.train_end]" in source
    assert "fit_hmm_on_indicators(hmm_training" in source
    assert "infer_filtered_regimes(hmm_model, history)" in source
    assert "train_regimes = history_regimes.loc[history_regimes['Date'].isin(split.train.loc[split.train['Name'] == 'AAPL', 'Date'])" in source
    assert "history_regimes[list((*hmm_features, 'Return'))].notna().all(axis=1)" in source
    assert "indicator_history = prepare_indicator_history" in source
    assert "split = chronological_split(features)" in source
    assert "history_regimes['Date'] > split.validation_end" in source
    assert "training_segments = contiguous_trading_segments(training)" in source
    assert "testing_segments = contiguous_trading_segments(testing)" in source
    assert "train_envs = [SimpleTradingEnv(group" in source
    assert "xlabel='Training pass'" in source


def test_hmm_gap_filtering_restarts_segments_with_and_without_missing_rows():
    class AlternatingModel:
        covariance_type = "diag"
        means_ = np.array([[0.], [0.]])
        covars_ = np.array([[[1.]], [[1.]]])
        startprob_ = np.array([1., 0.])
        transmat_ = np.array([[0., 1.], [1., 0.]])

    compressed = pd.DataFrame({
        "Date": ["2020-01-02", "2020-01-07"],
        "Session_Number": [0, 3], "MA_20": [0., 0.],
    })
    expanded = pd.DataFrame({
        "Date": ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"],
        "Session_Number": [0, 1, 2, 3], "MA_20": [0., np.nan, np.nan, 0.],
    })
    actual = infer_filtered_regimes(AlternatingModel(), compressed, ("MA_20",))
    expected = infer_filtered_regimes(AlternatingModel(), expanded, ("MA_20",))
    assert actual["Market_Regime"].tolist() == [0, 0]
    assert actual["Market_Regime"].tolist() == expected["Market_Regime"].dropna().tolist()


def test_hmm_training_never_learns_a_transition_across_discarded_market_sessions(monkeypatch):
    source = synthetic_bars(100).query("Name == 'AAA'").sort_values("Date").reset_index(drop=True)
    source.loc[60, "High"] = 0.0
    history = prepare_indicator_history(source)
    frame = history.dropna(subset=["MA_20", "RSI_14", "MACD"])
    from hmmlearn.hmm import GaussianHMM

    original_fit = GaussianHMM.fit
    recorded = []

    def recording_fit(self, values, lengths=None):
        recorded.append(lengths)
        return original_fit(self, values, lengths=lengths)

    monkeypatch.setattr(GaussianHMM, "fit", recording_fit)
    model, filtered = fit_hmm_on_indicators(frame, n_components=2)
    assert sum(recorded[0]) == len(frame)
    assert len(recorded[0]) >= 2
    assert filtered["Market_Regime"].notna().all()
    assert model.stock_trading_indicator_names_ == ("MA_20", "RSI_14", "MACD")


def test_hmm_fitting_rejects_history_without_consecutive_market_sessions():
    data = prepare_features(synthetic_bars()).query("Name == 'AAA'").iloc[:60].copy()
    data["Session_Number"] = np.arange(len(data)) * 2
    with pytest.raises(ValueError, match="Insufficient consecutive-session transitions"):
        fit_hmm_on_indicators(data, n_components=2)


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


def test_gaussian_log_emissions_avoid_finite_intermediate_square_overflow():
    class LargeVarianceModel:
        covariance_type = "diag"
        means_ = np.array([[0.]])
        covars_ = np.array([[[1e300]]])

    density = _gaussian_log_emissions(LargeVarianceModel(), np.array([[1e200]]))
    assert np.isfinite(density).all()
    assert density[0, 0] == pytest.approx(-0.5e100)


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


def test_hmm_filtering_restarts_after_missing_indicator_observations():
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
    # The missing emission splits the fitted/inferred sequences consistently.
    assert filtered.loc[2, "Market_Regime"] == 0


def test_hmm_rejects_descending_unsigned_market_sessions():
    frame = pd.DataFrame({
        "Date": ["2020-01-02", "2020-01-03"],
        "Session_Number": pd.Series([2, 1], dtype="uint64"),
        "MA_20": [1.0, 2.0],
    })
    with pytest.raises(ValueError, match="Session_Number"):
        _validated_indicators(frame, ("MA_20",))


def test_hmm_large_gap_restarts_in_constant_time():
    class SimpleModel:
        covariance_type = "diag"
        means_ = np.array([[0.], [0.]])
        covars_ = np.array([[[1.]], [[1.]]])
        startprob_ = np.array([1., 0.])
        transmat_ = np.array([[0., 1.], [1., 0.]])

    frame = pd.DataFrame({
        "Date": ["2020-01-02", "2020-01-03"],
        "Session_Number": pd.Series([0, 10**12], dtype="uint64"),
        "MA_20": [0., 0.],
    })
    assert infer_filtered_regimes(SimpleModel(), frame, ("MA_20",))["Market_Regime"].tolist() == [0, 0]


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


def test_hmm_training_rejects_missing_observations_instead_of_compressing_time():
    frame = prepare_features(synthetic_bars()).query("Name == 'AAA'").reset_index(drop=True)
    frame.loc[42, "MA_20"] = np.nan
    with pytest.raises(ValueError, match="HMM fitting requires complete indicator observations"):
        fit_hmm_on_indicators(frame.iloc[:70], n_components=2)


def test_historical_hmm_fit_reaches_configured_tolerance():
    data = load_data(DEFAULT_RAW_DATA)
    stock = prepare_features(data.loc[data["Name"] == "AAPL"])
    train_end = chronological_split(stock).train_end
    complete_indicators = stock.loc[stock["Date"] <= train_end].dropna(
        subset=["MA_20", "RSI_14", "MACD"]
    )
    model, _ = fit_hmm_on_indicators(complete_indicators, n_components=3)
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


def test_trading_environment_rejects_skipped_market_sessions():
    bars = pd.DataFrame({
        "Date": ["2020-01-02", "2020-01-06", "2020-01-07"],
        "Session_Number": [0, 2, 3],
        "Open": [10.0, 12.0, 13.0],
        "Close": [11.0, 12.5, 13.5],
    })
    with pytest.raises(ValueError, match="consecutive market sessions"):
        SimpleTradingEnv(bars)
    # Each contiguous slice remains a valid two-step simulation.
    assert len(SimpleTradingEnv(bars.iloc[1:]).data) == 2


def test_default_trading_state_uses_published_feature_schema():
    bars = pd.DataFrame({
        "Open": [99., 100., 101.], "Close": [100., 101., 102.],
        "Return_binned": [0, 1, 2], "RSI_14_binned": [2, 1, 0],
        "Market_Regime": [1, 0, 1],
    })
    env = SimpleTradingEnv(bars)
    assert env.state_columns == ("Return_binned", "RSI_14_binned", "Market_Regime")
    assert env.get_state() == (0, 2, 1, 0)
    with pytest.raises(ValueError, match="default state columns"):
        SimpleTradingEnv(bars.drop(columns="Market_Regime"))


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


@pytest.mark.parametrize("invalid_states", [
    [True, False, True], ["1", "0", "1"], ["1.0", "0.0", "1.0"],
    [1.0, 0.0, 1.0],
])
def test_trading_state_validation_requires_actual_integers(invalid_states):
    bars = pd.DataFrame({"Open": [10., 11., 12.], "Close": [11., 12., 13.],
                         "Bin": invalid_states})
    with pytest.raises(ValueError, match="finite integer states"):
        SimpleTradingEnv(bars, state_columns=("Bin",))
    bars["Bin"] = pd.Series([1, 0, 1], dtype="Int64")
    assert SimpleTradingEnv(bars, state_columns=("Bin",)).get_state() == (1, 0)


def test_trading_sorts_parsed_dates_and_rejects_invalid_or_duplicate_dates():
    bars = pd.DataFrame({
        "Date": ["2020-1-10", "2020-1-2", "2020-1-3"],
        "Open": [110.0, 90.0, 100.0], "Close": [115.0, 95.0, 105.0],
    })
    original_dates = bars["Date"].tolist()
    with pytest.raises(ValueError, match="consecutive market sessions"):
        SimpleTradingEnv(bars)
    contiguous = bars.assign(Date=["2020-1-6", "2020-1-2", "2020-1-3"])
    env = SimpleTradingEnv(contiguous)
    assert env.data["Date"].dt.day.tolist() == [2, 3, 6]
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


def test_terminal_next_state_must_have_valid_position_and_boolean_done():
    agent = QLearningAgent()
    with pytest.raises(ValueError, match="current position"):
        agent.update_q_value((0,), "buy", 1.0, (2,), done=True)
    with pytest.raises(ValueError, match="done must be a boolean"):
        agent.update_q_value((0,), "buy", 1.0, (1,), done="False")
    assert agent.q_table == {}
    agent.update_q_value((0,), "buy", 1.0, (1,), done=np.bool_(True))
    assert agent.q_table[(0,)]["buy"] == pytest.approx(0.1)


def test_overflowing_q_update_is_rejected_without_mutating_table():
    agent = QLearningAgent(alpha=1.0, gamma=1.0)
    agent.q_table[(1,)] = {"buy": 0.0, "sell": 1e308, "hold": 0.0}
    before = copy.deepcopy(agent.q_table)
    with pytest.raises(ValueError, match="Q-value update must remain finite"):
        agent.update_q_value((0,), "buy", 1e308, (1,))
    assert agent.q_table == before


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
    walk_forward = importlib.import_module("scripts.walk_forward_evaluation")
    assert callable(walk_forward.walk_forward_experiment)


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

    hold_prefix = "Next-open buy-and-hold P/L, same units: "
    buy_and_hold = float(next(line for line in lines if line.startswith(hold_prefix))[len(hold_prefix):])
    displayed_hold = f"**{'+' if buy_and_hold >= 0 else '-'}${abs(buy_and_hold):.2f}**"
    assert displayed_hold in (ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_bayesian_accuracy_matches_saved_evaluation():
    import nbformat
    import re

    notebook = nbformat.read(ROOT / "notebooks/Model_Training_Evaluation.ipynb", as_version=4)
    content = "\n".join(str(output.get("data", {}).get("text/plain", ""))
                        for cell in notebook.cells for output in cell.get("outputs", []))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in ("baseline", "enhanced"):
        match = re.search(rf"^{name}\s+(0\.\d+)\s+", content, flags=re.MULTILINE)
        assert match is not None
        assert f"{float(match.group(1)) * 100:.2f}%" in readme


def test_historical_aapl_trading_history_keeps_unlabeled_terminal_source_bar():
    observed = load_data(DEFAULT_RAW_DATA, tickers=("AAPL",))
    history = prepare_indicator_history(observed)
    labeled = prepare_features(observed)
    assert history["Date"].iloc[-1] == pd.Timestamp("2017-12-29")
    assert labeled["Date"].max() == pd.Timestamp("2017-12-28")
    assert history["Close"].iloc[-1] == pytest.approx(169.23)
    assert pd.notna(history["Return"].iloc[-1])


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


def test_notebook_magic_removal_does_not_modify_string_literals(tmp_path: Path):
    import nbformat
    from scripts.refresh_notebooks import reproduce_notebook

    path = tmp_path / "literal_magic.ipynb"
    expected = "%matplotlib inline\n"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        source="%matplotlib inline\nprint('%matplotlib inline')",
        execution_count=1,
        outputs=[nbformat.v4.new_output("stream", name="stdout", text=expected)],
    )]), path)
    result = reproduce_notebook(path, check=True)
    assert result.cells[0].outputs[0].text == expected

    multiline = tmp_path / "multiline_magic.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        source='text = """\n%matplotlib inline\n"""\nprint(text)',
        execution_count=1,
        outputs=[nbformat.v4.new_output("stream", name="stdout",
                                        text="\n%matplotlib inline\n\n")],
    )]), multiline)
    reproduce_notebook(multiline, check=True)


def test_notebook_check_reproduces_stderr_and_expression_results(tmp_path: Path):
    import nbformat
    from scripts.refresh_notebooks import reproduce_notebook

    streams = tmp_path / "stream_order.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        source="import sys\nprint('before')\n_ = sys.stderr.write('warning\\n')\nprint('after')",
        execution_count=1,
        outputs=[
            nbformat.v4.new_output("stream", name="stdout", text="before\n"),
            nbformat.v4.new_output("stream", name="stderr", text="warning\n"),
            nbformat.v4.new_output("stream", name="stdout", text="after\n"),
        ],
    )]), streams)
    result = reproduce_notebook(streams, check=True)
    assert [out.name for out in result.cells[0].outputs] == ["stdout", "stderr", "stdout"]

    expression = tmp_path / "expression.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell(source="1 + 1", execution_count=1,
            outputs=[nbformat.v4.new_output("execute_result", data={"text/plain": "2"},
                execution_count=1)]),
        nbformat.v4.new_code_cell(source="1 + 1;", execution_count=2, outputs=[]),
        nbformat.v4.new_code_cell(source="1 + 1; # suppress result", execution_count=3,
                                  outputs=[]),
    ]), expression)
    reproduced = reproduce_notebook(expression, check=True)
    assert reproduced.cells[0].outputs[0].data["text/plain"] == "2"
    assert reproduced.cells[1].outputs == []


def test_notebook_check_reproduces_rich_expression_result(tmp_path: Path):
    import nbformat
    from IPython.core.formatters import DisplayFormatter
    from scripts.refresh_notebooks import reproduce_notebook

    table = pd.DataFrame({"A": [1]})
    data, metadata = DisplayFormatter().format(table)
    path = tmp_path / "rich_expression.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        source="import pandas as pd\npd.DataFrame({'A': [1]})",
        execution_count=1,
        outputs=[nbformat.v4.new_output("execute_result", data=data,
                                        metadata=metadata, execution_count=1)],
    )]), path)
    reproduce_notebook(path, check=True)


def test_notebook_execution_restores_sys_path_even_after_failure(tmp_path: Path):
    import nbformat
    from scripts.refresh_notebooks import reproduce_notebook

    original = sys.path.copy()
    for failing in (False, True):
        source = "import sys\nsys.path.insert(0, 'unexpected-import-path')"
        if failing:
            source += "\nraise RuntimeError('expected')"
        path = tmp_path / f"sys_path_{failing}.ipynb"
        nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)]), path)
        if failing:
            with pytest.raises(RuntimeError, match="Notebook sys_path_True.ipynb"):
                reproduce_notebook(path, save=False)
        else:
            reproduce_notebook(path, save=False)
        assert sys.path == original


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


def test_notebook_check_catches_small_localized_chart_markers():
    from PIL import Image, ImageDraw
    from scripts.refresh_notebooks import _verify_outputs

    base = Image.new("RGB", (1000, 750), "white")
    modified = base.copy()
    ImageDraw.Draw(modified).rectangle((450, 350, 467, 367), fill="black")

    def png_output(chart):
        buffer = io.BytesIO()
        chart.save(buffer, format="PNG")
        return {"output_type": "display_data", "data": {
            "image/png": base64.b64encode(buffer.getvalue()).decode("ascii")}}

    with pytest.raises(RuntimeError, match="Published chart differs"):
        _verify_outputs([png_output(base)], [png_output(modified)], Path("chart.ipynb"), 2)


def test_notebook_check_catches_changed_rich_text_results():
    from scripts.refresh_notebooks import _verify_outputs

    before = [{"output_type": "execute_result", "data": {"text/plain": "accuracy: 0.70"}}]
    after = [{"output_type": "execute_result", "data": {"text/plain": "accuracy: 0.10"}}]
    with pytest.raises(RuntimeError, match="Published results differ"):
        _verify_outputs(before, after, Path("results.ipynb"), 2)


def test_q_update_rejects_impossible_position_transitions():
    agent = QLearningAgent()
    invalid = [((0,), "buy", (0,)), ((1,), "sell", (1,)),
               ((0,), "hold", (1,)), ((1,), "hold", (0,))]
    for state, action, next_state in invalid:
        for terminal in (False, True):
            with pytest.raises(ValueError, match="inconsistent with the trading action"):
                agent.update_q_value(state, action, 1.0, next_state, done=terminal)
    assert agent.q_table == {}


def test_trading_rejects_reward_overflow_without_mutating_episode():
    bars = pd.DataFrame({"Open": [1., 1e308], "Close": [1., 1e-308]})
    env = SimpleTradingEnv(bars, transaction_cost=0.9)
    before = (env.index, env.position, env.done)
    with pytest.raises(ValueError, match="Trading reward must remain finite"):
        env.step("buy")
    assert (env.index, env.position, env.done) == before


def test_q_update_avoids_intermediate_overflow_for_finite_result():
    agent = QLearningAgent(alpha=0.1, gamma=0)
    agent.q_table[(0,)] = {"buy": -1e308, "sell": 0., "hold": 0.}
    agent.update_q_value((0,), "buy", 1e308, (1,), done=True)
    assert agent.q_table[(0,)]["buy"] == pytest.approx(-8e307)


def test_quantile_binner_avoids_overflow_with_finite_large_values():
    frame = pd.DataFrame({"A": [-1e308, -1e308, 1e308, 1e308]})
    binner = QuantileBinner(2).fit(frame, ("A",))
    assert np.isfinite(binner.edges["A"]).all()
    assert binner.edges["A"] == pytest.approx([0.0])
    assert binner.transform(frame, ("A",))["A"].tolist() == [0, 0, 1, 1]


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -0.1, 1.1])
def test_metrics_reject_invalid_probability_values(invalid):
    with pytest.raises(ValueError, match="Probabilities must be finite"):
        classification_metrics(pd.Series([0, 0]), np.array([invalid, 0.2]))


def test_metrics_reject_fractional_classification_labels():
    with pytest.raises(ValueError, match="Classification labels must be binary"):
        classification_metrics(pd.Series([0.5, 1.0]), np.array([0.2, 0.8]))


def test_hmm_and_trading_normalize_single_ticker_aliases():
    frame = prepare_features(synthetic_bars()).query("Name == 'AAA'").reset_index(drop=True)
    frame.loc[::2, "Name"] = " aaa "
    valid = _validated_indicators(frame, ("MA_20", "RSI_14", "MACD"))
    assert set(valid["Name"]) == {"AAA"}
    model, inferred = fit_hmm_on_indicators(frame.iloc[:75], n_components=2)
    assert set(inferred["Name"]) == {"AAA"}
    assert model.n_components == 2
    env = SimpleTradingEnv(frame.iloc[:10], state_columns=())
    assert set(env.data["Name"]) == {"AAA"}


def test_hmm_rejects_empty_or_duplicate_indicator_selection():
    frame = pd.DataFrame({"MA_20": [1.] * 12})
    for indicators in ((), ("MA_20", "MA_20")):
        with pytest.raises(ValueError, match="nonempty set of distinct columns"):
            fit_hmm_on_indicators(frame, indicators=indicators, n_components=2)


def test_notebook_magic_preserves_fstring_multiline_contents(tmp_path: Path):
    import nbformat
    from scripts.refresh_notebooks import reproduce_notebook

    path = tmp_path / "fstring.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        source='value = f"""\n%matplotlib inline\n"""\nprint(value)',
        execution_count=1,
        outputs=[nbformat.v4.new_output("stream", name="stdout",
                                        text="\n%matplotlib inline\n\n")],
    )]), path)
    reproduce_notebook(path, check=True)


def test_notebook_compiler_uses_future_flags_only_when_imported(tmp_path: Path):
    import nbformat
    from scripts.refresh_notebooks import reproduce_notebook

    missing = tmp_path / "without_future.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        source="def func(x: UndefinedAnnotation): pass",
    )]), missing)
    with pytest.raises(RuntimeError, match="Notebook without_future.ipynb") as error:
        reproduce_notebook(missing, save=False)
    assert isinstance(error.value.__cause__, NameError)

    allowed = tmp_path / "with_future.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell("from __future__ import annotations"),
        nbformat.v4.new_code_cell(
            "def func(x: UndefinedAnnotation): pass\nprint(func.__annotations__)",
        ),
    ]), allowed)
    reproduced = reproduce_notebook(allowed, save=False)
    assert "UndefinedAnnotation" in reproduced.cells[1].outputs[0].text


def test_notebook_explicit_display_uses_rich_output_not_stdout(tmp_path: Path):
    import nbformat
    from scripts.refresh_notebooks import reproduce_notebook

    path = tmp_path / "rich_display.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        "import pandas as pd\ndisplay(pd.DataFrame({'A': [1]}))",
    )]), path)
    result = reproduce_notebook(path, save=False)
    outputs = result.cells[0].outputs
    assert len(outputs) == 1
    assert outputs[0].output_type == "display_data"
    assert "text/html" in outputs[0].data


def test_notebook_figure_capture_respects_show_order_and_closed_figures(tmp_path: Path):
    import nbformat
    from scripts.refresh_notebooks import reproduce_notebook

    path = tmp_path / "show_order.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\nax.plot([1, 2])\n"
        "plt.show()\nplt.close(fig)\nprint('after display')",
    )]), path)
    outputs = reproduce_notebook(path, save=False).cells[0].outputs
    assert [out.output_type for out in outputs] == ["display_data", "stream"]
    assert "image/png" in outputs[0].data
    assert outputs[1].text == "after display\n"


def test_notebook_output_check_detects_whitespace_changes():
    import nbformat
    from scripts.refresh_notebooks import _verify_outputs

    old = [nbformat.v4.new_output("stream", name="stdout", text="Profit: 1.00\n")]
    new = [nbformat.v4.new_output("stream", name="stdout", text="  Profit: 1.00")]
    with pytest.raises(RuntimeError, match="Published results differ"):
        _verify_outputs(old, new, Path("output.ipynb"), 1)


def test_eda_notebook_includes_saved_aapl_price_time_series():
    import nbformat

    notebook = nbformat.read(ROOT / "notebooks" / "EDA_Preprocessing.ipynb", as_version=4)
    plotting = [cell for cell in notebook.cells if cell.cell_type == "code"
                and "AAPL" in cell.source and "ax.plot" in cell.source]
    assert len(plotting) == 1
    assert any("image/png" in output.get("data", {}) for output in plotting[0].outputs)


def test_hmm_fit_records_feature_order_and_rejects_incompatible_inference():
    frame = prepare_features(synthetic_bars()).query("Name == 'AAA'").reset_index(drop=True)
    model, original = fit_hmm_on_indicators(frame, n_components=2)
    assert model.stock_trading_indicator_names_ == ("MA_20", "RSI_14", "MACD")
    assert infer_filtered_regimes(model, frame)["Market_Regime"].equals(original["Market_Regime"])
    for mismatched in (("MACD", "RSI_14", "MA_20"),
                       ("MA_20", "MACD_Signal", "MACD")):
        with pytest.raises(ValueError, match="names and order used during fitting"):
            infer_filtered_regimes(model, frame, indicators=mismatched)


def test_q_update_rejects_state_dimension_mismatch_atomically():
    agent = QLearningAgent()
    for terminal in (False, True):
        with pytest.raises(ValueError, match="same dimensions"):
            agent.update_q_value((0, 0), "buy", 1.0, (0, 0, 1), done=terminal)
    assert agent.q_table == {}


def test_episode_total_reward_overflow_is_rejected():
    bars = pd.DataFrame({
        "Open": [1e-308, 1e-308, 1e308, 1e-308],
        "Close": [1e-308, 1e308, 1e308, 1e308],
    })
    env = SimpleTradingEnv(bars, transaction_cost=0, state_columns=())
    agent = QLearningAgent(epsilon=0)
    agent.q_table = {
        (0,): {"buy": 2.0, "sell": 0.0, "hold": 0.0},
        (1,): {"buy": 0.0, "sell": 2.0, "hold": 0.0},
    }
    before = copy.deepcopy(agent.q_table)
    with pytest.raises(ValueError, match="Total trading reward must remain finite"):
        agent.simulate_episode(env, train=False)
    assert agent.q_table == before


@pytest.mark.parametrize("placeholder", ["N/A", "NaN", "NULL", "None", " null "])
def test_shared_ticker_validator_rejects_placeholders_in_direct_apis(placeholder):
    original = synthetic_bars(60).query("Name == 'AAA'").reset_index(drop=True)
    invalid = original.assign(Name=placeholder)
    with pytest.raises(ValueError, match="ticker name"):
        add_technical_indicators(invalid)
    with pytest.raises(ValueError, match="ticker with a valid name"):
        SimpleTradingEnv(invalid.iloc[:4], state_columns=())
    indicators = add_technical_indicators(original)
    with pytest.raises(ValueError, match="ticker with a valid name"):
        _validated_indicators(indicators.assign(Name=placeholder), ("MA_20", "RSI_14", "MACD"))


def test_shared_ticker_validator_preserves_legitimate_na_symbol():
    original = synthetic_bars(60).query("Name == 'AAA'").reset_index(drop=True)
    legitimate = original.assign(Name=" NA ")
    indicator_frame = add_technical_indicators(legitimate)
    assert set(indicator_frame["Name"]) == {"NA"}
    assert set(_validated_indicators(indicator_frame, ("MA_20", "RSI_14", "MACD"))["Name"]) == {"NA"}
    env = SimpleTradingEnv(legitimate.sort_values("Date").iloc[:4], state_columns=())
    assert set(env.data["Name"]) == {"NA"}


def test_eda_notebook_uses_validated_percent_returns_and_reports_exclusions():
    import nbformat

    notebook = nbformat.read(ROOT / "notebooks" / "EDA_Preprocessing.ipynb", as_version=4)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    audit = [cell for cell in cells if "cleaned = load_data(DEFAULT_RAW_DATA)" in cell.source]
    assert len(audit) == 1
    assert "177" in "".join(
        str(output.get("data", {})) for output in audit[0].outputs
    )
    histograms = [cell for cell in cells if "plt.hist" in cell.source]
    boxplots = [cell for cell in cells if "ax.boxplot" in cell.source]
    assert len(histograms) == len(boxplots) == 1
    assert "returns" in histograms[0].source
    assert "ticker_returns" in boxplots[0].source
    assert "df['Close']" not in histograms[0].source + boxplots[0].source


@pytest.mark.parametrize("start,end", [
    ("2006-01-13", "2006-01-17"),  # Martin Luther King Jr. Day
    ("2024-03-28", "2024-04-01"),  # Good Friday
    ("2012-10-26", "2012-10-31"),  # Hurricane Sandy market closure
    ("2025-01-08", "2025-01-10"),  # National day of mourning
])
def test_date_only_equities_respect_real_market_holidays(start, end):
    bars = pd.DataFrame({
        "Date": [start, end], "Name": ["AAPL", "AAPL"],
        "Open": [10.0, 11.0], "High": [11.0, 13.0],
        "Low": [9.0, 10.5], "Close": [10.5, 12.0], "Volume": [100, 100],
    })
    clean = validate_bars(bars)
    assert clean["Session_Number"].tolist() == [0, 1]
    history = prepare_indicator_history(clean)
    assert history["Return"].iloc[1] == pytest.approx(12 / 10.5 - 1)
    assert SimpleTradingEnv(bars).data["Date"].dt.strftime("%Y-%m-%d").tolist() == [start, end]


def test_date_only_equities_still_detect_missing_ordinary_market_sessions():
    bars = pd.DataFrame({
        "Date": ["2024-01-12", "2024-01-17"], "Name": ["AAPL"] * 2,
        "Open": [10.0, 11.0], "High": [11.0, 13.0],
        "Low": [9.0, 10.5], "Close": [10.5, 12.0], "Volume": [100, 100],
    })
    clean = validate_bars(bars)
    assert clean["Session_Number"].tolist() == [0, 2]
    assert pd.isna(prepare_indicator_history(clean)["Return"].iloc[1])
    with pytest.raises(ValueError, match="consecutive market sessions"):
        SimpleTradingEnv(bars)


def test_inferred_nyse_holidays_match_every_bundled_market_date():
    from market_calendar import _scheduled_sessions

    raw = pd.read_csv(DEFAULT_RAW_DATA, usecols=["Date"])
    observed = pd.DatetimeIndex(pd.to_datetime(raw["Date"]).unique()).sort_values()
    scheduled = _scheduled_sessions(observed.min(), observed.max())
    assert scheduled.equals(observed)


def test_bundled_calendar_ignores_unrelated_invalid_dates_and_ticker_names(tmp_path, monkeypatch):
    import data_processing as processing

    reference = pd.DataFrame({
        "Date": ["2006-01-13", "2006-01-17", "not-a-date", "2006-01-16"],
        "Name": ["AAPL", "AAPL", "BBB", "N/A"],
        "Open": [10.0] * 4, "High": [12.0] * 4,
        "Low": [9.0] * 4, "Close": [11.0] * 4, "Volume": [100] * 4,
    })
    combined = tmp_path / DEFAULT_RAW_DATA.name
    reference.to_csv(combined, index=False)
    monkeypatch.setattr(processing, "DEFAULT_RAW_DATA", combined)
    selected = processing.load_data(combined, tickers=("AAPL",))
    assert selected["Session_Number"].tolist() == [0, 1]
    with pytest.raises(ValueError, match="parseable ISO calendar dates"):
        processing.load_data(combined)


def test_custom_csv_in_data_raw_is_not_misidentified_as_bundled(tmp_path, monkeypatch):
    import data_processing as processing

    bundled = tmp_path / DEFAULT_RAW_DATA.name
    bundled.write_text(
        "Date,Name,Open,High,Low,Close,Volume\n"
        "2006-01-13,AAPL,10,12,9,11,100\n",
        encoding="utf-8",
    )
    custom = tmp_path / "my_custom_equity_data.csv"
    custom.write_text(
        "Date,Name,Open,High,Low,Close,Volume\n"
        "2024-01-12,AAPL,10,12,9,11,100\n"
        "2024-01-16,AAPL,11,13,10,12,100\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(processing, "DEFAULT_RAW_DATA", bundled)
    assert not processing._is_bundled_raw_csv(custom)
    assert processing.load_data(custom)["Session_Number"].tolist() == [0, 1]


def test_bundled_invalid_only_weekday_is_excluded_without_compressing_time(tmp_path, monkeypatch):
    import data_processing as processing

    combined = tmp_path / DEFAULT_RAW_DATA.name
    combined.write_text(
        "Date,Name,Open,High,Low,Close,Volume\n"
        "2024-01-09,AAPL,10,12,9,11,100\n"
        "2024-01-10,AAPL,10,-1,9,11,100\n"
        "2024-01-11,AAPL,11,13,10,12,100\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(processing, "DEFAULT_RAW_DATA", combined)
    selected = processing.load_data(combined, tickers=("AAPL",))
    assert selected["Session_Number"].tolist() == [0, 2]
    assert pd.isna(prepare_indicator_history(selected)["Return"].iloc[1])


def test_bundled_reference_calendar_cache_changes_with_source_bytes(tmp_path, monkeypatch):
    import data_processing as processing

    combined = tmp_path / DEFAULT_RAW_DATA.name
    header = "Date,Name,Open,High,Low,Close,Volume\n"
    first = "2024-01-09,AAPL,10,12,9,11,100\n"
    last = "2024-01-11,AAPL,11,13,10,12,100\n"
    combined.write_text(header + first + last, encoding="utf-8")
    monkeypatch.setattr(processing, "DEFAULT_RAW_DATA", combined)
    assert len(processing._bundled_session_calendar()) == 2
    combined.write_text(header + first + "2024-01-10,AAPL,10,12,9,11,100\n" + last,
                        encoding="utf-8")
    assert len(processing._bundled_session_calendar()) == 3


def test_supplied_session_ordinals_do_not_allow_weekend_stock_bars():
    bars = pd.DataFrame({
        "Date": ["2024-01-12", "2024-01-13"], "Name": ["AAPL"] * 2,
        "Open": [10.0, 11.0], "High": [12.0, 13.0],
        "Low": [9.0, 10.0], "Close": [11.0, 12.0], "Volume": [100, 100],
        "Session_Number": [0, 1],
    })
    with pytest.raises(ValueError, match="weekdays"):
        validate_bars(bars)
    with pytest.raises(ValueError, match="weekdays"):
        add_technical_indicators(bars)
    with pytest.raises(ValueError, match="weekdays"):
        SimpleTradingEnv(bars)


def test_boolean_market_prices_and_volumes_are_never_interpreted_as_numbers():
    bars = pd.DataFrame({
        "Date": ["2024-01-09", "2024-01-10"], "Name": ["AAPL"] * 2,
        "Open": [True, 10.0], "High": [12.0, 12.0],
        "Low": [0.5, 9.0], "Close": [11.0, 11.0], "Volume": [100, 100],
    })
    assert len(validate_bars(bars)) == 1
    with pytest.raises(ValueError, match="Indicator OHLC prices"):
        add_technical_indicators(bars)
    with pytest.raises(ValueError, match="booleans"):
        SimpleTradingEnv(bars)
    volume_boolean = bars.assign(Open=[10.0, 10.0], Volume=[True, 100])
    assert len(validate_bars(volume_boolean)) == 1


def test_holdout_trading_is_split_into_independent_contiguous_episodes():
    rows = pd.DataFrame({
        "Date": pd.bdate_range("2024-01-08", periods=6),
        "Name": ["AAPL"] * 6, "Open": [10.0] * 6,
        "Close": [11.0] * 6, "Session_Number": [0, 1, 3, 4, 6, 8],
    })
    segments = contiguous_trading_segments(rows)
    assert [segment["Session_Number"].tolist() for segment in segments] == [[0, 1], [3, 4]]
    policy = QLearningAgent(epsilon=0.0)
    original_table = copy.deepcopy(policy.q_table)
    rewards = [policy.simulate_episode(SimpleTradingEnv(segment), train=False)
               for segment in segments]
    assert len(rewards) == 2
    assert policy.q_table == original_table


def test_eda_source_uses_repo_root_and_the_model_pipeline_calendar():
    import nbformat

    notebook = nbformat.read(ROOT / "notebooks/EDA_Preprocessing.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert "df = pd.read_csv(root / 'data/raw/" in source
    assert "cleaned = load_data(DEFAULT_RAW_DATA)" in source
    assert "cleaned = validate_bars(df)" not in source
    reference = prepare_indicator_history(load_data(DEFAULT_RAW_DATA))
    assert reference["Return"].notna().sum() == 93_228


def test_direct_indicators_detect_missing_weekdays_without_session_metadata():
    bars = synthetic_bars(85).query("Name == 'AAA'").sort_values("Date").reset_index(drop=True)
    missing_date = bars.loc[45, "Date"]
    next_date = bars.loc[46, "Date"]
    with_metadata = bars.assign(Session_Number=np.arange(len(bars)))
    shortened = bars.loc[bars.Date != missing_date].copy()
    shortened_with_metadata = with_metadata.loc[with_metadata.Date != missing_date].copy()
    result = add_technical_indicators(shortened)
    reference = add_technical_indicators(shortened_with_metadata)
    crossing = result.loc[result.Date == next_date].iloc[0]
    assert pd.isna(crossing["Return"])
    assert pd.isna(crossing["RSI_14"])
    assert pd.isna(crossing["ATR"])
    for name in ("Return", "RSI_14", "ATR", "Return_Lag1"):
        pd.testing.assert_series_equal(result[name], reference[name], check_names=False)


def test_direct_hmm_restarts_at_missing_weekday_without_session_metadata():
    class AlternatingModel:
        covariance_type = "diag"
        means_ = np.array([[0.], [0.]])
        covars_ = np.array([[[1.]], [[1.]]])
        startprob_ = np.array([1., 0.])
        transmat_ = np.array([[0., 1.], [1., 0.]])

    frame = pd.DataFrame({
        "Date": ["2024-02-26", "2024-02-28"],
        "MA_20": [0., 0.],
    })
    result = infer_filtered_regimes(AlternatingModel(), frame, ("MA_20",))
    assert result.Market_Regime.tolist() == [0, 0]


def test_direct_hmm_fitting_splits_weekday_gaps_without_session_metadata(monkeypatch):
    from hmmlearn.hmm import GaussianHMM

    source = prepare_features(synthetic_bars(130)).query("Name == 'AAA'")
    source = source.dropna(subset=["MA_20", "RSI_14", "MACD"])
    source = source.drop(columns="Session_Number").sort_values("Date").reset_index(drop=True)
    source = source.drop(index=source.index[50]).reset_index(drop=True)
    lengths_seen = []
    original_fit = GaussianHMM.fit

    def recording_fit(self, values, lengths=None):
        lengths_seen.append(lengths)
        return original_fit(self, values, lengths=lengths)

    monkeypatch.setattr(GaussianHMM, "fit", recording_fit)
    fit_hmm_on_indicators(source, n_components=2)
    assert len(lengths_seen[0]) == 2
    assert sum(lengths_seen[0]) == len(source)


def test_direct_trading_rejects_weekday_gap_without_session_metadata():
    frame = pd.DataFrame({
        "Date": ["2024-02-26", "2024-02-28"],
        "Open": [10., 11.], "Close": [11., 12.],
    })
    with pytest.raises(ValueError, match="consecutive market sessions"):
        SimpleTradingEnv(frame)
    frame["Session_Number"] = [0, 1]
    assert len(SimpleTradingEnv(frame).data) == 2


@pytest.mark.parametrize("invalid", [0, -1, -0.5, np.nan, np.inf, True, "5"])
def test_bdeu_prior_strength_rejects_invalid_values(invalid):
    frame = pd.DataFrame({"Return": [0, 1] * 10, "Trend": [0, 1] * 10})
    with pytest.raises(ValueError, match="equivalent_sample_size"):
        fit_network(frame, ("Return",), 2, invalid)


@pytest.mark.parametrize("invalid", [2, "bad", None, np.nan])
def test_network_predictions_reject_unknown_categorical_evidence(invalid):
    frame = pd.DataFrame({"Return": [0, 1] * 10, "Trend": [0, 1] * 10})
    fitted = fit_network(frame, ("Return",), 2, 5)
    with pytest.raises(ValueError, match="Unknown Bayesian-network states for Return"):
        predict_up_probability(fitted, pd.DataFrame({"Return": [0, invalid]}))
    with pytest.raises(ValueError, match="Missing Bayesian-network parent feature"):
        predict_up_probability(fitted, pd.DataFrame({"Other": [0]}))


@pytest.mark.parametrize("symbol", ["<NA>", "123", "-AAPL", "AAPL!", "...", "A B"])
def test_ticker_validation_rejects_malformed_nonplaceholder_names(symbol):
    frame = synthetic_bars(40).query("Name == 'AAA'").assign(Name=symbol)
    with pytest.raises(ValueError, match="ticker name"):
        add_technical_indicators(frame)


def test_ticker_validation_accepts_delimited_share_classes():
    frame = synthetic_bars(40).query("Name == 'AAA'").assign(Name="brk.b")
    assert set(add_technical_indicators(frame).Name) == {"BRK.B"}


@pytest.mark.parametrize("field", ["alpha", "gamma", "epsilon"])
@pytest.mark.parametrize("value", [True, False, np.bool_(True)])
def test_q_learning_rejects_boolean_hyperparameters(field, value):
    with pytest.raises(ValueError, match="Invalid Q-learning parameters"):
        QLearningAgent(**{field: value})


def test_notebook_execution_restores_environment_variables(tmp_path: Path, monkeypatch):
    import nbformat
    from scripts.refresh_notebooks import reproduce_notebook

    monkeypatch.setenv("STOCK_TRADING_AUDIT_EXISTING", "original")
    monkeypatch.delenv("STOCK_TRADING_AUDIT_NEW", raising=False)
    original = os.environ.copy()
    for failing in (False, True):
        source = (
            "import os\n"
            "os.environ['STOCK_TRADING_AUDIT_EXISTING'] = 'modified'\n"
            "os.environ['STOCK_TRADING_AUDIT_NEW'] = 'added'"
        )
        if failing:
            source += "\nraise RuntimeError('expected failure')"
        path = tmp_path / f"environment_{failing}.ipynb"
        nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)]), path)
        if failing:
            with pytest.raises(RuntimeError, match="Notebook environment_True.ipynb"):
                reproduce_notebook(path, save=False)
        else:
            reproduce_notebook(path, save=False)
        assert dict(os.environ) == original


def test_executable_target_uses_next_open_and_preserves_historical_default():
    bars = synthetic_bars()
    bars.loc[(bars["Name"] == "AAA") & (bars.index % 2 == 0), "Open"] = (
        bars.loc[(bars["Name"] == "AAA") & (bars.index % 2 == 0), "Close"] + 0.5
    )
    default = prepare_features(bars)
    historical = prepare_features(bars, target="close_to_close")
    executable = prepare_features(bars, target="next_open_to_close")
    pd.testing.assert_frame_equal(default, historical)
    assert executable[["Name", "Date", "Target_Date"]].equals(
        historical[["Name", "Date", "Target_Date"]]
    )
    original = bars.set_index(["Name", "Date"])
    expected = [int(original.loc[(row.Name, row.Target_Date), "Close"]
                    > original.loc[(row.Name, row.Target_Date), "Open"])
                for row in executable.itertuples()]
    assert executable["Trend"].tolist() == expected
    with pytest.raises(ValueError, match="target must be one of"):
        prepare_features(bars, target="same_day")


def test_trading_slippage_and_drawdown_metrics():
    bars = pd.DataFrame({"Open": [100., 110., 120.], "Close": [100., 115., 125.]})
    env = SimpleTradingEnv(bars, transaction_cost=0.001, slippage_bps=100,
                           state_columns=())
    _, buy_reward, done, first = env.step("buy")
    assert not done
    assert first["execution_price"] == pytest.approx(111.1)
    assert first["slippage_cost"] == pytest.approx(1.1)
    assert buy_reward == pytest.approx(5 - 1.1 - 0.1111)
    _, sell_reward, done, second = env.step("sell")
    assert done
    assert second["execution_price"] == pytest.approx(118.8)
    assert sell_reward == pytest.approx(5 - 1.2 - 0.1188)
    report = env.episode_metrics()
    assert report["pnl"] == pytest.approx(buy_reward + sell_reward)
    assert report["trades"] == 2
    assert report["fees_dollars"] == pytest.approx(0.2299)
    assert report["slippage_dollars"] == pytest.approx(2.3)
    assert report["intraday_exposure_fraction"] == pytest.approx(0.5)
    assert report["overnight_exposure_fraction"] == pytest.approx(0.5)
    assert report["terminal_position"] == 0

    falling = SimpleTradingEnv(
        pd.DataFrame({"Open": [100., 100., 90.], "Close": [100., 95., 92.]}),
        transaction_cost=0, state_columns=(),
    )
    falling.step("buy")
    falling.step("sell")
    assert falling.episode_metrics()["max_drawdown_dollars"] == pytest.approx(10)
    assert falling.reset() == (0,)
    assert falling.episode_metrics()["trades"] == 0


@pytest.mark.parametrize("kwargs", [
    {"slippage_bps": -1}, {"slippage_bps": float("nan")},
    {"slippage_bps": True}, {"slippage_bps": 10_000},
    {"transaction_cost": False}, {"transaction_cost": float("inf")},
])
def test_trading_rejects_invalid_execution_costs(kwargs):
    with pytest.raises(ValueError):
        SimpleTradingEnv(pd.DataFrame({"Open": [1, 2], "Close": [1, 2]}), **kwargs)


def test_paired_market_date_bootstrap_is_deterministic_and_paired():
    dates = pd.bdate_range("2026-01-01", periods=20).repeat(2)
    observed = pd.DataFrame({
        "Date": dates, "actual": np.arange(40) % 2,
        "majority": 0, "baseline_prob": 0.1,
        "enhanced_prob": np.where(np.arange(40) % 2, 0.9, 0.1),
    })
    first = paired_block_accuracy_intervals(
        observed, block_sessions=5, bootstrap_samples=100, seed=11,
    )
    assert first == paired_block_accuracy_intervals(
        observed, block_sessions=5, bootstrap_samples=100, seed=11,
    )
    assert first["baseline"] == [0.0, 0.0]
    assert first["enhanced"] == [0.5, 0.5]


def test_walk_forward_executable_evaluation_has_disjoint_test_folds(tmp_path: Path):
    csv = tmp_path / "walk_forward.csv"
    bars = synthetic_bars(150)
    bars.loc[bars.index % 3 == 0, "Open"] = bars.loc[bars.index % 3 == 0, "Close"] + 0.3
    bars.to_csv(csv, index=False)
    report = walk_forward_experiment(
        csv, target="next_open_to_close", folds=2,
        bootstrap_samples=100, seed=3,
    )
    assert report["target"] == "next_open_to_close"
    assert len(report["folds"]) == 2
    assert report["folds"][0]["test_end"] <= report["folds"][1]["validation_end"]
    assert report["holdout_observations"] == sum(
        fold["observations"]["test"] for fold in report["folds"]
    )
    assert 0 <= report["baseline"]["accuracy"] <= 1
    assert 0 <= report["enhanced"]["accuracy"] <= 1
    with pytest.raises(ValueError, match="folds must be"):
        walk_forward_experiment(csv, folds=6)


def test_mixed_exchange_holiday_does_not_contaminate_nyse_ticker(tmp_path: Path):
    # NYSE was closed on 2025-01-20, while BBB's exchange was open.
    source = pd.DataFrame({
        "Name": ["AAA", "AAA", "BBB", "BBB", "BBB"],
        "Date": ["2025-01-17", "2025-01-21", "2025-01-17", "2025-01-20", "2025-01-21"],
        "Open": [100, 110, 200, 210, 220],
        "High": [101, 111, 201, 211, 221],
        "Low": [99, 109, 199, 209, 219],
        "Close": [100, 110, 200, 210, 220],
        "Volume": [1000] * 5,
    })
    csv = tmp_path / "mixed_exchanges.csv"
    source.to_csv(csv, index=False)
    combined = load_data(csv)
    aaa = combined.loc[combined.Name == "AAA"]
    bbb = combined.loc[combined.Name == "BBB"]
    assert aaa.Session_Number.tolist() == [0, 1]
    assert bbb.Session_Number.tolist() == [0, 1, 2]
    assert add_technical_indicators(combined).loc[
        lambda rows: (rows.Name == "AAA") & (rows.Date == pd.Timestamp("2025-01-21")),
        "Return",
    ].iloc[0] == pytest.approx(0.1)
    pd.testing.assert_frame_equal(load_data(csv, tickers=("AAA",)), aaa.reset_index(drop=True))


def test_mixed_exchange_explicit_ordinals_are_valid_per_ticker():
    dates = ["2025-01-17", "2025-01-21", "2025-01-17", "2025-01-20", "2025-01-21"]
    names = ["AAA", "AAA", "BBB", "BBB", "BBB"]
    prices = [100, 101, 200, 201, 202]
    source = pd.DataFrame({
        "Name": names, "Date": dates, "Session_Number": [0, 1, 0, 1, 2],
        "Open": prices, "Close": prices,
        "High": [value + 1 for value in prices],
        "Low": [value - 1 for value in prices], "Volume": [1000] * 5,
    })
    checked = validate_bars(source)
    assert checked.loc[checked.Name == "AAA", "Session_Number"].tolist() == [0, 1]
    corrupted = source.copy()
    corrupted.loc[(corrupted.Name == "BBB") & (corrupted.Date == "2025-01-21"),
                  "Session_Number"] = 1
    with pytest.raises(ValueError, match="Session_Number"):
        validate_bars(corrupted)


def test_mlk_nyse_holiday_only_applies_from_1998():
    def two_bars(first: str, second: str) -> pd.DataFrame:
        return pd.DataFrame({
            "Name": ["AAA", "AAA"], "Date": [first, second],
            "Open": [100, 101], "High": [101, 102],
            "Low": [99, 100], "Close": [100, 101], "Volume": [1000, 1000],
        })

    # Monday 1997-01-20 was still a regular NYSE session.
    assert validate_bars(two_bars("1997-01-17", "1997-01-21"))["Session_Number"].tolist() == [0, 2]
    # The third Monday in January became an NYSE holiday in 1998.
    assert validate_bars(two_bars("1998-01-16", "1998-01-20"))["Session_Number"].tolist() == [0, 1]


@pytest.mark.parametrize("column,value", [
    ("baseline_prob", np.nan), ("enhanced_prob", np.inf),
    ("baseline_prob", -0.01), ("enhanced_prob", 1.01),
    ("baseline_prob", "not a probability"),
    ("actual", 2), ("majority", np.nan),
])
def test_block_bootstrap_rejects_invalid_scores(column, value):
    scored = pd.DataFrame({
        "Date": pd.bdate_range("2026-01-05", periods=30),
        "actual": [0, 1] * 15, "majority": [0] * 30,
        "baseline_prob": [0.2] * 30, "enhanced_prob": [0.8] * 30,
    })
    if isinstance(value, str):
        scored[column] = scored[column].astype(object)
    scored.loc[0, column] = value
    with pytest.raises(ValueError, match="Bootstrap"):
        paired_block_accuracy_intervals(scored, block_sessions=10, bootstrap_samples=100)


def test_block_bootstrap_rejects_one_independent_date_block():
    scored = pd.DataFrame({
        "Date": pd.bdate_range("2026-01-05", periods=8),
        "actual": [0, 1] * 4, "majority": [0] * 8,
        "baseline_prob": [0.8] * 8, "enhanced_prob": [0.2] * 8,
    })
    with pytest.raises(ValueError, match="at least two chronological date blocks"):
        paired_block_accuracy_intervals(scored, block_sessions=10, bootstrap_samples=100)


def test_block_bootstrap_rejects_intraday_timestamps_as_sessions():
    dates = [day + pd.Timedelta(hours=hour)
             for day in pd.bdate_range("2026-01-05", periods=8) for hour in (10, 15)]
    scored = pd.DataFrame({
        "Date": dates, "actual": [0, 1] * 8, "majority": [0] * 16,
        "baseline_prob": [0.8] * 16, "enhanced_prob": [0.2] * 16,
    })
    with pytest.raises(ValueError, match="Bootstrap dates"):
        paired_block_accuracy_intervals(scored, block_sessions=10, bootstrap_samples=100)


def test_custom_source_ordinals_remain_stable_when_selecting_later_ticker(tmp_path: Path):
    source = pd.DataFrame({
        "Name": ["BBB"] * 4 + ["AAA"] * 2,
        "Date": ["2025-01-17", "2025-01-21", "2025-01-22", "2025-01-23",
                 "2025-01-22", "2025-01-23"],
        "Open": [100] * 6, "High": [101] * 6,
        "Low": [99] * 6, "Close": [100] * 6, "Volume": [1000] * 6,
    })
    csv = tmp_path / "late_ticker.csv"
    source.to_csv(csv, index=False)
    combined = load_data(csv)
    selected = load_data(csv, tickers=("AAA",))
    pd.testing.assert_frame_equal(
        selected, combined.loc[combined.Name == "AAA"].reset_index(drop=True),
    )
    assert selected.Session_Number.tolist() == [2, 3]


def test_selected_modern_ticker_ignores_unrelated_unverified_historical_dates(tmp_path: Path):
    source = pd.DataFrame({
        "Name": ["AAA", "AAA", "BBB", "BBB"],
        "Date": ["2025-01-17", "2025-01-21", "1968-07-09", "1968-07-11"],
        "Open": [100] * 4, "High": [101] * 4,
        "Low": [99] * 4, "Close": [100] * 4, "Volume": [1000] * 4,
    })
    csv = tmp_path / "historic_and_modern.csv"
    source.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="before 1971"):
        load_data(csv)
    assert load_data(csv, tickers=("AAA",)).Session_Number.tolist() == [0, 1]


def test_pre_1971_nyse_dates_require_verified_calendar():
    # During the 1968 paperwork crisis, 1968-07-10 was an NYSE closure.
    bars = pd.DataFrame({
        "Name": ["AAA", "AAA"], "Date": ["1968-07-09", "1968-07-11"],
        "Open": [100, 101], "High": [101, 102],
        "Low": [99, 100], "Close": [100, 101], "Volume": [1000, 1000],
    })
    with pytest.raises(ValueError, match="before 1971 require verified Session_Number"):
        validate_bars(bars)
    assert validate_bars(bars.assign(Session_Number=[0, 1])).Session_Number.tolist() == [0, 1]


def test_trading_segments_never_combine_different_symbols():
    bars = pd.DataFrame({
        "Name": ["AAA", "BBB", "AAA", "BBB"],
        "Date": pd.to_datetime(["2025-01-21", "2025-01-21",
                                "2025-01-22", "2025-01-22"]),
        "Session_Number": [0, 0, 1, 1],
    })
    segments = contiguous_trading_segments(bars)
    assert len(segments) == 2
    assert {tuple(group.Name) for group in segments} == {("AAA", "AAA"), ("BBB", "BBB")}
    undated = bars.drop(columns="Date").sort_values("Name")
    assert len(contiguous_trading_segments(undated)) == 2


def test_next_session_targets_remain_exact_with_large_uint64_ordinals():
    count = 90
    dates = pd.bdate_range("2025-01-02", periods=count)
    close = 100 + 0.1 * np.arange(count)
    bars = pd.DataFrame({
        "Name": ["AAA"] * count, "Date": dates,
        "Session_Number": (2**53 + 1 + np.arange(count)).astype("uint64"),
        "Open": close, "High": close + 1, "Low": close - 1,
        "Close": close, "Volume": [1000] * count,
    })
    prepared = prepare_features(bars)
    assert len(prepared) > 40
    assert (prepared.Target_Date > prepared.Date).all()
    assert prepared.Trend.eq(1).all()


def test_exchange_metadata_shares_session_gaps_only_within_same_exchange(tmp_path: Path):
    source = pd.DataFrame({
        "Name": ["AAA"] * 2 + ["BBB"] * 2 + ["CCC"] * 3,
        "Exchange": ["NYSE"] * 2 + ["LSE"] * 5,
        "Date": ["2025-01-17", "2025-01-21",
                 "2025-01-17", "2025-01-21",
                 "2025-01-17", "2025-01-20", "2025-01-21"],
        "Open": [100] * 7, "High": [101] * 7,
        "Low": [99] * 7, "Close": [100] * 7, "Volume": [1000] * 7,
    })
    csv = tmp_path / "exchange_calendar.csv"
    source.to_csv(csv, index=False)
    combined = load_data(csv)
    assert combined.loc[combined.Name == "AAA", "Session_Number"].tolist() == [0, 1]
    assert combined.loc[combined.Name == "BBB", "Session_Number"].tolist() == [0, 2]
    assert combined.loc[combined.Name == "CCC", "Session_Number"].tolist() == [0, 1, 2]
    for symbol in ("AAA", "BBB", "CCC"):
        pd.testing.assert_frame_equal(
            load_data(csv, tickers=(symbol,)),
            combined.loc[combined.Name == symbol].reset_index(drop=True),
        )
    invalid = source.copy()
    invalid.loc[(invalid.Name == "BBB") & (invalid.Date == "2025-01-21"),
                "Exchange"] = "NYSE"
    with pytest.raises(ValueError, match="exactly one Exchange"):
        validate_bars(invalid)


def test_walk_forward_rejects_too_few_bootstrap_dates_before_training(monkeypatch):
    from walk_forward_evaluation import walk_forward_experiment as evaluate
    import walk_forward_evaluation as walk_module

    count = 115
    index = np.arange(count)
    close = 100 + index * 0.08 + 2 * np.sin(index * 0.35)
    bars = validate_bars(pd.DataFrame({
        "Date": pd.bdate_range("2020-01-02", periods=count),
        "Name": ["AAA"] * count, "Open": close - 0.2,
        "High": close + 1, "Low": close - 1,
        "Close": close, "Volume": [1000] * count,
    }))
    assert len(prepare_features(bars)) >= 80
    monkeypatch.setattr(walk_module, "load_data", lambda *args, **kwargs: bars)
    monkeypatch.setattr(walk_module, "train_and_select",
                        lambda *args, **kwargs: pytest.fail("Model training began too soon"))
    with pytest.raises(ValueError, match="too few test dates"):
        evaluate("synthetic.csv", folds=1, bootstrap_samples=100)


@pytest.mark.parametrize("closed,before,after", [
    ("1972-11-07", "1972-11-06", "1972-11-08"),
    ("1972-12-28", "1972-12-27", "1972-12-29"),
    ("1973-01-25", "1973-01-24", "1973-01-26"),
    ("1976-11-02", "1976-11-01", "1976-11-03"),
    ("1977-07-14", "1977-07-13", "1977-07-15"),
    ("1980-11-04", "1980-11-03", "1980-11-05"),
    ("1985-09-27", "1985-09-26", "1985-09-30"),
    ("1994-04-27", "1994-04-26", "1994-04-28"),
])
def test_historical_nyse_full_day_closures_do_not_create_sessions(closed, before, after):
    bars = pd.DataFrame({
        "Name": ["AAA"] * 2, "Exchange": ["NYSE"] * 2,
        "Date": [before, after], "Open": [100, 101],
        "High": [101, 102], "Low": [99, 100],
        "Close": [100, 101], "Volume": [10, 10],
    })
    valid = validate_bars(bars)
    assert valid.Session_Number.tolist() == [0, 1], closed


@pytest.mark.parametrize("exchange", ["NYSE", "XNYS"])
@pytest.mark.parametrize("with_metadata", [False, True])
def test_declared_nyse_rejects_bar_during_known_closed_market(exchange, with_metadata):
    bars = pd.DataFrame({
        "Name": ["AAA"] * 3, "Exchange": [exchange] * 3,
        "Date": ["2001-09-10", "2001-09-11", "2001-09-17"],
        "Open": [100] * 3, "High": [101] * 3,
        "Low": [99] * 3, "Close": [100] * 3,
        "Volume": [10] * 3,
    })
    if with_metadata:
        bars["Session_Number"] = [0, 1, 2]
    with pytest.raises(ValueError, match="closed-market date: 2001-09-11"):
        validate_bars(bars)
    # Observed dates outside the reference NYSE calendar remain available to
    # other explicitly identified exchanges.
    other_exchange = bars.assign(Exchange="XOTHER")
    assert len(validate_bars(other_exchange)) == 3


def _extra_same_exchange_market_date() -> pd.DataFrame:
    rows = []
    for symbol in ("AAA", "BBB"):
        for index, day in enumerate(pd.bdate_range("2024-01-12", "2024-01-17")):
            if symbol == "AAA" and day == pd.Timestamp("2024-01-15"):
                continue
            price = 100.0 + index
            rows.append({
                "Name": symbol, "Exchange": "XTEST", "Date": day,
                "Open": price, "High": price + 2, "Low": price - 1,
                "Close": price + 1, "Volume": 1000,
            })
    return pd.DataFrame(rows)


def test_direct_indicators_preserve_peer_observed_exchange_session_gaps():
    source = _extra_same_exchange_market_date()
    direct = add_technical_indicators(source)
    validated = add_technical_indicators(validate_bars(source))
    for frame in (direct, validated):
        aaa = frame.loc[frame.Name == "AAA"].sort_values("Date")
        assert aaa.Session_Number.tolist() == [0, 2, 3]
        assert pd.isna(aaa.loc[aaa.Date == pd.Timestamp("2024-01-16"), "Return"].iloc[0])


def test_direct_trading_segments_preserve_shared_exchange_calendar():
    source = _extra_same_exchange_market_date()
    segments = contiguous_trading_segments(source)
    assert len(segments) == 2
    assert [len(segment) for segment in segments] == [2, 4]
    aaa = next(segment for segment in segments if segment.Name.iloc[0] == "AAA")
    assert aaa.Date.dt.strftime("%Y-%m-%d").tolist() == ["2024-01-16", "2024-01-17"]
    assert aaa.Session_Number.tolist() == [2, 3]


@pytest.mark.parametrize("with_metadata", [False, True])
def test_trading_segments_reject_duplicate_ticker_dates(with_metadata):
    dates = pd.to_datetime([
        "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-04", "2024-01-05",
    ])
    source = pd.DataFrame({"Name": ["AAA"] * len(dates), "Date": dates})
    if with_metadata:
        source["Session_Number"] = [0, 1, 1, 2, 3]
    with pytest.raises(ValueError, match="Duplicate ticker/date"):
        contiguous_trading_segments(source)


def test_trading_segments_parse_dates_before_chronological_sorting():
    source = pd.DataFrame({
        "Name": ["AAA"] * 3,
        "Date": ["2025-01-9", "2025-01-10", "2025-01-13"],
    })
    segments = contiguous_trading_segments(source)
    assert len(segments) == 1
    assert segments[0].Date.dt.strftime("%Y-%m-%d").tolist() == [
        "2025-01-09", "2025-01-10", "2025-01-13",
    ]


@pytest.mark.parametrize("forward", [False, True])
def test_empty_session_adjacency_has_empty_boolean_result(forward):
    positions = pd.Series([], index=pd.Index([], name="observation"), dtype="int64")
    result = adjacent_session_flags(positions, forward=forward)
    assert result.empty and result.dtype == bool
    pd.testing.assert_index_equal(result.index, positions.index)
