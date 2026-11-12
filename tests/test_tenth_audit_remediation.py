"""Independent tenth-audit regressions: provenance, partitions, and markets."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from scripts import data_processing
from scripts.data_processing import validate_bars
from scripts.date_validation import (
    mark_verified_daily_dates,
    require_verified_daily_date_source,
)
from scripts.hmm_integration import fit_hmm_on_indicators
from scripts.model_training import (
    ChronologicalSplit,
    _validate_direct_model_split,
    train_and_select,
)
from scripts.technical_indicators import add_technical_indicators


def _attested_bars():
    frame = pd.DataFrame(
        {
            "Name": ["AAA", "AAA", "AAA"],
            "Date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "Target_Date": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
            "Open": [10, 11, 12],
            "High": [12, 13, 14],
            "Low": [9, 10, 11],
            "Close": [11, 12, 13],
            "Volume": [100, 100, 100],
        }
    )
    return mark_verified_daily_dates(frame)


def test_valid_target_attestation_survives_bar_validation():
    prepared = validate_bars(_attested_bars())
    require_verified_daily_date_source(prepared, column="Target_Date")


def test_changed_target_cannot_be_reauthorized_by_bar_validation():
    changed = _attested_bars()
    changed.loc[0, "Target_Date"] = pd.Timestamp("2024-01-08")
    with pytest.raises(ValueError, match="unknown original precision or changed"):
        validate_bars(changed)


def test_changed_target_cannot_be_reauthorized_by_indicator_calculation():
    changed = _attested_bars()
    changed.loc[0, "Target_Date"] = pd.Timestamp("2024-01-08")
    output = add_technical_indicators(changed)
    with pytest.raises(ValueError, match="unknown original precision or changed"):
        require_verified_daily_date_source(output, column="Target_Date")


def _row(date, target):
    return {
        "Name": "AAA",
        "Date": date,
        "Target_Date": target,
        "Trend": 1,
        "Return": 0.01,
        "Intraday_Return": 0.01,
        "Daily_Range": 0.02,
    }


def _good_split():
    return ChronologicalSplit(
        train=pd.DataFrame([_row("2024-01-02", "2024-01-03")]),
        validation=pd.DataFrame([_row("2024-01-04", "2024-01-05")]),
        test=pd.DataFrame([_row("2024-01-08", "2024-01-09")]),
        train_end=pd.Timestamp("2024-01-03"),
        validation_end=pd.Timestamp("2024-01-05"),
    )


def test_direct_model_training_accepts_valid_chronological_partition():
    _validate_direct_model_split(_good_split())


@pytest.mark.parametrize("corruption", ["overlap", "future_train_target", "duplicate"])
def test_direct_model_training_rejects_leakage_before_fitting(corruption):
    good = _good_split()
    if corruption == "overlap":
        bad = replace(good, validation=good.train.copy())
    elif corruption == "future_train_target":
        train = good.train.copy()
        train.loc[0, "Target_Date"] = "2024-01-04"
        bad = replace(good, train=train)
    else:
        train = pd.concat([good.train, good.train], ignore_index=True)
        bad = replace(good, train=train)
    with pytest.raises(ValueError, match="partition|duplicate|Duplicate|next market session"):
        train_and_select(
            bad,
            ("Return", "Intraday_Return", "Daily_Range"),
            num_bins=2,
        )


def test_target_attestation_is_checked_for_direct_model_training():
    good = _good_split()
    train = good.train.copy()
    train["Date"] = pd.to_datetime(train["Date"])
    train["Target_Date"] = pd.to_datetime(train["Target_Date"])
    mark_verified_daily_dates(train)
    train.loc[0, "Target_Date"] = pd.Timestamp("2024-01-04")
    with pytest.raises(ValueError, match="unknown original precision or changed"):
        _validate_direct_model_split(replace(good, train=train))


@pytest.mark.parametrize("invalid", [None, 0, -1, 1.5, True, "2", np.float64(2)])
def test_hmm_rejects_invalid_component_counts_before_fit(invalid):
    with pytest.raises(ValueError, match="HMM n_components"):
        fit_hmm_on_indicators(pd.DataFrame(), n_components=invalid)


@pytest.mark.parametrize(
    "invalid",
    [None, -1, True, np.bool_(True), "12", 1.5, 2**32],
)
def test_hmm_rejects_unsupported_or_nondeterministic_seeds(invalid):
    with pytest.raises(ValueError, match="HMM random_state"):
        fit_hmm_on_indicators(pd.DataFrame(), random_state=invalid)


def _historical_bundled_source(tmp_path, monkeypatch):
    path = tmp_path / data_processing.DEFAULT_RAW_DATA.name
    entries = [
        ("1985-09-26", "MSFT"),
        ("1985-09-27", "MSFT"),
        ("1985-09-30", "MSFT"),
        ("1985-09-26", "IBM"),
        ("1985-09-30", "IBM"),
    ]
    path.write_text(
        "Date,Name,Open,High,Low,Close,Volume\n"
        + "".join(f"{day},{symbol},10,12,9,11,100\n" for day, symbol in entries),
        encoding="utf-8",
    )
    monkeypatch.setattr(data_processing, "DEFAULT_RAW_DATA", path)
    return path


def test_bundled_nasdaq_halfday_preserved_without_fake_nyse_gap(tmp_path, monkeypatch):
    path = _historical_bundled_source(tmp_path, monkeypatch)
    calendar = data_processing._bundled_session_calendar()
    assert calendar.strftime("%Y-%m-%d").tolist() == [
        "1985-09-26",
        "1985-09-27",
        "1985-09-30",
    ]
    nasdaq = data_processing.load_data(path, tickers=("MSFT",))
    nyse = data_processing.load_data(path, tickers=("IBM",))
    assert nasdaq["Session_Number"].tolist() == [0, 1, 2]
    assert nyse["Session_Number"].tolist() == [0, 1]
    assert set(nasdaq["Exchange"]) == {"NASDAQ"}
    assert set(nyse["Exchange"]) == {"NYSE"}


def test_bundled_nyse_closed_date_stays_rejected_even_if_nasdaq_open(
    tmp_path, monkeypatch
):
    path = _historical_bundled_source(tmp_path, monkeypatch)
    source = pd.read_csv(path)
    source.loc[len(source)] = ["1985-09-27", "IBM", 10, 12, 9, 11, 100]
    source.to_csv(path, index=False)
    with pytest.raises(ValueError, match="closed-market date"):
        data_processing.load_data(path, tickers=("IBM",))


def test_bundled_modern_source_keeps_historical_output_schema(tmp_path, monkeypatch):
    path = tmp_path / data_processing.DEFAULT_RAW_DATA.name
    path.write_text(
        "Date,Name,Open,High,Low,Close,Volume\n"
        "2024-01-12,AAPL,10,12,9,11,100\n"
        "2024-01-16,AAPL,11,13,10,12,100\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(data_processing, "DEFAULT_RAW_DATA", path)
    loaded = data_processing.load_data(path)
    assert "Exchange" not in loaded
    assert loaded["Session_Number"].tolist() == [0, 1]
