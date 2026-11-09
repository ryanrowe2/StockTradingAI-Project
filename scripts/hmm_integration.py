"""Hidden Markov regimes with chronological, forward-only state inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

if __package__:
    from .date_validation import parse_daily_dates
    from .session_validation import validate_session_positions
    from .ticker_validation import normalize_ticker_symbols
else:
    from date_validation import parse_daily_dates
    from session_validation import validate_session_positions
    from ticker_validation import normalize_ticker_symbols


DEFAULT_INDICATORS = ("MA_20", "RSI_14", "MACD")


def _gaussian_log_emissions(model: GaussianHMM, observations: np.ndarray) -> np.ndarray:
    """Compute Gaussian log densities from documented HMM parameters.

    ``covars_`` exposes full covariance matrices for every covariance type.
    For diagonal/spherical models, use the direct Gaussian formula to retain
    the numerical behavior of the experiment without private hmmlearn methods.
    """
    means = np.asarray(model.means_, dtype=float)
    covariances = np.asarray(model.covars_, dtype=float)
    if model.covariance_type in {"diag", "spherical"}:
        variances = np.maximum(
            np.diagonal(covariances, axis1=-2, axis2=-1), np.finfo(float).tiny
        )
        # Squaring a large finite difference *before* dividing by an even
        # larger variance overflows unnecessarily (e.g. 1e200**2 / 1e300).
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            standardized = ((observations[:, None, :] - means[None, :, :])
                            / np.sqrt(variances)[None, :, :])
            return -0.5 * (
                observations.shape[1] * np.log(2 * np.pi)
                + np.log(variances).sum(axis=1)[None, :]
                + np.square(standardized).sum(axis=2)
            )
    return np.column_stack([
        np.atleast_1d(multivariate_normal.logpdf(
            observations, mean=means[state], cov=covariances[state]
        ))
        for state in range(model.n_components)
    ])


def _validated_indicators(frame: pd.DataFrame, indicators: tuple[str, ...],
                          *, keep_missing: bool = False) -> pd.DataFrame:
    if not indicators or len(set(indicators)) != len(indicators):
        raise ValueError("HMM indicators must be a nonempty set of distinct columns")
    absent = set(indicators) - set(frame.columns)
    if absent:
        raise ValueError(f"Missing HMM indicators: {sorted(absent)}")
    if "Name" in frame:
        symbols = frame["Name"]
        normalized = normalize_ticker_symbols(
            symbols, context="Fit/infer HMM regimes for one ticker with a valid name at a time",
        )
        if normalized.nunique() != 1:
            raise ValueError("Fit/infer HMM regimes for one ticker with a valid name at a time")
    ordered = frame.copy()
    if "Name" in ordered:
        ordered["Name"] = normalized
    if "Date" in ordered:
        ordered["Date"] = parse_daily_dates(ordered["Date"], context="HMM dates")
        ordered = ordered.sort_values("Date", kind="stable")
    if "Date" in ordered and ordered["Date"].duplicated().any():
        raise ValueError("Duplicate dates for a single ticker")
    validate_session_positions(ordered)
    clean = ordered.dropna(subset=list(indicators)).copy()
    if not np.isfinite(clean[list(indicators)].to_numpy(dtype=float)).all():
        raise ValueError("Indicator values must be finite")
    return ordered if keep_missing else clean


def infer_filtered_regimes(model: GaussianHMM, df: pd.DataFrame,
                           indicators: tuple[str, ...] = DEFAULT_INDICATORS) -> pd.DataFrame:
    """Assign each state's most likely value using only observations so far.

    In particular, do not use HMM ``predict``/Viterbi smoothing: it uses future
    observations to revise earlier states, invalidating historical backtests.
    """
    indicators = tuple(indicators)
    fitted_indicators = getattr(model, "stock_trading_indicator_names_", None)
    if fitted_indicators is not None and indicators != fitted_indicators:
        raise ValueError(
            "HMM inference indicators must match the names and order used during fitting"
        )
    # Work with positional row identifiers: concatenated training/holdout
    # frames can have repeated index labels, which are unsafe for .loc writes.
    rows = df.reset_index(drop=True)
    ordered = _validated_indicators(rows, indicators, keep_missing=True)
    output = rows.copy()
    if "Name" in output:
        output["Name"] = ordered["Name"]
    output["Market_Regime"] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    complete = ordered[list(indicators)].notna().all(axis=1)
    if not complete.any():
        output.index = df.index
        return output
    observations = ordered.loc[complete, list(indicators)].to_numpy(dtype=float)
    emissions = _gaussian_log_emissions(model, observations)
    # A zero prior or transition is impossible, not merely very unlikely.
    # Keep it at -inf in log space instead of assigning artificial mass.
    with np.errstate(divide="ignore"):
        transition = np.log(model.transmat_)
        log_start = np.log(model.startprob_)
    log_alpha = log_start.copy()
    regimes = []
    emission_index = 0
    previous_observed_session = None
    missing_since_observation = False
    for step, (row_index, has_observation) in enumerate(complete.items()):
        # Fitting treats complete contiguous segments as independent HMM
        # sequences. Filtering must use the *same* boundary rule, restarting
        # after missing indicators or missing source-market sessions.
        current_session = (int(ordered.loc[row_index, "Session_Number"])
                           if "Session_Number" in ordered else step)
        if has_observation:
            if previous_observed_session is not None:
                if missing_since_observation or current_session != previous_observed_session + 1:
                    log_alpha = log_start.copy()
                else:
                    log_alpha = logsumexp(log_alpha[:, None] + transition, axis=0)
            log_alpha = log_alpha + emissions[emission_index]
            emission_index += 1
            normalization = logsumexp(log_alpha)
            if not np.isfinite(normalization):
                raise ValueError("HMM filtering produced no finite state probabilities")
            log_alpha = log_alpha - normalization
            regimes.append((row_index, int(np.argmax(log_alpha))))
            previous_observed_session = current_session
            missing_since_observation = False
        elif previous_observed_session is not None:
            missing_since_observation = True
    for row_index, regime in regimes:
        output.loc[row_index, "Market_Regime"] = regime
    output.index = df.index
    return output


def fit_hmm_on_indicators(df: pd.DataFrame,
                          indicators: tuple[str, ...] = DEFAULT_INDICATORS,
                          n_components: int = 3,
                          random_state: int = 42) -> tuple[GaussianHMM, pd.DataFrame]:
    """Fit on caller-supplied training data and infer its filtered regimes.

    Holdout observations must be supplied only to ``infer_filtered_regimes``
    using the fitted training-period model, never to this fitting function.
    """
    indicators = tuple(indicators)
    clean = _validated_indicators(df, indicators, keep_missing=True)
    # Fitting cannot directly accept missing-emission rows. Instead, break
    # complete observations into independent contiguous segments; no fitted
    # transition then crosses a discarded market session.
    if clean[list(indicators)].isna().any().any():
        raise ValueError("HMM fitting requires complete indicator observations")
    if n_components < 1 or len(clean) < max(n_components * 3, 10):
        raise ValueError("Insufficient training observations for the HMM")
    X = clean[list(indicators)].to_numpy(dtype=float)
    model = GaussianHMM(
        n_components=n_components, covariance_type="diag", n_iter=500,
        tol=1e-4, random_state=random_state, min_covar=1e-4,
    )
    if "Session_Number" in clean:
        positions = clean["Session_Number"].tolist()
        boundaries = np.array([0] + [i for i in range(1, len(positions))
                                     if positions[i] != positions[i - 1] + 1])
        lengths = np.diff(np.r_[boundaries, len(clean)]).tolist()
    else:
        lengths = [len(clean)]
    if sum(length - 1 for length in lengths) < n_components:
        raise ValueError("Insufficient consecutive-session transitions for the HMM")
    model.fit(X, lengths=lengths)
    history = model.monitor_.history
    if len(history) >= 2 and abs(history[-1] - history[-2]) >= model.tol:
        raise RuntimeError("HMM fitting did not reach the specified tolerance")
    # GaussianHMM sees a NumPy array, so record the feature identity ourselves.
    # Otherwise, inference can silently treat a reordered column as another one.
    model.stock_trading_indicator_names_ = indicators
    return model, infer_filtered_regimes(model, df, indicators)
