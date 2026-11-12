# Experiment Reproduction Guide

This guide documents the reproducible Bayesian-network, HMM, and Q-learning experiments in the [Probabilistic Stock Trading Agent](../README.md).

## Environment

Use Python 3.10, 3.11, or 3.12 from the repository root:

```bash
python -m venv .venv
# Activate .venv for your operating system.
python -m pip install -r requirements-dev.txt
```

For interactive notebook use, optionally install JupyterLab in the activated
environment with `python -m pip install jupyterlab`, then launch `jupyter lab`
from the repository root. JupyterLab is not required for scripted reproduction.

To reproduce the exact dependency versions used for the published numerical
results on Python 3.12/Linux, install `requirements-repro-py312.txt` instead.
That file includes pinned transitive dependencies; the normal requirements
files retain broader version compatibility.

The combined historical source is `data/raw/all_stocks_2006-01-01_to_2018-01-01.csv`, containing daily OHLCV observations and ticker identifiers.

## 1. Feature preparation

```bash
python scripts/data_processing.py
# Single-ticker example:
python scripts/data_processing.py --ticker AAPL --output data/generated/aapl_stock_features.csv
```

The source CSV is validated for required columns, explicit ISO year-first
calendar dates (including non-zero-padded months/days), finite numeric values,
positive OHLC prices, coherent daily high/low ranges, and unique `(Name, Date)`
observations. Relative dates, numeric timestamps, timezones, and intraday
times are rejected to preserve one observation per trading day. To reject
NumPy dates with only year, month, or week precision, pass the original
unconverted record mappings to `validate_raw_bar_records(records)` in
`scripts/data_processing.py`; this checks each original scalar before pandas
can assign an invented day. Raw ingestion also rejects NumPy sub-day timestamp
units, Python `datetime` values, and pandas `Timestamp` objects even when their
clock reads midnight. Once an ordinary DataFrame constructor has
converted a coarse NumPy date into a timestamp, its original precision is
irrecoverable. To avoid silently accepting this ambiguity, `validate_bars()`
rejects existing pandas datetime64/Timestamp dates unless the caller explicitly
attests to verified source precision with `verified_datetime_dates=True`.
The same explicit attestation applies to `prepare_features()` and
`prepare_indicator_history()` when their input dates are pandas timestamps,
including a DataFrame that `load_data()` has already validated. For example:

```python
from scripts.data_processing import DEFAULT_RAW_DATA, load_data, prepare_features

raw = load_data(DEFAULT_RAW_DATA)
features = prepare_features(raw, verified_datetime_dates=True)
```

Keep the opt-in disabled when the original dates have not been checked.
Both preparation entry points now enforce the raw validator's date-provenance
policy. The standalone NYSE calendar validator treats missing (`NaT`) dates
as invalid instead of identifying them as historical sessions. It also rejects
pre-1971 Sundays and Saturdays after May 24, 1952; earlier sessions still
require independently verified market ordinals.
Incomplete or
contradictory bars are excluded without guessing replacement prices. Feature
preparation rejects an input ticker if insufficient valid price history remains
after indicator warm-up, so no ticker is silently omitted from the prepared data.
Validation also rejects a ticker if every one of its price bars is invalid.
Single-ticker selection is applied before validating OHLCV records and dates,
so unrelated symbols cannot cause a selected ticker's run to fail. Every
bundled per-ticker CSV uses the full 31-stock dataset's reference calendar,
assembled from weekday dates with valid ticker identifiers. Incomplete price
bars still establish their source date as a potential market session, so an
invalid-only date is excluded from the price sample without compressing time.
Malformed dates and invalid ticker identifiers cannot contaminate the
reference calendar. Therefore its labels
and session positions agree with the same ticker selected from the combined
source, even when individual stock files omit sessions. Custom CSVs, including
new files placed in `data/raw/`, are not mistaken for bundled originals.
Without `Session_Number`, date-only U.S. equity inputs from 1971 onward use an
NYSE/Nasdaq holiday calendar, including historical presidential Election Day closures
from 1972 to 1980 and exceptional full-day closures, rather than counting
holidays as trading sessions. The MLK holiday rule begins in 1998. Dates before
1971 require independently verified `Session_Number` metadata: the older NYSE
holiday schedule, including the 1968 Wednesday closures, cannot be accurately
inferred using the modern weekday calendar. An optional `Exchange` column groups stocks by their declared market;
recognized exchange schedules reveal missing trading sessions while preventing
foreign-exchange dates from contaminating NYSE history. The aliases `NYSE`, `XNYS`, `NASDAQ`, and `XNAS` use the U.S.
calendar, including when `Exchange` is omitted. The aliases `LSE` and `XLON`
use the London Stock Exchange session calendar, while `TSX` and `XTSE`
use Toronto's calendar (including Boxing Day closures).
Observed dates outside any recognized market's actual calendar are rejected,
with or without supplied ordinals. Unrecognized explicit exchange labels require verified `Session_Number`
metadata; automatic inference is disabled for those markets.
From 1971 onward, supplied ordinals for every recognized exchange must
advance by the exact number of scheduled sessions between observations,
preventing missing trading days from being concealed by compressed positions.
For custom CSV ticker selection, impossible closed-market observations on a
recognized exchange belonging only to unselected peers are ignored as calendar
evidence; valid same-exchange dates still reveal missing sessions in the selected
ticker.
Without `Exchange`, U.S. market holidays are enforced, including for
caller-supplied `Session_Number` values. For exact unsupported exchange
calendars, declare an exchange and provide verified `Session_Number`. Historical NYSE
Saturday trading is supported through its last actual Saturday session on
1952-05-24, but only with independently verified `Session_Number` metadata;
later Saturdays and all Sundays are rejected. Invalid source identifiers
do not contribute dates to the calendar. Literal symbols such as `NA` remain ticker identifiers instead
of being parsed as missing values, while missing-name placeholders such as
`N/A`, `NaN`, `NULL`, `<NA>`, and malformed symbols are rejected by all four
input paths: raw data validation, direct
indicator calculation, HMM fitting/inference, and trading environments.
All paths normalize whitespace and case before checking ticker identity.
Direct multi-ticker indicators and trading-episode segmentation establish
the full input's exchange session positions before isolating individual stocks.
Trading episodes reject duplicate ticker/date rows and sort by parsed dates,
including valid non-zero-padded ISO strings.
In the included 93,612-row source, 93,435 bars pass OHLCV validation;
177 incomplete or inconsistent records are excluded without imputation.
Before exclusion, validation assigns `Session_Number` from the bundled
observed-session calendar or the recognized exchange's scheduled calendar.
Unknown explicit exchanges cannot infer sessions and must supply independently
verified ordinals to establish adjacency. Source-wide
valid dates anchor ordinals before ticker selection, so the same custom stock
has identical session numbers in full and selected CSV loads. Positions survive
price exclusion and subsetting, allowing indicators, HMM boundaries, and the
trading simulator to detect missing sessions. Supplied positions must have an
increasing one-to-one correspondence with valid daily market dates within each
ticker or declared exchange; different exchanges can use different ordinals
for the same date. Adjacent ordinals are compared using exact Python integers
to preserve even large valid 64-bit values during target construction.
Direct indicator calls validate positive, finite, internally coherent OHLC
prices, normalize accepted numeric strings, and check supplied session positions;
Volume remains optional for that standalone API but is checked for finite
nonnegative values when supplied. Boolean OHLCV values are rejected instead
of being silently converted to numeric ones or zeros.

The pipeline computes indicators on each ticker's chronological series independently:

| Indicator | Definition / implementation |
| --- | --- |
| MA_20 | Twenty-observation simple moving average of closing price |
| ATR | Fourteen-observation moving average of true range using the prior close; interrupted by missing sessions |
| RSI_14 | Fourteen-consecutive-session rolling gain/loss RSI; missing bars interrupt the window |
| MACD | Twelve-period EMA minus twenty-six-period EMA, plus nine-period signal EMA |
| Bollinger Bands | Twenty-period mean ± two rolling standard deviations |
| Return / Return_Lag1 | Fractional change between consecutive source-market sessions and its preceding value; unavailable across missing bars |
| Close_Lag1 | Previous observed closing price for the same ticker |
| Intraday_Return | (Close − Open) / Open |
| Daily_Range | (High − Low) / Open |
| MACD_Pct | MACD divided by current Close |

`Target_Date` is the date of that ticker's next **source-market session**, provided
both that session and its closing price are valid. `Trend=1` when its closing price
exceeds the current close, otherwise `Trend=0`. A row without a valid immediately
following market-session observation is excluded, along with indicator warm-up rows.
The target never mixes one-session and multi-session horizons. Moving averages,
EMA/MACD, and Bollinger Bands retain their explicitly observation-based windows
without invented prices.

The generated feature table is saved to `data/generated/stock_features.csv` by
default. Feature preparation requires only `Target_Date` and the requested model
features, not unrelated indicators. The default four-feature enhanced sample
contains 92,284 eligible observations; passing
`features=BASELINE_FEATURES` retains all 93,028 baseline-eligible rows. The
published two-model comparison deliberately uses the enhanced-eligible rows
for both models so the test samples match. Raw stock prices retain their
original units; transformations learned from data are handled in the
model-training step.

The [EDA notebook](../notebooks/EDA_Preprocessing.ipynb) displays the raw
missingness and the validator's retained/excluded row counts. Its histogram
uses daily percentage returns from the exact same bundled market-session
calendar as the modeling pipeline (93,228 available close-to-close returns),
and its box plots compare those percentage returns within each ticker rather
than mixing stocks' nominal closing prices. Returns spanning invalid source
sessions are omitted from those comparisons. Its CSV path resolves correctly
from both the repository root and the notebook directory.

## 2. Bayesian-network experiment

```bash
python scripts/model_training.py
# Optional subset:
python scripts/model_training.py --ticker AAPL --num-bins 4 --output data/generated/aapl_model_metrics.json
```

The optional AAPL-only experiment calculates its own chronological boundaries:
training through 2013-04-01 and validation through 2015-08-17. The published
31-ticker experiment uses 2013-03-28 and 2015-08-13 respectively. These
different date windows mean the single-ticker metrics are not an identical-
period comparison against the full-source experiment.

The experiment proceeds as follows:

1. **Prepare:** read validated raw observations and construct ticker-safe features and next-session labels.
2. **Partition by date:** verify that every target date follows its observation date, then split the combined market calendar into 60% training, 20% validation, and 20% test. Rows whose target date crosses a training or validation boundary are excluded from those earlier partitions.
   Every ticker must retain observations in all three partitions; missing ticker
   coverage produces an explicit error naming the affected symbols and periods.
3. **Fit discretization:** calculate quantile edges on training observations; apply exactly those edges to validation features.
4. **Fit candidates:** build a discrete Bayesian network with feature-to-`Trend` edges for each feature set. Estimate CPTs with a BDeu prior and candidate equivalent sample sizes of `5, 10, 15, 20`.
5. **Select:** choose prior strength using validation balanced accuracy, then accuracy as a tie-breaker.
6. **Refit:** fit quantile edges and network parameters on the combined training and validation partitions using the selected configuration.
7. **Report:** evaluate this final model on the later test dates, alongside the training-majority benchmark.

**Model feature sets**

- Baseline: `Return`, `Intraday_Return`, `Daily_Range`
- Enhanced: `Return`, `Intraday_Return`, `Daily_Range`, `MACD_Pct`

These are fixed, interpretable directed graphs. The scale-independent price features avoid treating a higher nominal stock price as a different price-pattern regime. Every feature is a parent of `Trend`; the CPT for `Trend` represents the conditional probability of next-market-session upward movement given the binned feature values.

**Metrics and outputs**

The training script prints and saves JSON metrics under `data/generated/model_metrics.json`, including accuracy, balanced accuracy, precision, recall, F1, ROC AUC, validation scores, selected prior strength, date boundaries, sample counts, and per-partition ticker counts (`tickers_per_partition`).

For the included combined source, the evaluation periods are:

| Partition | Observations | Date boundary |
| --- | ---: | --- |
| Training | 55,202 | Through 2013-03-28 |
| Validation | 18,526 | 2013-03-29 through 2015-08-13 |
| Test | 18,494 | 2015-08-14 through 2017-12-28 |

The [model evaluation notebook](../notebooks/Model_Training_Evaluation.ipynb) runs the same experiment and displays both models' metrics side by side.

### Execution-aligned direction and additional holdout windows

The original `Trend` target is `Close[t+1] > Close[t]`. Its positive label
includes overnight movement before the next-open fill used by the trading
simulator. The optional `next_open_to_close` target instead evaluates
`Close[t+1] > Open[t+1]` while keeping the decision features at the completed
current session. This is a distinct experimental question; the default
`close_to_close` target and published results remain unchanged.

```bash
python -m scripts.model_training --target next_open_to_close --output data/generated/executable_target_metrics.json
python -m scripts.walk_forward_evaluation --target next_open_to_close --output data/generated/executable_walk_forward_metrics.json
```

The walk-forward command performs five disjoint historical test windows.
In fold one, the first 40% of available feature dates are training data,
the next 10% validation, and the following 10% test. Each subsequent fold
advances all three boundaries by 10%, retaining an expanding training history.
No test-period label is used for that fold's bin fitting or BDeu selection;
labels crossing a period boundary are purged. Once a prior test period is
historical, its data may be used to train a subsequent fold. The resulting
aggregate covers only the nonoverlapping forward test windows, and the fixed
experimental design is never selected using their outcomes.

The report includes each fold's metrics and sample sizes, pooled out-of-sample
metrics, and 95% paired bootstrap intervals for accuracy *differences* versus
the training-majority benchmark. Bootstrap units are blocks of at most
ten consecutive **actual market sessions**: a missing session starts a new
block, and all tickers observed on the same date stay together. Block
resampling partially accounts for serial and cross-sectional dependence.
The full five-fold results for both targets are checked in CI against
[`expected_walk_forward.json`](expected_walk_forward.json) using the pinned
Python 3.12 environment. These descriptive
intervals do not imply statistical independence across long market regimes or
prove future trading profitability. Change the block length, fold count, or
bootstrap sample count using `--block-sessions`, `--folds`, and
`--bootstrap-samples`. The helper checks binary classification labels, finite
prediction probabilities in [0, 1], and at least two distinct chronological
date blocks. Intraday timestamps and timezone-bearing dates are rejected;
each observation must represent one daily market date. Fold feasibility is
checked before fitting any Bayesian model, so a short holdout or oversized
block setting fails explicitly instead of presenting a degenerate 95% interval.

## 3. Gaussian HMM regime experiment

Open [HMM_and_RL_Experiments.ipynb](../notebooks/HMM_and_RL_Experiments.ipynb).

The notebook selects **AAPL**, computes indicators, and uses the same global
market-date train/validation/test boundaries as the Bayesian experiment.
A three-state diagonal-covariance Gaussian HMM is fitted to the training
period's `MA_20`, `RSI_14`, and `MACD`. Direct HMM fitting requires complete
indicator observations and explicitly raises on missing emissions. The notebook
first drops incomplete rows from its fitting matrix; `fit_hmm_on_indicators`
then supplies contiguous-sequence lengths so no learned
transition crosses an excluded source session or missing indicator period.
Forward filtering applies the same boundary policy: each complete contiguous
segment begins with the HMM's initial-state probabilities.

`infer_filtered_regimes` computes forward probabilities from the fitted HMM emission and transition parameters:

$$
P(z_t\mid x_{1:t}) \propto P(x_t\mid z_t)\sum_{z_{t-1}}
P(z_t\mid z_{t-1})P(z_{t-1}\mid x_{1:t-1}).
$$

Each reported regime therefore depends on observations through time `t`, with no dependence on later observations. The HMM is fitted on all *complete* causal indicator observations through the training cutoff, including the final training day whose next-day label is excluded from supervised training. Forward filtering uses the indicator history through the final source session, including both supervised-split boundary days, while the Q-learning training sequence retains the label-purged training rows. No validation or test observation contributes to HMM parameter fitting. Regime identifiers are statistical latent-state labels and can change in meaning between independently fitted models.
Gaussian emission likelihoods are calculated from the fitted model's public
mean and covariance parameters, without relying on internal hmmlearn methods.
The fitting helper stores its ordered indicator names on the model and
inference rejects a reordered or substituted indicator tuple; equal feature
counts do not imply interchangeable HMM dimensions.
Impossible initial states and transitions retain zero probability. When an
indicator observation is missing, the filter leaves that row's regime undefined
and starts a new independent sequence at the next complete observation.
`Session_Number` also identifies excluded source sessions, causing the same
restart rather than an artificial adjacent transition. This is consistent with
the independent-segment fitting likelihood and avoids loops proportional to
the size of an arbitrary session gap. Gaussian log densities standardize
residuals before squaring to avoid avoidable intermediate overflow.

## 4. Q-learning experiment

The same notebook fits three-bin quantiles on AAPL's training-period consecutive-session return and RSI, combines those two binned features with the filtered HMM regime and the current position, and trains the tabular policy for 75 passes across separate contiguous training segments. A gap in feature-eligible training **or test** rows ends an episode instead of allowing execution several market sessions later. Each segment starts flat; its final position is marked to market without an assumed sale. The frozen-policy test P/L and next-open buy-and-hold comparator sum comparable per-segment results, excluding isolated one-bar fragments where no trading step is possible. For the bundled AAPL holdout, all test bars are contiguous and the published totals remain unchanged. These columns, `Return_binned`, `RSI_14_binned`, and `Market_Regime`, also form the environment's default state schema when present.
The training chart's horizontal axis counts **75 training passes**, each
aggregating the rewards of all segment episodes in that pass; it does not count
individual segment episodes.

`contiguous_trading_segments` partitions inputs by normalized ticker before
detecting missing sessions. It never constructs an episode containing bars
from different stocks, including when their local session ordinals overlap.

The `SimpleTradingEnv` has three actions:

- **Buy:** enter one-share long position when flat; unavailable while already long.
- **Sell:** exit the position when long; unavailable while flat.
- **Hold:** leave the position unchanged.

Q-learning state components must be integers. Both the trading environment and
the agent reject fractional, string, and boolean components instead of
silently converting them to bins. Q-learning updates require equal current and
next-state dimensions, and enforce action-to-position consistency, including
at the terminal step. The trading environment refuses to advance its state if
a price calculation yields a nonfinite reward; episode aggregation also rejects
nonfinite cumulative returns.

Decisions made after observing a daily close are filled at the next trading session's open. The simulator checks consecutiveness when supplied with source-market ordinals; it rejects compressed multi-session steps. Holdings existing before an action remain exposed to the overnight price change. The reward for a step is:

$$
r_t = \mathrm{position}_{t}(\mathrm{Open}_{t+1}-\mathrm{Close}_t)
+ \mathrm{position}_{t+1}(\mathrm{Close}_{t+1}-\mathrm{Open}_{t+1})
- \mathrm{transaction\_cost}\times\mathrm{Open}_{t+1}\times\mathbf{1}[\mathrm{trade}].
$$

Q-values are updated using the discounted next-state value for non-terminal steps. At the final step, the next-state bootstrap value is zero. Terminal next-position values and the `done` flag are validated, and updates that would make a Q-value nonfinite are rejected without changing the table. To score the holdout period, the notebook disables both exploration and Q-table updates and compares one-share dollar profit/loss with a next-open buy-and-hold entry.

An optional `max_steps` argument permits nonnegative integer episode limits;
invalid values are rejected before resetting the environment.

Both strategies are marked to market at the last validated source close,
including the final session without a following prediction target. Positions
still held at that close remain open; no hypothetical final sale or exit fee
is included in either strategy's reported P/L.

With the corrected feature sample, the AAPL frozen policy produced **+$29.10**
in one-share holdout P/L, versus **+$53.07** from the next-open buy-and-hold
comparator. The implementation is intentionally a transparent research
simulator; it can be extended with portfolio sizing, risk controls, and realistic
order execution.

The same notebook now evaluates that *unchanged frozen policy* under 0, 10,
and 25 basis points of next-open adverse slippage on entries and exits.
`SimpleTradingEnv(..., slippage_bps=...)` applies an unfavorable fill price,
charges the fractional transaction fee on that fill, and computes the one-share
mark-to-market reward without future-price information. The
`episode_metrics()` method reports cumulative P/L, realized transaction costs,
slippage, trade count, maximum marked-to-market drawdown in dollars, fractions
of intraday and overnight intervals invested, and terminal position. If a position remains open,
the summary does not charge an imaginary exit. Multiple disconnected source
segments are analyzed as separate episodes, including separately measured
drawdowns; these are not interpreted as a continuously invested portfolio.
The simulation remains deliberately simplified: it does not model order-book
spreads, liquidity constraints, changing bid/ask quotes, market impact,
financing, or capital-based risk management.

## 5. Verification

Run the regression tests:

```bash
python -m pytest tests -q
python -m ruff check scripts tests --isolated --select E4,E7,E9,F,B
```

The test suite covers finite OHLCV and date validation, global market-session
uniqueness, all bundled individual-stock versus combined-source feature parity,
invalid unrelated dates, direct weekday-gap handling, model-specific feature
eligibility, invalid priors and unknown categorical evidence, numeric-string
input normalization, ticker-name formats, per-ticker indicators, price-scale
invariance, chronological target boundaries, training-only binning, Bayesian
CPT predictions, package imports, unsigned HMM session validation,
segment-consistent prefix-invariant HMM filtering, legal trading actions,
consecutive trading steps, next-open rewards and transaction costs, immutable
holdout evaluation, terminal Q-learning updates, notebook environment isolation,
complete synthetic-data experiments, and saved notebook outputs. Additional
regressions cover exchange holiday closures, invalid-only bundled dates,
unrelated malformed records, custom CSV detection, weekend metadata,
boolean OHLCV inputs, source-calendar cache invalidation, and segmented
holdout trading. The extra executable-target evaluation, walk-forward fold
separation, block-bootstrap estimates, and slippage/drawdown calculations
also have dedicated regression checks. Edge-case tests additionally cover
mixed-exchange calendars, independent per-ticker ordinals, the start of the
NYSE's MLK holiday in 1998, invalid bootstrap scores, and insufficient
bootstrap blocks. Further regressions cover exact 64-bit session arithmetic,
stable custom CSV ordinals across ticker selection, declared exchange calendars,
pre-1971 historical-calendar rejection without verified metadata, single-stock
trading episodes, daily-only bootstrap inputs, and preflight fold viability.
The historical-session and direct-API cases additionally cover documented
1972–1994 NYSE full-day closures, fabricated holiday observations on declared
NYSE data, exchange-shared indicators and episodes, duplicate ticker/date
records, non-zero-padded date ordering, and empty adjacency input.

The same checks run automatically through [GitHub Actions](../.github/workflows/python-checks.yml) on pushes and pull requests. To regenerate the GitHub-visible notebook tables and figures, run:

```bash
python -m scripts.refresh_notebooks
```

The saved notebook results are certified against the pinned **Python 3.12 /
Ubuntu 24.04** environment. The 3.10 and 3.11 checks certify source code,
regression tests, lint, and formatting, but not byte-for-byte notebook output.

`python -m scripts.refresh_notebooks --check` reruns the notebooks without modifying them. The in-process runner captures stdout, stderr, explicit `display()` calls, final-expression results (including rich MIME representations), and Matplotlib PNGs in their execution order. It compares saved output order and text exactly, fully decodes saved PNGs, and detects both broad and localized changes to generated charts. Jupyter's `%matplotlib inline` directive is removed only when it is an actual standalone magic command, never when it appears inside a string. Python future-import compiler flags persist from one cell to the next.
The `reproduce_notebook(path, check=True)` Python helper resolves relative data
paths from the notebook's directory and restores the caller's working directory,
`sys.path` import search path, and environment variables even when a cell fails.
It excludes the caller's existing Matplotlib figures from capture and cleanup,
and closes newly created figures when execution fails, preventing stale-chart
contamination.

## References

- [pgmpy: probabilistic graphical models](https://pgmpy.org/)
- [hmmlearn: Gaussian HMMs](https://hmmlearn.readthedocs.io/)
- [scikit-learn: metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Source dataset: Stock Time Series 20050101 to 20171231](https://www.kaggle.com/datasets/szrlee/stock-time-series-20050101-to-20171231)

**Exchange and date-source safeguards.** NYSE/XNYS, NASDAQ/XNAS, LSE/XLON, and
TSX/XTSE observations use their corresponding trading calendars; inferred and
supplied `Session_Number` values must preserve actual session gaps. Unknown
explicit exchange identifiers must provide independently verified
`Session_Number` values, rather than inheriting fictional sessions from the
U.S. calendar. Unlabeled equity records use the U.S. calendar. Direct `session_positions`, `validate_session_positions`,
`add_technical_indicators`, `SimpleTradingEnv`,
`contiguous_trading_segments`, and HMM entry points require an explicit
`verified_datetime_dates=True` attestation when supplied already-coerced
pandas timestamps without an existing verified-date marker.
Explicit calendar bounds must encompass all observed dates: out-of-range
bounds raise a validation error instead of producing negative positions. Results of `validate_bars` carry their verified-date marker
through normal pandas frame operations; raw ISO date strings remain supported.
Do not attest to converted month/year-only dates without inspecting the
original source precision.
