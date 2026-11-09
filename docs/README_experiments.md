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
times are rejected to preserve one observation per trading day. Incomplete or
contradictory bars are excluded without guessing replacement prices. Feature
preparation rejects an input ticker if insufficient valid price history remains
after indicator warm-up, so no ticker is silently omitted from the prepared data.
Validation also rejects a ticker if every one of its price bars is invalid.
Single-ticker selection is applied before validating OHLCV records and dates,
so unrelated symbols cannot cause a selected ticker's run to fail. Literal
symbols such as `NA` remain ticker identifiers instead of being parsed as
missing values, while missing-name placeholders such as `N/A`, `NaN`, and
`NULL` are rejected. Indicator, HMM, and trading inputs normalize whitespace
and case before grouping or checking ticker identity.

The pipeline computes indicators on each ticker's chronological series independently:

| Indicator | Definition / implementation |
| --- | --- |
| MA_20 | Twenty-observation simple moving average of closing price |
| ATR | Fourteen-observation moving average of true range, using prior close |
| RSI_14 | Fourteen-observation rolling gain/loss RSI |
| MACD | Twelve-period EMA minus twenty-six-period EMA, plus nine-period signal EMA |
| Bollinger Bands | Twenty-period mean ± two rolling standard deviations |
| Return / Return_Lag1 | Daily fractional close-to-close change and its previous value |
| Close_Lag1 | Previous observed closing price for the same ticker |
| Intraday_Return | (Close − Open) / Open |
| Daily_Range | (High − Low) / Open |
| MACD_Pct | MACD divided by current Close |

`Target_Date` is the date of that ticker's next available observation. `Trend=1` when its closing price exceeds the current close, otherwise `Trend=0`. A final row without a subsequent observation is excluded, along with indicator warm-up rows.

The generated feature table is saved to `data/generated/stock_features.csv` by default. Raw stock prices retain their original units; transformations learned from data are handled in the model-training step.

## 2. Bayesian-network experiment

```bash
python scripts/model_training.py
# Optional subset:
python scripts/model_training.py --ticker AAPL --num-bins 4 --output data/generated/aapl_model_metrics.json
```

The experiment proceeds as follows:

1. **Prepare:** read validated raw observations and construct ticker-safe features and next-observation labels.
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

These are fixed, interpretable directed graphs. The scale-independent price features avoid treating a higher nominal stock price as a different price-pattern regime. Every feature is a parent of `Trend`; the CPT for `Trend` represents the conditional probability of next-observation upward movement given the binned feature values.

**Metrics and outputs**

The training script prints and saves JSON metrics under `data/generated/model_metrics.json`, including accuracy, balanced accuracy, precision, recall, F1, ROC AUC, validation scores, selected prior strength, date boundaries, sample counts, and per-partition ticker counts (`tickers_per_partition`).

For the included combined source, the evaluation periods are:

| Partition | Observations | Date boundary |
| --- | ---: | --- |
| Training | 55,334 | Through 2013-04-03 |
| Validation | 18,472 | 2013-04-04 through 2015-08-14 |
| Test | 18,513 | 2015-08-17 through 2017-12-28 |

The [model evaluation notebook](../notebooks/Model_Training_Evaluation.ipynb) runs the same experiment and displays both models' metrics side by side.

## 3. Gaussian HMM regime experiment

Open [HMM_and_RL_Experiments.ipynb](../notebooks/HMM_and_RL_Experiments.ipynb).

The notebook selects **AAPL** for a contiguous trading sequence, computes indicators, and partitions observations chronologically. A three-state diagonal-covariance Gaussian HMM is fitted to the training period's `MA_20`, `RSI_14`, and `MACD`.
HMM fitting requires complete indicator observations. The fit rejects missing
training indicators instead of compressing nonadjacent dates into artificial
single-step transitions; missing observations can still be handled during
forward inference.

`infer_filtered_regimes` computes forward probabilities from the fitted HMM emission and transition parameters:

$$
P(z_t\mid x_{1:t}) \propto P(x_t\mid z_t)\sum_{z_{t-1}}
P(z_t\mid z_{t-1})P(z_{t-1}\mid x_{1:t-1}).
$$

Each reported regime therefore depends on observations through time `t`, with no dependence on later observations. The HMM is fitted on all causal indicator observations through the training cutoff, including the final training day whose next-day label is excluded from supervised training. Forward filtering then uses the entire indicator history, including both supervised-split boundary days, while the Q-learning training sequence retains the label-purged training rows. No validation or test observation contributes to HMM parameter fitting. Regime identifiers are statistical latent-state labels and can change in meaning between independently fitted models.
Gaussian emission likelihoods are calculated from the fitted model's public
mean and covariance parameters, without relying on internal hmmlearn methods.
Impossible initial states and transitions retain zero probability. When an
indicator observation is missing, the filter advances the latent state using
the transition matrix without assigning a regime to that incomplete row.

## 4. Q-learning experiment

The same notebook fits three-bin quantiles on the training period's daily return and RSI, combines those two binned features with the filtered HMM regime and the current position, and trains the tabular policy on the earlier single-ticker training sequence for 75 episodes.

The `SimpleTradingEnv` has three actions:

- **Buy:** enter one-share long position when flat; unavailable while already long.
- **Sell:** exit the position when long; unavailable while flat.
- **Hold:** leave the position unchanged.

Q-learning state components must be integers. Both the trading environment and
the agent reject fractional, string, and boolean components instead of
silently converting them to bins. Q-learning updates also enforce action-to-position
consistency, including at the terminal step. The trading environment refuses
to advance its state if a price calculation yields a nonfinite reward.

Decisions made after observing a daily close are filled at the next trading session's open. Holdings existing before an action remain exposed to the overnight price change. The reward for a step is:

$$
r_t = \mathrm{position}_{t}(\mathrm{Open}_{t+1}-\mathrm{Close}_t)
+ \mathrm{position}_{t+1}(\mathrm{Close}_{t+1}-\mathrm{Open}_{t+1})
- \mathrm{transaction\_cost}\times\mathrm{Open}_{t+1}\times\mathbf{1}[\mathrm{trade}].
$$

Q-values are updated using the discounted next-state value for non-terminal steps. At the final step, the next-state bootstrap value is zero. Terminal next-position values and the `done` flag are validated, and updates that would make a Q-value nonfinite are rejected without changing the table. To score the holdout period, the notebook disables both exploration and Q-table updates and compares one-share dollar profit/loss with a next-open buy-and-hold entry.

An optional `max_steps` argument permits nonnegative integer episode limits;
invalid values are rejected before resetting the environment.

Both strategies are marked to market at the last observed close. Positions
still held at that close remain open; no hypothetical final sale or exit fee
is included in either strategy's reported P/L.

The implementation is intentionally a transparent research simulator; it can be extended with portfolio sizing, risk controls, and realistic order execution.

## 5. Verification

Run the regression tests:

```bash
python -m pytest tests -q
```

The test suite covers finite OHLCV and date validation, per-ticker labels and indicators, price-scale invariance, chronological target boundaries, training-only binning, Bayesian CPT predictions, package imports, prefix-invariant HMM state filtering, legal trading actions, next-open rewards and transaction costs, immutable holdout evaluation, terminal Q-learning updates, complete synthetic-data experiments, and saved notebook outputs.

The same checks run automatically through [GitHub Actions](../.github/workflows/python-checks.yml) on pushes and pull requests. To regenerate the GitHub-visible notebook tables and figures, run:

```bash
python -m scripts.refresh_notebooks
```

`python -m scripts.refresh_notebooks --check` reruns the notebooks without modifying them. The in-process runner captures stdout, stderr, explicit `display()` calls, final-expression results (including rich MIME representations), and Matplotlib PNGs in their execution order. It compares saved output order and text exactly, fully decodes saved PNGs, and detects both broad and localized changes to generated charts. Jupyter's `%matplotlib inline` directive is removed only when it is an actual standalone magic command, never when it appears inside a string. Python future-import compiler flags persist from one cell to the next.
The `reproduce_notebook(path, check=True)` Python helper resolves relative data
paths from the notebook's directory and restores the caller's working directory
and `sys.path` import search path.
It excludes the caller's existing Matplotlib figures from capture and cleanup,
and closes newly created figures when execution fails, preventing stale-chart
contamination.

## References

- [pgmpy: probabilistic graphical models](https://pgmpy.org/)
- [hmmlearn: Gaussian HMMs](https://hmmlearn.readthedocs.io/)
- [scikit-learn: metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Source dataset: Stock Time Series 20050101 to 20171231](https://www.kaggle.com/datasets/szrlee/stock-time-series-20050101-to-20171231)
