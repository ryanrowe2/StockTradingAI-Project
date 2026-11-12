"""Single-symbol, long-or-flat Q-learning trading research environment.

The agent can buy one share, sell its existing share, or hold. Decisions based
on a completed daily bar execute at the following session's opening price.
Every step is marked to market, with transaction costs and terminal handling.
This is an experimental simulator, not an execution or profitability claim.
"""

from __future__ import annotations

import math
import random
from itertools import pairwise
from numbers import Integral, Real

import numpy as np
import pandas as pd

if __package__:
    from .date_validation import (
        mark_verified_daily_dates,
        parse_daily_dates,
        require_verified_daily_date_source,
    )
    from .session_validation import (
        adjacent_session_flags,
        session_positions,
        validate_session_positions,
    )
    from .ticker_validation import normalize_ticker_symbols
    from .numeric_validation import real_float_array
else:
    from date_validation import (
        mark_verified_daily_dates,
        parse_daily_dates,
        require_verified_daily_date_source,
    )
    from session_validation import (
        adjacent_session_flags,
        session_positions,
        validate_session_positions,
    )
    from ticker_validation import normalize_ticker_symbols
    from numeric_validation import real_float_array


class SimpleTradingEnv:
    """Sequential one-share simulator with next-session-open execution."""

    DEFAULT_STATE_COLUMNS = ("Return_binned", "RSI_14_binned", "Market_Regime")

    def __init__(
        self,
        data: pd.DataFrame,
        transaction_cost: float = 0.001,
        state_columns: tuple[str, ...] | None = None,
        slippage_bps: float = 0.0,
        *,
        verified_datetime_dates: bool = False,
    ):
        if not {"Open", "Close"}.issubset(data.columns) or len(data) < 2:
            raise ValueError(
                "The trading environment needs at least two daily open/close bars"
            )
        if "Name" in data:
            symbols = data["Name"]
            normalized_symbols = normalize_ticker_symbols(
                symbols,
                context="A trading episode must contain exactly one ticker with a valid name",
            )
            if normalized_symbols.nunique() != 1:
                raise ValueError(
                    "A trading episode must contain exactly one ticker with a valid name"
                )
        if (
            not isinstance(transaction_cost, Real)
            or isinstance(transaction_cost, (bool, np.bool_))
            or not np.isfinite(transaction_cost)
            or not 0 <= transaction_cost < 1
        ):
            raise ValueError("transaction_cost must be a fraction between 0 and 1")
        if (
            not isinstance(slippage_bps, Real)
            or isinstance(slippage_bps, (bool, np.bool_))
            or not np.isfinite(slippage_bps)
            or not 0 <= slippage_bps < 10_000
        ):
            raise ValueError(
                "slippage_bps must be finite basis points from 0 to less than 10000"
            )
        require_verified_daily_date_source(
            data, verified_datetime_dates=verified_datetime_dates
        )
        self.data = data.copy()
        if "Name" in self.data:
            self.data["Name"] = normalized_symbols
        if "Date" in self.data:
            self.data["Date"] = parse_daily_dates(
                self.data["Date"], context="Trading dates"
            )
            mark_verified_daily_dates(self.data, columns=("Date",))
            self.data = self.data.sort_values("Date", kind="stable")
        self.data = self.data.reset_index(drop=True)
        if "Date" in self.data and self.data["Date"].duplicated().any():
            raise ValueError("Duplicate trading dates")
        validate_session_positions(self.data, require_consecutive=True)
        positions = session_positions(self.data).tolist()
        if any(
            next_position != position + 1
            for position, next_position in pairwise(positions)
        ):
            raise ValueError(
                "Trading observations must occupy consecutive market sessions"
            )
        if (
            self.data[["Open", "Close"]]
            .apply(
                lambda column: column.map(
                    lambda value: isinstance(value, (bool, np.bool_))
                )
            )
            .any()
            .any()
        ):
            raise ValueError("Open/close prices must be numeric prices, not booleans")
        prices = real_float_array(
            self.data[["Open", "Close"]], context="Open/close prices"
        )
        if np.any(prices <= 0):
            raise ValueError("Open/close prices must be finite and positive")
        if state_columns is None:
            provided = set(self.DEFAULT_STATE_COLUMNS).intersection(self.data.columns)
            if provided and len(provided) != len(self.DEFAULT_STATE_COLUMNS):
                raise ValueError(
                    "Supply all default state columns, or specify state_columns"
                )
            state_columns = self.DEFAULT_STATE_COLUMNS if provided else ()
        if len(set(state_columns)) != len(state_columns):
            raise ValueError("State columns must be unique")
        missing = set(state_columns) - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing trading state columns: {sorted(missing)}")
        if state_columns:
            state_values = self.data[list(state_columns)].to_numpy(dtype=object)
            if any(
                not isinstance(value, Integral) or isinstance(value, (bool, np.bool_))
                for value in state_values.flat
            ):
                raise ValueError(
                    "Trading state columns must contain finite integer states"
                )
        self.state_columns = tuple(state_columns)
        self.transaction_cost = float(transaction_cost)
        self.slippage_bps = float(slippage_bps)
        self.reset()

    def reset(self) -> tuple[int, ...]:
        self.index = 0
        self.position = 0
        self.done = False
        self.total_pnl = 0.0
        self.total_fees = 0.0
        self.total_slippage = 0.0
        self.trade_count = 0
        self.invested_steps = 0
        self.overnight_invested_steps = 0
        self.equity_curve = [0.0]
        return self.get_state()

    def get_state(self) -> tuple[int, ...]:
        # A mixed price/state row may be coerced to float64 by pandas,
        # corrupting exact int64 / uint64 state values above 2**53.
        return tuple(
            int(self.data[column].iloc[self.index]) for column in self.state_columns
        ) + (self.position,)

    def step(self, action: str) -> tuple[tuple[int, ...], float, bool, dict]:
        if self.done:
            raise RuntimeError("Episode finished. Call reset() before stepping again")
        if action not in {"buy", "sell", "hold"}:
            raise ValueError(f"Unrecognized action: {action}")
        if (action == "buy" and self.position == 1) or (
            action == "sell" and self.position == 0
        ):
            raise ValueError(f"Cannot {action} when position is {self.position}")
        decision_close = float(self.data.iloc[self.index]["Close"])
        next_bar = self.data.iloc[self.index + 1]
        market_open = float(next_bar["Open"])
        next_close = float(next_bar["Close"])
        # Existing holdings remain exposed to the overnight price change.
        next_position = self.position
        overnight_pnl = self.position * (market_open - decision_close)
        fee = 0.0
        execution_price = market_open
        slippage_cost = 0.0
        if action == "buy":
            next_position = 1
            execution_price = market_open * (1 + self.slippage_bps / 10_000)
            slippage_cost = execution_price - market_open
            fee = execution_price * self.transaction_cost
        elif action == "sell":
            next_position = 0
            execution_price = market_open * (1 - self.slippage_bps / 10_000)
            slippage_cost = market_open - execution_price
            fee = execution_price * self.transaction_cost
        intraday_pnl = next_position * (next_close - market_open)
        reward = overnight_pnl + intraday_pnl - fee - slippage_cost
        try:
            updated_pnl = math.fsum((self.total_pnl, reward))
            updated_fees = math.fsum((self.total_fees, fee))
            updated_slippage = math.fsum((self.total_slippage, slippage_cost))
        except OverflowError as exc:
            raise ValueError("Total trading reward must remain finite") from exc
        if not np.isfinite(reward):
            raise ValueError("Trading reward must remain finite")
        if not all(
            np.isfinite(value)
            for value in (updated_pnl, updated_fees, updated_slippage)
        ):
            raise ValueError("Total trading reward must remain finite")
        self.total_pnl = updated_pnl
        self.total_fees = updated_fees
        self.total_slippage = updated_slippage
        self.trade_count += action != "hold"
        self.invested_steps += next_position
        self.overnight_invested_steps += self.position
        self.equity_curve.append(updated_pnl)
        self.position = next_position
        self.index += 1
        self.done = self.index == len(self.data) - 1
        return (
            self.get_state(),
            float(reward),
            self.done,
            {
                "position": self.position,
                "fee": fee,
                "execution_price": execution_price,
                "slippage_cost": slippage_cost,
            },
        )

    def episode_metrics(self) -> dict[str, float | int]:
        """Summarize marked-to-market episode results without forcing an exit."""
        equity = np.asarray(self.equity_curve, dtype=float)
        # Subtracting two finite equity values can still overflow float64.
        # Fail explicitly rather than report an infinite drawdown.
        try:
            with np.errstate(over="raise", invalid="raise"):
                drawdown = np.maximum.accumulate(equity) - equity
        except FloatingPointError as exc:
            raise ValueError("Trading drawdown must remain finite") from exc
        if not np.isfinite(drawdown).all():
            raise ValueError("Trading drawdown must remain finite")
        steps = len(equity) - 1
        return {
            "pnl": float(self.total_pnl),
            "max_drawdown_dollars": float(drawdown.max()),
            "trades": self.trade_count,
            "fees_dollars": float(self.total_fees),
            "slippage_dollars": float(self.total_slippage),
            "intraday_exposure_fraction": self.invested_steps / steps if steps else 0.0,
            "overnight_exposure_fraction": self.overnight_invested_steps / steps
            if steps
            else 0.0,
            "terminal_position": self.position,
        }


def contiguous_trading_segments(
    data: pd.DataFrame, *, verified_datetime_dates: bool = False
) -> list[pd.DataFrame]:
    """Create single-symbol episodes without crossing missing market sessions."""
    if data.empty:
        return []
    require_verified_daily_date_source(
        data, verified_datetime_dates=verified_datetime_dates
    )
    ordered = data.copy()
    if "Name" in ordered:
        ordered["Name"] = normalize_ticker_symbols(
            ordered["Name"],
            context="Trading episodes require valid ticker names",
        )
    if "Date" in ordered:
        ordered["Date"] = parse_daily_dates(ordered["Date"], context="Trading dates")
        mark_verified_daily_dates(ordered, columns=("Date",))
        ordered = ordered.sort_values("Date", kind="stable")
    validate_session_positions(ordered)
    # Retain source-wide exchange calendars before isolating individual stocks.
    if "Session_Number" not in ordered and "Date" in ordered:
        ordered["Session_Number"] = session_positions(ordered)
    symbol_groups = (
        (group for _, group in ordered.groupby("Name", sort=False))
        if "Name" in ordered
        else (ordered,)
    )
    segments = []
    for symbol in symbol_groups:
        boundaries = (~adjacent_session_flags(session_positions(symbol))).cumsum()
        segments.extend(
            group.copy() for _, group in symbol.groupby(boundaries) if len(group) >= 2
        )
    return segments


class QLearningAgent:
    """Tabular epsilon-greedy learner with correct terminal-state updates."""

    def __init__(
        self,
        actions: tuple[str, ...] = ("buy", "sell", "hold"),
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        seed: int = 42,
    ):
        actions = tuple(actions)
        supported = {"buy", "sell", "hold"}
        if (
            not actions
            or any(not isinstance(action, str) for action in actions)
            or len(set(actions)) != len(actions)
            or any(action not in supported for action in actions)
            or ("hold" not in actions and not {"buy", "sell"}.issubset(actions))
            or any(
                not isinstance(value, Real)
                or isinstance(value, (bool, np.bool_))
                or not np.isfinite(value)
                for value in (alpha, gamma, epsilon)
            )
            or not 0 < alpha <= 1
            or not 0 <= gamma <= 1
            or not 0 <= epsilon <= 1
            or not isinstance(seed, (int, np.integer))
            or isinstance(seed, (bool, np.bool_))
            or seed < 0
        ):
            raise ValueError("Invalid Q-learning parameters")
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: dict[tuple[int, ...], dict[str, float]] = {}
        self.rng = random.Random(int(seed))

    def get_state(self, state_features) -> tuple[int, ...]:
        try:
            state = tuple(state_features)
        except TypeError as exc:
            raise ValueError("State components must be integers") from exc
        if any(
            not isinstance(value, Integral) or isinstance(value, (bool, np.bool_))
            for value in state
        ):
            raise ValueError("State components must be integers")
        return tuple(int(value) for value in state)

    def _values(self, state: tuple[int, ...]) -> dict[str, float]:
        return self.q_table.setdefault(state, {action: 0.0 for action in self.actions})

    def _valid_actions(self, state: tuple[int, ...]) -> tuple[str, ...]:
        if not state or state[-1] not in (0, 1):
            raise ValueError(
                "Last state component must be the current position (0 or 1)"
            )
        prohibited = "sell" if state[-1] == 0 else "buy"
        valid = tuple(action for action in self.actions if action != prohibited)
        if not valid:
            raise ValueError("Agent has no valid actions for the current position")
        return valid

    def choose_action(self, state_features, *, explore: bool = True) -> str:
        state = self.get_state(state_features)
        available = self._valid_actions(state)
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(available)
        # Holdout evaluation must not alter the learned Q-table. A state never
        # observed during training takes the conservative no-trade action.
        if not explore and state not in self.q_table:
            return "hold" if "hold" in available else available[0]
        estimates = self._values(state) if explore else self.q_table[state]
        return max(available, key=estimates.__getitem__)

    def update_q_value(
        self,
        state_features,
        action: str,
        reward: float,
        next_state_features,
        done: bool = False,
    ) -> None:
        if not isinstance(done, (bool, np.bool_)):
            raise ValueError("done must be a boolean")
        state = self.get_state(state_features)
        if action not in self._valid_actions(state):
            raise ValueError(f"Unrecognized action: {action}")
        next_state = self.get_state(next_state_features)
        if len(next_state) != len(state):
            raise ValueError(
                "Current and next trading states must have the same dimensions"
            )
        valid_next_actions = self._valid_actions(next_state)
        expected_position = {"buy": 1, "sell": 0, "hold": state[-1]}[action]
        if next_state[-1] != expected_position:
            raise ValueError("Next position is inconsistent with the trading action")
        if isinstance(reward, (bool, np.bool_)):
            raise ValueError("Reward must be a finite real number")
        # A float() conversion can discard the imaginary part of a NumPy
        # complex scalar. Reuse the shared strict scalar validation instead.
        scalar_reward = real_float_array([reward], context="Reward")
        if scalar_reward.shape != (1,):
            raise ValueError("Reward must be a finite real scalar")
        reward = float(scalar_reward[0])
        # Calculate and validate the update before changing either state entry.
        future = (
            0.0
            if done
            else max(
                self.q_table.get(next_state, {}).get(candidate, 0.0)
                for candidate in valid_next_actions
            )
        )
        prior = self.q_table.get(state, {}).get(action, 0.0)
        try:
            updated = math.fsum(
                (
                    (1 - self.alpha) * prior,
                    self.alpha * reward,
                    self.alpha * self.gamma * future,
                )
            )
        except OverflowError as exc:
            raise ValueError("Q-value update must remain finite") from exc
        if not np.isfinite(updated):
            raise ValueError("Q-value update must remain finite")
        if not done:
            self._values(next_state)
        current = self._values(state)
        current[action] = updated

    def simulate_episode(
        self, env: SimpleTradingEnv, max_steps: int | None = None, *, train: bool = True
    ) -> float:
        if max_steps is not None and (
            not isinstance(max_steps, Integral)
            or isinstance(max_steps, (bool, np.bool_))
            or max_steps < 0
        ):
            raise ValueError("max_steps must be a nonnegative integer")
        state = env.reset()
        total_reward = 0.0
        steps = max_steps if max_steps is not None else len(env.data) - 1
        for _ in range(steps):
            action = self.choose_action(state, explore=train)
            next_state, reward, done, _ = env.step(action)
            try:
                next_total_reward = math.fsum((total_reward, reward))
            except OverflowError as exc:
                raise ValueError("Total trading reward must remain finite") from exc
            if not np.isfinite(next_total_reward):
                raise ValueError("Total trading reward must remain finite")
            if train:
                self.update_q_value(state, action, reward, next_state, done)
            state = next_state
            total_reward = next_total_reward
            if done:
                break
        return float(total_reward)


if __name__ == "__main__":
    demonstration = pd.DataFrame(
        {
            "Open": [99, 101, 102, 102, 103, 105, 105],
            "Close": [100, 102, 101, 103, 105, 104, 106],
        }
    )
    env = SimpleTradingEnv(demonstration)
    agent = QLearningAgent()
    for _ in range(50):
        agent.simulate_episode(env)
    print(
        f"Demonstration episode reward (learning disabled): {agent.simulate_episode(env, train=False):.2f}"
    )
