"""Single-symbol, long-or-flat Q-learning trading research environment.

The agent can buy one share, sell its existing share, or hold. Decisions based
on a completed daily bar execute at the following session's opening price.
Every step is marked to market, with transaction costs and terminal handling.
This is an experimental simulator, not an execution or profitability claim.
"""

from __future__ import annotations

import random
from numbers import Integral

import numpy as np
import pandas as pd

if __package__:
    from .date_validation import parse_daily_dates
else:
    from date_validation import parse_daily_dates


class SimpleTradingEnv:
    """Sequential one-share simulator with next-session-open execution."""

    DEFAULT_STATE_COLUMNS = ("Open_binned", "High_binned", "Low_binned", "Market_Regime")

    def __init__(self, data: pd.DataFrame, transaction_cost: float = 0.001,
                 state_columns: tuple[str, ...] | None = None):
        if not {"Open", "Close"}.issubset(data.columns) or len(data) < 2:
            raise ValueError("The trading environment needs at least two daily open/close bars")
        if "Name" in data:
            symbols = data["Name"]
            if (symbols.isna().any() or symbols.astype(str).str.strip().eq("").any()
                    or symbols.nunique() != 1):
                raise ValueError("A trading episode must contain exactly one ticker with a valid name")
        if not 0 <= transaction_cost < 1:
            raise ValueError("transaction_cost must be a fraction between 0 and 1")
        self.data = data.copy()
        if "Date" in self.data:
            self.data["Date"] = parse_daily_dates(self.data["Date"], context="Trading dates")
            self.data = self.data.sort_values("Date", kind="stable")
        self.data = self.data.reset_index(drop=True)
        if "Date" in self.data and self.data["Date"].duplicated().any():
            raise ValueError("Duplicate trading dates")
        prices = self.data[["Open", "Close"]].to_numpy(dtype=float)
        if not np.isfinite(prices).all() or np.any(prices <= 0):
            raise ValueError("Open/close prices must be finite and positive")
        if state_columns is None:
            provided = set(self.DEFAULT_STATE_COLUMNS).intersection(self.data.columns)
            if provided and len(provided) != len(self.DEFAULT_STATE_COLUMNS):
                raise ValueError("Supply all default state columns, or specify state_columns")
            state_columns = self.DEFAULT_STATE_COLUMNS if provided else ()
        if len(set(state_columns)) != len(state_columns):
            raise ValueError("State columns must be unique")
        missing = set(state_columns) - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing trading state columns: {sorted(missing)}")
        if state_columns:
            state_values = self.data[list(state_columns)].to_numpy(dtype=object)
            if any(not isinstance(value, Integral) or isinstance(value, (bool, np.bool_))
                   for value in state_values.flat):
                raise ValueError("Trading state columns must contain finite integer states")
        self.state_columns = tuple(state_columns)
        self.transaction_cost = transaction_cost
        self.reset()

    def reset(self) -> tuple[int, ...]:
        self.index = 0
        self.position = 0
        self.done = False
        return self.get_state()

    def get_state(self) -> tuple[int, ...]:
        row = self.data.iloc[self.index]
        return tuple(int(row[field]) for field in self.state_columns) + (self.position,)

    def step(self, action: str) -> tuple[tuple[int, ...], float, bool, dict]:
        if self.done:
            raise RuntimeError("Episode finished. Call reset() before stepping again")
        if action not in {"buy", "sell", "hold"}:
            raise ValueError(f"Unrecognized action: {action}")
        if (action == "buy" and self.position == 1) or (action == "sell" and self.position == 0):
            raise ValueError(f"Cannot {action} when position is {self.position}")
        decision_close = float(self.data.iloc[self.index]["Close"])
        next_bar = self.data.iloc[self.index + 1]
        execution_price = float(next_bar["Open"])
        next_close = float(next_bar["Close"])
        # Existing holdings remain exposed to the overnight price change.
        overnight_pnl = self.position * (execution_price - decision_close)
        fee = 0.0
        if action == "buy":
            self.position = 1
            fee = execution_price * self.transaction_cost
        elif action == "sell":
            self.position = 0
            fee = execution_price * self.transaction_cost
        intraday_pnl = self.position * (next_close - execution_price)
        reward = overnight_pnl + intraday_pnl - fee
        self.index += 1
        self.done = self.index == len(self.data) - 1
        return self.get_state(), float(reward), self.done, {
            "position": self.position, "fee": fee, "execution_price": execution_price,
        }


class QLearningAgent:
    """Tabular epsilon-greedy learner with correct terminal-state updates."""

    def __init__(self, actions: tuple[str, ...] = ("buy", "sell", "hold"),
                 alpha: float = 0.1, gamma: float = 0.95,
                 epsilon: float = 0.1, seed: int = 42):
        actions = tuple(actions)
        supported = {"buy", "sell", "hold"}
        if (not actions or any(not isinstance(action, str) for action in actions)
                or len(set(actions)) != len(actions)
                or any(action not in supported for action in actions)
                or ("hold" not in actions and not {"buy", "sell"}.issubset(actions))
                or not 0 < alpha <= 1 or not 0 <= gamma <= 1 or not 0 <= epsilon <= 1):
            raise ValueError("Invalid Q-learning parameters")
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: dict[tuple[int, ...], dict[str, float]] = {}
        self.rng = random.Random(seed)

    def get_state(self, state_features) -> tuple[int, ...]:
        try:
            state = tuple(state_features)
        except TypeError as exc:
            raise ValueError("State components must be integers") from exc
        if any(not isinstance(value, Integral) or isinstance(value, (bool, np.bool_))
               for value in state):
            raise ValueError("State components must be integers")
        return tuple(int(value) for value in state)

    def _values(self, state: tuple[int, ...]) -> dict[str, float]:
        return self.q_table.setdefault(state, {action: 0.0 for action in self.actions})

    def _valid_actions(self, state: tuple[int, ...]) -> tuple[str, ...]:
        if not state or state[-1] not in (0, 1):
            raise ValueError("Last state component must be the current position (0 or 1)")
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

    def update_q_value(self, state_features, action: str, reward: float,
                       next_state_features, done: bool = False) -> None:
        if not isinstance(done, (bool, np.bool_)):
            raise ValueError("done must be a boolean")
        state = self.get_state(state_features)
        if action not in self._valid_actions(state):
            raise ValueError(f"Unrecognized action: {action}")
        next_state = self.get_state(next_state_features)
        valid_next_actions = self._valid_actions(next_state)
        try:
            reward = float(reward)
        except (TypeError, ValueError) as exc:
            raise ValueError("Reward must be finite") from exc
        if not np.isfinite(reward):
            raise ValueError("Reward must be finite")
        # Calculate and validate the update before changing either state entry.
        future = (0.0 if done else max(self.q_table.get(next_state, {}).get(candidate, 0.0)
                                      for candidate in valid_next_actions))
        prior = self.q_table.get(state, {}).get(action, 0.0)
        updated = prior + self.alpha * (reward + self.gamma * future - prior)
        if not np.isfinite(updated):
            raise ValueError("Q-value update must remain finite")
        if not done:
            self._values(next_state)
        current = self._values(state)
        current[action] = updated

    def simulate_episode(self, env: SimpleTradingEnv, max_steps: int | None = None,
                         *, train: bool = True) -> float:
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
            if train:
                self.update_q_value(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        return float(total_reward)


if __name__ == "__main__":
    demonstration = pd.DataFrame({
        "Open": [99, 101, 102, 102, 103, 105, 105],
        "Close": [100, 102, 101, 103, 105, 104, 106],
    })
    env = SimpleTradingEnv(demonstration)
    agent = QLearningAgent()
    for _ in range(50):
        agent.simulate_episode(env)
    print(f"Demonstration episode reward (learning disabled): {agent.simulate_episode(env, train=False):.2f}")
