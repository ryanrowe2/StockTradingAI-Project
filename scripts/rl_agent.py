"""Single-symbol, long-or-flat Q-learning trading research environment.

The agent can buy one share, sell its existing share, or hold. Decisions based
on a completed daily bar execute at the following session's opening price.
Every step is marked to market, with transaction costs and terminal handling.
This is an experimental simulator, not an execution or profitability claim.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd


class SimpleTradingEnv:
    """Sequential one-share simulator with next-session-open execution."""

    def __init__(self, data: pd.DataFrame, transaction_cost: float = 0.001):
        if not {"Open", "Close"}.issubset(data.columns) or len(data) < 2:
            raise ValueError("The trading environment needs at least two daily open/close bars")
        if "Name" in data and data["Name"].nunique() != 1:
            raise ValueError("A trading episode must contain exactly one ticker")
        if not 0 <= transaction_cost < 1:
            raise ValueError("transaction_cost must be a fraction between 0 and 1")
        self.data = data.sort_values("Date").reset_index(drop=True) if "Date" in data else data.reset_index(drop=True)
        if "Date" in self.data and self.data["Date"].duplicated().any():
            raise ValueError("Duplicate trading dates")
        prices = self.data[["Open", "Close"]].to_numpy(dtype=float)
        if not np.isfinite(prices).all() or np.any(prices <= 0):
            raise ValueError("Open/close prices must be finite and positive")
        self.transaction_cost = transaction_cost
        self.reset()

    def reset(self) -> tuple[int, ...]:
        self.index = 0
        self.position = 0
        self.done = False
        return self.get_state()

    def get_state(self) -> tuple[int, ...]:
        row = self.data.iloc[self.index]
        fields = ("Open_binned", "High_binned", "Low_binned", "Market_Regime")
        return tuple(int(row[field]) if field in row and pd.notna(row[field]) else 0
                     for field in fields) + (self.position,)

    def step(self, action: str) -> tuple[tuple[int, ...], float, bool, dict]:
        if self.done:
            raise RuntimeError("Episode finished. Call reset() before stepping again")
        if action not in {"buy", "sell", "hold"}:
            raise ValueError(f"Unrecognized action: {action}")
        decision_close = float(self.data.iloc[self.index]["Close"])
        next_bar = self.data.iloc[self.index + 1]
        execution_price = float(next_bar["Open"])
        next_close = float(next_bar["Close"])
        # Existing holdings remain exposed to the overnight price change.
        overnight_pnl = self.position * (execution_price - decision_close)
        fee = 0.0
        if action == "buy" and self.position == 0:
            self.position = 1
            fee = execution_price * self.transaction_cost
        elif action == "sell" and self.position == 1:
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
        if not actions or not 0 < alpha <= 1 or not 0 <= gamma <= 1 or not 0 <= epsilon <= 1:
            raise ValueError("Invalid Q-learning parameters")
        self.actions = tuple(actions)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: dict[tuple[int, ...], dict[str, float]] = {}
        self.rng = random.Random(seed)

    def get_state(self, state_features) -> tuple[int, ...]:
        return tuple(int(value) for value in state_features)

    def _values(self, state: tuple[int, ...]) -> dict[str, float]:
        return self.q_table.setdefault(state, {action: 0.0 for action in self.actions})

    def choose_action(self, state_features, *, explore: bool = True) -> str:
        state = self.get_state(state_features)
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        # Holdout evaluation must not alter the learned Q-table. A state never
        # observed during training takes the conservative no-trade action.
        if not explore and state not in self.q_table:
            return "hold" if "hold" in self.actions else self.actions[0]
        estimates = self._values(state) if explore else self.q_table[state]
        return max(self.actions, key=estimates.__getitem__)

    def update_q_value(self, state_features, action: str, reward: float,
                       next_state_features, done: bool = False) -> None:
        state = self.get_state(state_features)
        current = self._values(state)
        if action not in current:
            raise ValueError(f"Unrecognized action: {action}")
        future = 0.0 if done else max(self._values(self.get_state(next_state_features)).values())
        current[action] += self.alpha * (reward + self.gamma * future - current[action])

    def simulate_episode(self, env: SimpleTradingEnv, max_steps: int | None = None,
                         *, train: bool = True) -> float:
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
