"""Tabular Q-learning baseline (paper's own comparison point, Sec. IV).

Reproduces "a well-tuned version of the Q learning algorithm proposed in
[Wang & Zhang]... the electricity prices and the energy levels are
discretized into 100 and 10 intervals, respectively." This is deliberately
NOT the same discretization as this repo's own qlearning_realtime_arbitrage_2018
project (10 price bins / 9 energy bins, tuned separately for that paper's
own case study) -- 100/10 here matches THIS paper's specific stated
benchmark, for a fair comparison against its own PPO/PPO-RNN results.

Uses the exact same environment and reward (Eq. 4) as the PPO agent -- the
only difference is the state representation (discretized price/energy
only, no average-cost-basis or RNN hidden feature), which is the point of
the comparison.

The paper doesn't restate alpha/epsilon for this "well-tuned" baseline
(only cites Wang & Zhang's algorithm by reference), so this uses that
paper's own literal values (alpha=0.5, epsilon=0.9) for the learning rate
and exploration probability, but this paper's own gamma=0.999 (rather than
Wang & Zhang's 0.9) so Q-learning and PPO are optimizing the same
discounted objective on the same MDP -- an assumption, not a stated value.
"""

import numpy as np

from src.environment import StorageArbitrageEnv


class QLearningAgent:
    def __init__(self, n_price_bins, n_energy_bins, n_actions=3,
                 alpha=0.5, gamma=0.999, epsilon=0.9, seed=0):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.q = np.zeros((n_price_bins, n_energy_bins, n_actions))
        self.rng = np.random.default_rng(seed)

    def select_action(self, state, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.integers(self.n_actions)
        price_bin, energy_bin = state
        return int(np.argmax(self.q[price_bin, energy_bin]))

    def update(self, state, action, reward, next_state):
        price_bin, energy_bin = state
        current = self.q[price_bin, energy_bin, action]
        if next_state is None:
            target = reward
        else:
            next_price_bin, next_energy_bin = next_state
            target = reward + self.gamma * np.max(self.q[next_price_bin, next_energy_bin])
        self.q[price_bin, energy_bin, action] = (1 - self.alpha) * current + self.alpha * target


def fit_price_bin_edges(prices, n_price_bins, method="quantile"):
    """Same causal-fitting principle used throughout this project collection:
    equal-width bins waste resolution on heavy-tailed real-time price spikes
    (see qlearning_realtime_arbitrage_2018's README for the measured effect),
    so quantile (equal-frequency) is the default; equal_width is available
    for the paper's more literal "100 price intervals" reading.
    """
    prices = np.asarray(prices, dtype=float)
    if method == "equal_width":
        return np.linspace(prices.min(), prices.max(), n_price_bins + 1)
    if method == "quantile":
        edges = np.quantile(prices, np.linspace(0, 1, n_price_bins + 1))
        edges[0], edges[-1] = prices.min(), prices.max()
        return np.unique(edges) if len(np.unique(edges)) >= 2 else np.linspace(prices.min(), prices.max() + 1, n_price_bins + 1)
    raise ValueError(f"Unknown method: {method!r}")


def discretize_price(price, edges):
    idx = np.searchsorted(edges, price, side="right") - 1
    return int(np.clip(idx, 0, len(edges) - 2))


def discretize_energy(energy, e_min, e_max, n_energy_bins):
    edges = np.linspace(e_min, e_max, n_energy_bins + 1)
    idx = np.searchsorted(edges, energy, side="right") - 1
    return int(np.clip(idx, 0, n_energy_bins - 1))


def train_qlearning(prices, n_price_bins=100, n_energy_bins=10, price_bin_method="quantile",
                     bin_calibration_hours=24 * 30, env_kwargs=None, seed=0, **agent_kwargs):
    """One online pass through `prices` (matching this repo's other Q-learning
    project's training style, and the paper's own "trained using data of
    2000 weeks" framing doesn't apply here -- the Q-learning baseline is
    Wang & Zhang's original single online pass, not PPO's repeated-sampling
    scheme).
    """
    env_kwargs = env_kwargs or {}
    e_min = env_kwargs.get("e_min", 0.0)
    e_max = env_kwargs.get("capacity_mwh", 8.0)

    calibration_prices = prices[:min(bin_calibration_hours, len(prices))]
    price_edges = fit_price_bin_edges(calibration_prices, n_price_bins, price_bin_method)

    env = StorageArbitrageEnv(prices, hidden_states=None, **env_kwargs)
    agent = QLearningAgent(len(price_edges) - 1, n_energy_bins, seed=seed, **agent_kwargs)

    def to_disc_state(raw_state):
        energy, _avg_cost, price = raw_state[0], raw_state[1], raw_state[2]
        return (discretize_price(price, price_edges), discretize_energy(energy, e_min, e_max, n_energy_bins))

    raw_state = env.reset()
    state = to_disc_state(raw_state)
    history = []
    cumulative_profit = 0.0
    done = False
    while not done:
        action = agent.select_action(state)
        next_raw_state, price, c, d, reward, done = env.step(action)
        next_state = to_disc_state(next_raw_state) if next_raw_state is not None else None
        agent.update(state, action, reward, next_state)
        cumulative_profit += reward
        history.append({"t": env.t - 1, "price": price, "action": action, "c": c, "d": d,
                         "reward": reward, "cumulative_profit": cumulative_profit})
        state = next_state
        if done:
            break

    return agent, history, cumulative_profit, price_edges
