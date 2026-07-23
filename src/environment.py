"""Storage arbitrage MDP (paper Sec. II).

State s_t = (E_t, c_t, rho_t, h_t):
  E_t   remaining battery energy
  c_t   running average cost-basis of the energy currently stored (Eq. 3) --
        an accounting device, not a market price: charging blends the new
        energy's cost into this average; discharging doesn't change it.
        Realized profit only shows up in the reward when discharging,
        compared against this basis -- like FIFO/weighted-average-cost
        inventory accounting applied to stored electricity.
  rho_t current real-time price
  h_t   RNN hidden feature built from the price sequence (src/price_encoder.py)
        -- supplied to this environment precomputed, not built internally,
        since it depends only on prices (not on the agent's actions) and can
        be computed once per price series (see price_encoder.py's docstring).

Action: bang-bang, matching Wang & Zhang's Lemma 1 (cited directly in this
paper): discharge at the max feasible rate, charge at the max feasible
rate, or hold. HOLD is action index 0 (not 1 or 2) so that an untrained or
tied Q-table/policy defaults to the harmless no-op rather than an
infeasible discharge-from-empty -- see qlearning_realtime_arbitrage_2018's
src/environment.py for the failure mode this ordering avoids; the same
reasoning applies to this project's Q-learning baseline.
"""

import numpy as np


class StorageArbitrageEnv:
    HOLD, CHARGE, DISCHARGE = 0, 1, 2

    def __init__(
        self,
        prices,
        hidden_states=None,
        capacity_mwh=8.0,
        max_rate_mw=2.0,
        e_min=0.0,
        efficiency_charge=1.0,
        efficiency_discharge=1.0,
        wear_cost=1.0,
        tau=1.0,
    ):
        self.prices = np.asarray(prices, dtype=float)
        self.T = len(self.prices)
        self.hidden_states = hidden_states  # (T, hidden_size) array, or None
        self.e_min = e_min
        self.e_max = capacity_mwh
        self.c_max = max_rate_mw
        self.d_max = max_rate_mw
        self.eta_c = efficiency_charge
        self.eta_d = efficiency_discharge
        self.beta = wear_cost
        self.tau = tau

        self.t = 0
        self.energy = e_min  # E_1 = E_min (paper's own initial condition)
        self.avg_cost = 0.0  # c_1 = 0

    def feasible_rates(self):
        """(D~max, C~max): max feasible discharge / charge given current SoC headroom."""
        d_tilde = min(self.d_max, (self.energy - self.e_min) / self.tau)
        c_tilde = min(self.c_max, (self.e_max - self.energy) / self.tau)
        return max(d_tilde, 0.0), max(c_tilde, 0.0)

    def reset(self):
        return self.reset_at(0)

    def reset_at(self, t):
        """Start a fresh episode (E=e_min, avg_cost=0) at an arbitrary index
        into the price series, rather than always at t=0 -- used by PPO
        training, which samples random one-week (168h) windows from a much
        longer training price series rather than always starting at its
        beginning (paper Sec. IV: "D=10 trajectories... obtained via
        sampling with replacement").
        """
        self.t = t
        self.energy = self.e_min
        self.avg_cost = 0.0
        return self._state()

    def _state(self):
        raw = (self.energy, self.avg_cost, self.prices[self.t])
        if self.hidden_states is None:
            return raw
        return raw + tuple(self.hidden_states[self.t])

    def step(self, action):
        """Advance one hour. Returns (next_state, price, c, d, reward, done).

        c/d are the realized charge/discharge power for this step (one of
        them always 0, per the bang-bang action space).
        """
        price = self.prices[self.t]
        d_tilde, c_tilde = self.feasible_rates()

        p_d = p_c = 0.0
        if action == self.DISCHARGE and d_tilde > 0:
            p_d = d_tilde
        elif action == self.CHARGE and c_tilde > 0:
            p_c = c_tilde
        # else HOLD, or the requested action had no feasible headroom -> hold

        # Reward (Eq. 4). Charging's cost isn't booked immediately -- it's
        # folded into avg_cost below and only realized as profit/loss later
        # when discharging, which is why summing just the discharge terms
        # over an episode gives the total realized arbitrage profit (the
        # paper states this explicitly under Eq. 4).
        if p_d > 0:
            reward = (price * self.eta_d - self.avg_cost) * p_d * self.tau - self.beta * p_d
        elif p_c > 0:
            reward = -self.beta * p_c
        else:
            reward = 0.0

        # Update the average cost basis (Eq. 3) using E_t, BEFORE E updates.
        if p_c > 0:
            new_energy = p_c * self.tau
            denom = self.energy + new_energy
            self.avg_cost = (self.avg_cost * self.energy + price * new_energy / self.eta_c) / denom

        # Energy dynamics (Eq. 1) -- no efficiency term here; eta only
        # enters the reward, matching the paper's own Eq. 1 vs Eq. 4 split.
        self.energy = self.energy + (p_c - p_d) * self.tau
        if self.energy <= 1e-9:
            self.avg_cost = 0.0  # paper: c resets to 0 whenever E hits 0

        self.t += 1
        done = self.t >= self.T
        next_state = self._state() if not done else None
        return next_state, price, p_c, p_d, reward, done
