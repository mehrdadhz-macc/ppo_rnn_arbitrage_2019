"""PPO actor-critic training (paper Sec. III-B, Algorithm 1).

Separate actor and critic networks (paper Fig. 1 shows them as two distinct
boxes, not a shared trunk), each 2 hidden layers of 128 and 32 units, ReLU,
no output activation -- Sec. IV's stated architecture. Categorical policy
over the 3 discrete bang-bang actions (Eq. 5).

Each of the K=200 outer updates collects D=10 independent one-week (T=168h)
trajectories, each starting at a uniformly random offset into the training
price series with the battery reset to empty (paper: "trained using data of
2000 weeks, obtained via sampling with replacement" -- 200 updates x 10
trajectories/update = 2000). Advantages come from GAE (Eq. 12); the value
target for the regression in Eq. 11 is advantage + baseline value, the
standard identity for GAE-derived returns. Both networks are optimized for
100 gradient steps per outer update (paper Sec. IV), policy lr=1e-3, value
lr=1e-4, both via Adam.

The paper bootstraps the value target at the end of each truncated
168h window with the critic's own estimate there (Eq. 11), rather than
treating the window's end as a true terminal state -- appropriate since a
"week" is an artificial slice of a much longer price series, not a genuine
episode boundary. This project does the same via `reset_at`, letting a
trajectory's underlying environment continue one step past the sampled
window purely to read off that bootstrap value.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.environment import StorageArbitrageEnv


class ActorCritic(nn.Module):
    def __init__(self, state_dim, n_actions=3, hidden1=128, hidden2=32):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def policy_dist(self, states):
        return Categorical(logits=self.actor(states))

    def value(self, states):
        return self.critic(states).squeeze(-1)

    def act(self, state, greedy=False):
        """state: (state_dim,) tensor. Returns (action:int, log_prob:float, value:float)."""
        with torch.no_grad():
            dist = self.policy_dist(state.unsqueeze(0))
            action = torch.argmax(dist.logits, dim=-1) if greedy else dist.sample()
            log_prob = dist.log_prob(action)
            value = self.value(state.unsqueeze(0))
        return action.item(), log_prob.item(), value.item()


def compute_gae(rewards, values, bootstrap_value, gamma, lam):
    """Eq. 12. Returns (advantages, value_targets) as python lists."""
    T = len(rewards)
    values_ext = values + [bootstrap_value]
    deltas = [rewards[t] + gamma * values_ext[t + 1] - values_ext[t] for t in range(T)]
    advantages = [0.0] * T
    running = 0.0
    for t in reversed(range(T)):
        running = deltas[t] + gamma * lam * running
        advantages[t] = running
    returns = [advantages[t] + values[t] for t in range(T)]
    return advantages, returns


class PPOTrainer:
    def __init__(self, state_dim, n_actions=3, gamma=0.999, lam=0.97, clip_eps=0.2,
                 policy_lr=1e-3, value_lr=1e-4, n_grad_steps=100, env_kwargs=None, seed=0):
        torch.manual_seed(seed)
        self.model = ActorCritic(state_dim, n_actions)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.n_grad_steps = n_grad_steps
        self.env_kwargs = env_kwargs or {}
        self.policy_optimizer = torch.optim.Adam(self.model.actor.parameters(), lr=policy_lr)
        self.value_optimizer = torch.optim.Adam(self.model.critic.parameters(), lr=value_lr)

    def collect_trajectory(self, prices, hidden_states, traj_len, rng):
        env = StorageArbitrageEnv(prices, hidden_states, **self.env_kwargs)
        start = rng.integers(0, len(prices) - traj_len)  # leaves room for one bootstrap step past the window
        env.reset_at(start)

        states, actions, log_probs, values, rewards = [], [], [], [], []
        for _ in range(traj_len):
            state = torch.as_tensor(np.asarray(env._state(), dtype=np.float32))
            action, log_prob, value = self.model.act(state)
            _, _, _, _, reward, done = env.step(action)
            states.append(state.numpy())
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            if done:
                break

        if env.t < env.T:
            bootstrap_state = torch.as_tensor(np.asarray(env._state(), dtype=np.float32))
            with torch.no_grad():
                bootstrap_value = self.model.value(bootstrap_state.unsqueeze(0)).item()
        else:
            bootstrap_value = 0.0

        return states, actions, log_probs, values, rewards, bootstrap_value

    def update(self, states, actions, old_log_probs, advantages, returns):
        states_t = torch.as_tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.as_tensor(np.array(actions), dtype=torch.long)
        old_log_probs_t = torch.as_tensor(np.array(old_log_probs), dtype=torch.float32)
        advantages_t = torch.as_tensor(np.array(advantages), dtype=torch.float32)
        returns_t = torch.as_tensor(np.array(returns), dtype=torch.float32)

        policy_loss_val = value_loss_val = 0.0
        for _ in range(self.n_grad_steps):
            dist = self.model.policy_dist(states_t)
            new_log_probs = dist.log_prob(actions_t)
            ratio = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages_t
            policy_loss = -torch.mean(torch.min(surr1, surr2))

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            values_pred = self.model.value(states_t)
            value_loss = torch.mean((values_pred - returns_t) ** 2)

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            policy_loss_val, value_loss_val = policy_loss.item(), value_loss.item()

        return policy_loss_val, value_loss_val

    def train(self, prices, hidden_states, n_updates=200, n_trajectories=10, traj_len=168, seed=0):
        rng = np.random.default_rng(seed)
        history = []
        for update_idx in range(n_updates):
            batch_states, batch_actions, batch_log_probs = [], [], []
            batch_advantages, batch_returns = [], []
            weekly_profits = []

            for _ in range(n_trajectories):
                states, actions, log_probs, values, rewards, bootstrap = self.collect_trajectory(
                    prices, hidden_states, traj_len, rng)
                advantages, returns = compute_gae(rewards, values, bootstrap, self.gamma, self.lam)
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_log_probs.extend(log_probs)
                batch_advantages.extend(advantages)
                batch_returns.extend(returns)
                weekly_profits.append(float(np.sum(rewards)))

            policy_loss, value_loss = self.update(
                batch_states, batch_actions, batch_log_probs, batch_advantages, batch_returns)

            mean_profit = float(np.mean(weekly_profits))
            history.append({"update": update_idx, "mean_weekly_profit": mean_profit,
                             "policy_loss": policy_loss, "value_loss": value_loss})
            if (update_idx + 1) % 10 == 0:
                print(f"  update {update_idx + 1}/{n_updates}: mean weekly profit=${mean_profit:,.2f}")

        return history
