"""Greedily evaluate the trained Q-learning / PPO / PPO-RNN policies on held-out test data.

Reproduces the paper's Fig. 3 comparison: cumulative profit of each method
replayed on a year's held-out last-3-months test split, using each
method's FROZEN, greedy (no exploration) policy.

Usage:
    venv/bin/python3 evaluate.py --run outputs/runs/<timestamp> --data data/test/pjm_rto_rt_hourly_lmp_2018_test.csv
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data_loader import load_price_series
from src.environment import StorageArbitrageEnv
from src.price_encoder import PriceEncoder, compute_hidden_states
from src.ppo_agent import ActorCritic
from src.qlearning_baseline import discretize_price, discretize_energy


def latest_run_dir():
    runs = sorted(Path("outputs/runs").glob("*/"))
    if not runs:
        raise SystemExit("No runs found under outputs/runs/. Run train.py first.")
    return runs[-1]


def infer_train_path(test_path):
    """data/test/..._test.csv -> data/train/..._train.csv, by this project's naming convention."""
    test_path = Path(test_path)
    return Path(str(test_path).replace("/test/", "/train/").replace("_test.csv", "_train.csv"))


def rollout_cumulative_profit(env, choose_action):
    state = env.reset()
    cumulative_profit = 0.0
    curve = []
    done = False
    while not done:
        action = choose_action(state)
        next_state, price, c, d, reward, done = env.step(action)
        cumulative_profit += reward
        curve.append(cumulative_profit)
        state = next_state
    return np.array(curve)


def evaluate_qlearning(run_dir, prices, env_kwargs):
    q_path = run_dir / "qlearning_q_table.npy"
    edges_path = run_dir / "qlearning_price_edges.npy"
    if not q_path.exists():
        return None
    q_table = np.load(q_path)
    price_edges = np.load(edges_path)
    e_min = env_kwargs.get("e_min", 0.0)
    e_max = env_kwargs.get("capacity_mwh", 8.0)
    n_energy_bins = q_table.shape[1]

    env = StorageArbitrageEnv(prices, hidden_states=None, **env_kwargs)

    def choose_action(raw_state):
        energy, _avg_cost, price = raw_state[0], raw_state[1], raw_state[2]
        price_bin = discretize_price(price, price_edges)
        energy_bin = discretize_energy(energy, e_min, e_max, n_energy_bins)
        return int(np.argmax(q_table[price_bin, energy_bin]))

    return rollout_cumulative_profit(env, choose_action)


def evaluate_ppo(run_dir, prices, env_kwargs):
    model_path = run_dir / "ppo_model.pt"
    if not model_path.exists():
        return None
    model = ActorCritic(state_dim=3)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    env = StorageArbitrageEnv(prices, hidden_states=None, **env_kwargs)

    def choose_action(raw_state):
        state_t = torch.as_tensor(np.asarray(raw_state, dtype=np.float32))
        action, _, _ = model.act(state_t, greedy=True)
        return action

    return rollout_cumulative_profit(env, choose_action)


def evaluate_ppo_rnn(run_dir, train_prices, prices, env_kwargs, encoder_hidden_size, encoder_alpha):
    """`train_prices` is prepended purely so the RNN's hidden state carries over
    continuously from where training left off (matching how train.py computes
    hidden states continuously over its own price series) -- the rollout below
    still only steps through `prices` (the test period). Without this, the RNN
    would evaluate starting from a cold h_0 right at the test boundary, throwing
    away the price-trend memory it would have accumulated by then in a genuinely
    continuous deployment.
    """
    model_path = run_dir / "ppo_rnn_model.pt"
    encoder_path = run_dir / "price_encoder.pt"
    if not model_path.exists() or not encoder_path.exists():
        return None

    encoder = PriceEncoder(hidden_size=encoder_hidden_size, alpha=encoder_alpha)
    encoder.load_state_dict(torch.load(encoder_path))
    full_prices = np.concatenate([train_prices, prices])
    hidden_states = compute_hidden_states(encoder, full_prices)[len(train_prices):]

    model = ActorCritic(state_dim=3 + encoder_hidden_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    env = StorageArbitrageEnv(prices, hidden_states=hidden_states, **env_kwargs)

    def choose_action(raw_state):
        state_t = torch.as_tensor(np.asarray(raw_state, dtype=np.float32))
        action, _, _ = model.act(state_t, greedy=True)
        return action

    return rollout_cumulative_profit(env, choose_action)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=None, help="Run directory (default: most recent under outputs/runs)")
    parser.add_argument("--data", default="data/test/pjm_rto_rt_hourly_lmp_2018_test.csv")
    parser.add_argument("--train-data", default=None,
                         help="Training-split CSV, used only to give the PPO-RNN price encoder "
                              "continuous history before the test period (default: inferred from --data)")
    parser.add_argument("--capacity-mwh", type=float, default=8.0)
    parser.add_argument("--max-rate-mw", type=float, default=2.0)
    parser.add_argument("--wear-cost", type=float, default=1.0)
    parser.add_argument("--encoder-hidden-size", type=int, default=16)
    parser.add_argument("--encoder-alpha", type=float, default=0.7)
    args = parser.parse_args()

    run_dir = Path(args.run) if args.run else latest_run_dir()
    _, prices = load_price_series(args.data)
    prices = prices.astype(np.float32)
    train_data_path = Path(args.train_data) if args.train_data else infer_train_path(args.data)
    _, train_prices = load_price_series(train_data_path)
    train_prices = train_prices.astype(np.float32)
    print(f"Evaluating run {run_dir} on {len(prices)} held-out hours from {args.data}")

    env_kwargs = dict(capacity_mwh=args.capacity_mwh, max_rate_mw=args.max_rate_mw, wear_cost=args.wear_cost)

    curves = {}
    for name, curve in [
        ("qlearning", evaluate_qlearning(run_dir, prices, env_kwargs)),
        ("ppo", evaluate_ppo(run_dir, prices, env_kwargs)),
        ("ppo_rnn", evaluate_ppo_rnn(run_dir, train_prices, prices, env_kwargs, args.encoder_hidden_size, args.encoder_alpha)),
    ]:
        if curve is not None:
            curves[name] = curve
            print(f"  {name}: held-out cumulative profit = ${curve[-1]:,.2f}")

    plt.figure(figsize=(9, 5))
    for name, curve in curves.items():
        plt.plot(curve, label=name)
    plt.xlabel("Time (hour)")
    plt.ylabel("Cumulative profit ($)")
    plt.title("Held-out evaluation (greedy policy, frozen models)")
    plt.legend()
    plt.tight_layout()

    out_path = run_dir / "eval_plot.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    (run_dir / "eval_summary.json").write_text(json.dumps(
        {"data": args.data, "results": {k: float(v[-1]) for k, v in curves.items()}}, indent=2))


if __name__ == "__main__":
    main()
