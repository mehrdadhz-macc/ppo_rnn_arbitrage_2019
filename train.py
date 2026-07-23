"""Train Q-learning, PPO, and PPO-RNN on one year's PJM-RTO training split (paper Sec. IV).

Usage:
    venv/bin/python3 train.py --data data/train/pjm_rto_rt_hourly_lmp_2018_train.csv
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data_loader import load_price_series
from src.environment import StorageArbitrageEnv
from src.price_encoder import fit_price_encoder, compute_hidden_states
from src.ppo_agent import PPOTrainer
from src.qlearning_baseline import train_qlearning


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/train/pjm_rto_rt_hourly_lmp_2018_train.csv")
    parser.add_argument("--methods", nargs="+", default=["qlearning", "ppo", "ppo_rnn"],
                         choices=["qlearning", "ppo", "ppo_rnn"])
    parser.add_argument("--capacity-mwh", type=float, default=8.0)
    parser.add_argument("--max-rate-mw", type=float, default=2.0)
    parser.add_argument("--wear-cost", type=float, default=1.0, help="beta, $/MW (Eq. 4)")
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--lam", type=float, default=0.97, help="GAE lambda")
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--n-updates", type=int, default=200, help="K in Algorithm 1")
    parser.add_argument("--n-trajectories", type=int, default=10, help="D in Algorithm 1")
    parser.add_argument("--traj-len", type=int, default=168, help="T in Algorithm 1 (1 week)")
    parser.add_argument("--encoder-hidden-size", type=int, default=16)
    parser.add_argument("--encoder-alpha", type=float, default=0.7)
    parser.add_argument("--encoder-steps", type=int, default=4000)
    parser.add_argument("--qlearning-price-bins", type=int, default=100)
    parser.add_argument("--qlearning-energy-bins", type=int, default=10)
    parser.add_argument("--qlearning-price-bin-method", choices=["equal_width", "quantile"], default="quantile")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None, help="Override outputs/runs/<timestamp>")
    args = parser.parse_args()

    _, prices = load_price_series(args.data)
    prices = prices.astype(np.float32)
    print(f"Loaded {len(prices)} hourly prices from {args.data} "
          f"(min={prices.min():.2f}, max={prices.max():.2f}, mean={prices.mean():.2f} $/MWh)")

    run_dir = Path(args.out_dir) if args.out_dir else Path("outputs/runs") / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(capacity_mwh=args.capacity_mwh, max_rate_mw=args.max_rate_mw, wear_cost=args.wear_cost)
    summary = {"data": args.data, "n_hours": len(prices), "args": vars(args), "results": {}}

    if "qlearning" in args.methods:
        print("\n=== Q-learning baseline ===")
        agent, history, cumulative_profit, price_edges = train_qlearning(
            prices, n_price_bins=args.qlearning_price_bins, n_energy_bins=args.qlearning_energy_bins,
            price_bin_method=args.qlearning_price_bin_method, env_kwargs=env_kwargs, seed=args.seed,
            gamma=args.gamma)
        print(f"  cumulative training profit = ${cumulative_profit:,.2f}")
        np.save(run_dir / "qlearning_q_table.npy", agent.q)
        np.save(run_dir / "qlearning_price_edges.npy", price_edges)
        np.save(run_dir / "qlearning_history.npy",
                np.array([(h["t"], h["price"], h["action"], h["c"], h["d"], h["reward"], h["cumulative_profit"])
                           for h in history]))
        summary["results"]["qlearning"] = {"cumulative_profit": cumulative_profit}

    encoder = None
    if "ppo_rnn" in args.methods:
        print("\n=== Pretraining price encoder (EMA + RNN) ===")
        encoder, losses = fit_price_encoder(
            prices, hidden_size=args.encoder_hidden_size, alpha=args.encoder_alpha,
            n_steps=args.encoder_steps, seed=args.seed)
        torch.save(encoder.state_dict(), run_dir / "price_encoder.pt")
        np.save(run_dir / "price_encoder_losses.npy", np.array(losses))
        print(f"  final auxiliary loss: {losses[-1]:.4f}")

    if "ppo" in args.methods:
        print("\n=== PPO (no RNN features) ===")
        trainer = PPOTrainer(state_dim=3, gamma=args.gamma, lam=args.lam, clip_eps=args.clip_eps,
                              env_kwargs=env_kwargs, seed=args.seed)
        history = trainer.train(prices, hidden_states=None, n_updates=args.n_updates,
                                 n_trajectories=args.n_trajectories, traj_len=args.traj_len, seed=args.seed)
        torch.save(trainer.model.state_dict(), run_dir / "ppo_model.pt")
        with open(run_dir / "ppo_history.json", "w") as f:
            json.dump(history, f)
        summary["results"]["ppo"] = {"final_mean_weekly_profit": history[-1]["mean_weekly_profit"]}

    if "ppo_rnn" in args.methods:
        print("\n=== PPO-RNN ===")
        hidden_states = compute_hidden_states(encoder, prices)
        trainer = PPOTrainer(state_dim=3 + args.encoder_hidden_size, gamma=args.gamma, lam=args.lam,
                              clip_eps=args.clip_eps, env_kwargs=env_kwargs, seed=args.seed)
        history = trainer.train(prices, hidden_states=hidden_states, n_updates=args.n_updates,
                                 n_trajectories=args.n_trajectories, traj_len=args.traj_len, seed=args.seed)
        torch.save(trainer.model.state_dict(), run_dir / "ppo_rnn_model.pt")
        with open(run_dir / "ppo_rnn_history.json", "w") as f:
            json.dump(history, f)
        summary["results"]["ppo_rnn"] = {"final_mean_weekly_profit": history[-1]["mean_weekly_profit"]}

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nRun artefacts written to {run_dir}")


if __name__ == "__main__":
    main()
