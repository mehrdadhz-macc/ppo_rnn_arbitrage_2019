"""Train Q-learning, PPO, and PPO-RNN on one year's PJM-RTO training split (paper Sec. IV).

PPO's exploration (and the price encoder's random h_0, per its own docstring)
are seeded random processes -- as this collection's sibling project
(qlearning_realtime_arbitrage_2018) found and rigorously confirmed with
100-trial paired testing, single-run comparisons between methods can be
swung by an order of magnitude by seed alone. --n-trials runs several
independent trials, saves every trial's models separately under trial_NN/,
and reports the mean and std of each method's profit across trials -- the
expected-value estimate results should actually be judged on, not any one
trial's number. --seed doesn't feed into any trial directly; it seeds an
RNG that GENERATES each trial's own seed, so the same --seed always
reproduces the same set of trial seeds, but two adjacent trials never
differ by a suspiciously simple +1.

Usage:
    venv/bin/python3 train.py --data data/train/pjm_rto_rt_hourly_lmp_2018_train.csv --n-trials 10
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


def run_trial(prices, seed, trial_dir, args, env_kwargs):
    """Train every requested method once, with the given seed, saving
    artefacts under trial_dir. Returns {method: {"cumulative_profit"|
    "final_mean_weekly_profit": ...}} for whichever methods ran.
    """
    trial_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    if "qlearning" in args.methods:
        agent, history, cumulative_profit, price_edges = train_qlearning(
            prices, n_price_bins=args.qlearning_price_bins, n_energy_bins=args.qlearning_energy_bins,
            price_bin_method=args.qlearning_price_bin_method, env_kwargs=env_kwargs, seed=seed,
            gamma=args.gamma)
        print(f"    qlearning: cumulative training profit = ${cumulative_profit:,.2f}")
        np.save(trial_dir / "qlearning_q_table.npy", agent.q)
        np.save(trial_dir / "qlearning_price_edges.npy", price_edges)
        np.save(trial_dir / "qlearning_history.npy",
                np.array([(h["t"], h["price"], h["action"], h["c"], h["d"], h["reward"], h["cumulative_profit"])
                           for h in history]))
        results["qlearning"] = {"cumulative_profit": cumulative_profit}

    encoder = None
    if "ppo_rnn" in args.methods:
        encoder, losses = fit_price_encoder(
            prices, hidden_size=args.encoder_hidden_size, alpha=args.encoder_alpha,
            n_steps=args.encoder_steps, seed=seed)
        torch.save(encoder.state_dict(), trial_dir / "price_encoder.pt")
        np.save(trial_dir / "price_encoder_losses.npy", np.array(losses))
        print(f"    price encoder: final auxiliary loss = {losses[-1]:.4f}")

    if "ppo" in args.methods:
        trainer = PPOTrainer(state_dim=3, gamma=args.gamma, lam=args.lam, clip_eps=args.clip_eps,
                              env_kwargs=env_kwargs, seed=seed)
        history = trainer.train(prices, hidden_states=None, n_updates=args.n_updates,
                                 n_trajectories=args.n_trajectories, traj_len=args.traj_len, seed=seed)
        torch.save(trainer.model.state_dict(), trial_dir / "ppo_model.pt")
        with open(trial_dir / "ppo_history.json", "w") as f:
            json.dump(history, f)
        print(f"    ppo: final mean weekly profit = ${history[-1]['mean_weekly_profit']:,.2f}")
        results["ppo"] = {"final_mean_weekly_profit": history[-1]["mean_weekly_profit"]}

    if "ppo_rnn" in args.methods:
        hidden_states = compute_hidden_states(encoder, prices)
        trainer = PPOTrainer(state_dim=3 + args.encoder_hidden_size, gamma=args.gamma, lam=args.lam,
                              clip_eps=args.clip_eps, env_kwargs=env_kwargs, seed=seed)
        history = trainer.train(prices, hidden_states=hidden_states, n_updates=args.n_updates,
                                 n_trajectories=args.n_trajectories, traj_len=args.traj_len, seed=seed)
        torch.save(trainer.model.state_dict(), trial_dir / "ppo_rnn_model.pt")
        with open(trial_dir / "ppo_rnn_history.json", "w") as f:
            json.dump(history, f)
        print(f"    ppo_rnn: final mean weekly profit = ${history[-1]['mean_weekly_profit']:,.2f}")
        results["ppo_rnn"] = {"final_mean_weekly_profit": history[-1]["mean_weekly_profit"]}

    return results


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
    parser.add_argument("--seed", type=int, default=0,
                         help="Seeds the RNG that GENERATES each trial's own seed (not used directly "
                              "as a trial seed) -- same --seed always reproduces the same set of "
                              "trial seeds, but they aren't a predictable seed/seed+1/seed+2 sequence")
    parser.add_argument("--n-trials", type=int, default=1,
                         help="Independent trials, each with its own seed; results are reported "
                              "as mean +/- std across trials")
    parser.add_argument("--out-dir", default=None, help="Override outputs/runs/<timestamp>")
    args = parser.parse_args()

    _, prices = load_price_series(args.data)
    prices = prices.astype(np.float32)
    print(f"Loaded {len(prices)} hourly prices from {args.data} "
          f"(min={prices.min():.2f}, max={prices.max():.2f}, mean={prices.mean():.2f} $/MWh)")

    run_dir = Path(args.out_dir) if args.out_dir else Path("outputs/runs") / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(capacity_mwh=args.capacity_mwh, max_rate_mw=args.max_rate_mw, wear_cost=args.wear_cost)

    # Generated once (not per method) so every method's trial i shares the
    # same seed -- a paired comparison that reduces variance attributable to
    # pure randomness luck when comparing methods against each other.
    trial_seeds = np.random.default_rng(args.seed).integers(0, 2**31 - 1, size=args.n_trials).tolist()
    print(f"Running {args.n_trials} trial(s), seeds {trial_seeds}")

    trial_results = {method: [] for method in args.methods}
    for trial, seed in enumerate(trial_seeds):
        print(f"\n=== Trial {trial} (seed={seed}) ===")
        trial_dir = run_dir / f"trial_{trial:02d}"
        results = run_trial(prices, seed, trial_dir, args, env_kwargs)
        for method, r in results.items():
            trial_results[method].append(r)

    summary = {"data": args.data, "n_hours": len(prices), "args": vars(args),
               "seeds": trial_seeds, "results": {}}

    print(f"\n=== Summary over {args.n_trials} trial(s) ===")
    for method, per_trial in trial_results.items():
        if not per_trial:
            continue
        key = "cumulative_profit" if method == "qlearning" else "final_mean_weekly_profit"
        values = [r[key] for r in per_trial]
        mean_v, std_v = float(np.mean(values)), float(np.std(values))
        label = "cumulative training profit" if method == "qlearning" else "final mean weekly profit"
        print(f"  {method}: {label} = ${mean_v:,.2f} (std ${std_v:,.2f}) over {len(values)} trial(s)")
        summary["results"][method] = {
            "n_trials": len(values),
            "trial_values": values,
            "mean": mean_v,
            "std": std_v,
        }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nRun artefacts written to {run_dir}")


if __name__ == "__main__":
    main()
