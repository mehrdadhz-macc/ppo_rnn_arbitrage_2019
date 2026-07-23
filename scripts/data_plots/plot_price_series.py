"""Plot the raw price series (+ EMA-smoothed overlay) for every train/test split.

Auto-discovers every CSV under data/train/ and data/test/ (however many
years are present) and produces one plot per file, so it doesn't need
updating as more years get downloaded. The EMA overlay uses this project's
own alpha=0.7 default (src/price_encoder.py, Eq. 6) -- the same smoothing
the RNN price encoder actually sees, not just a generic moving average.

Usage:
    venv/bin/python3 scripts/data_plots/plot_price_series.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_loader import load_price_series
from src.price_encoder import compute_ema


def plot_one(csv_path, alpha, out_dir):
    _, prices = load_price_series(csv_path)
    ema = compute_ema(prices, alpha)

    plt.figure(figsize=(10, 4))
    plt.plot(prices, label="Real-time price (PJM-RTO)", linewidth=0.7)
    plt.plot(ema, label=f"EMA-smoothed (alpha={alpha})", linewidth=1.2)
    plt.xlabel("Time (hour)")
    plt.ylabel("Price ($/MWh)")
    plt.title(f"PJM-RTO real-time hourly price ({csv_path.stem})")
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / f"{csv_path.stem}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  {csv_path} ({len(prices)} hours, min={prices.min():.2f}, max={prices.max():.2f}) -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data", help="Base data directory (default: data)")
    parser.add_argument("--out-dir", default="outputs/data_plots", help="Where to save plots")
    parser.add_argument("--alpha", type=float, default=0.7, help="EMA smoothing (Eq. 6 default)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted((data_dir / "train").glob("*.csv")) + sorted((data_dir / "test").glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSVs found under {data_dir}/train or {data_dir}/test -- run the data pipeline first.")

    for csv_path in csv_files:
        plot_one(csv_path, args.alpha, out_dir)


if __name__ == "__main__":
    main()
