"""Load PJM-RTO hourly LMP CSVs produced by scripts/data_generation/*.py."""

import pandas as pd


def load_price_series(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df["timestamp"].to_numpy(), df["lmp_total"].to_numpy(dtype=float)
