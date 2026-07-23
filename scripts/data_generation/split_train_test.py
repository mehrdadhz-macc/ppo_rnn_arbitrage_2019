"""Stage 3/3: split the cleaned combined series into per-year train/test CSVs.

Paper's own split (Sec. IV): "Electricity prices from the first 9 months and
the last 3 months are used as the training and testing data," repeated
independently for each of 2016, 2017, and 2018 -- three separate
train/test experiments, not one combined 2016-2018 split.

The clean CSV's timestamp column is UTC (see preprocess_pjm_data.py's
docstring for why). Splitting directly on the UTC month would misclassify
the last few hours of September as October (UTC is a few hours ahead of
Eastern), so year/month here are computed from the Eastern-local view of
each timestamp -- used only to decide which split an hour belongs to, not
stored back into the output (the output stays in UTC).

Usage:
    venv/bin/python3 scripts/data_generation/split_train_test.py
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clean-csv", default="data/raw/pjm_rto_rt_hourly_lmp_2016_2018_clean.csv",
                         help="Output of preprocess_pjm_data.py")
    parser.add_argument("--years", type=int, nargs="+", default=[2016, 2017, 2018])
    parser.add_argument("--out-dir", default="data", help="Base output directory (default: data)")
    args = parser.parse_args()

    clean_path = Path(args.clean_csv)
    if not clean_path.exists():
        raise SystemExit(f"{clean_path} not found -- run preprocess_pjm_data.py first.")

    df = pd.read_csv(clean_path, parse_dates=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    eastern_local = df["timestamp"].dt.tz_convert("America/New_York")
    out_base = Path(args.out_dir)

    for year in args.years:
        year_mask = eastern_local.dt.year == year
        year_df = df[year_mask]
        if year_df.empty:
            print(f"  {year}: no rows in {clean_path}, skipping")
            continue

        month = eastern_local[year_mask].dt.month
        train_df = year_df[month <= 9]
        test_df = year_df[month >= 10]

        train_path = out_base / "train" / f"pjm_rto_rt_hourly_lmp_{year}_train.csv"
        test_path = out_base / "test" / f"pjm_rto_rt_hourly_lmp_{year}_test.csv"
        train_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.parent.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        print(f"  {year}: train (Jan-Sep) {len(train_df)} rows -> {train_path}")
        print(f"        test (Oct-Dec)  {len(test_df)} rows -> {test_path}")


if __name__ == "__main__":
    main()
