"""Combine the raw cached daily JSON into one CSV with EVERY column and EVERY
zone/hub node PJM returned -- not just the single PJM-RTO `lmp_total` series
preprocess_pjm_data.py extracts for training. This is purely for inspecting
what the API actually returns; it's not part of the required pipeline
(download -> preprocess -> split).

Columns kept, straight from PJM's own field names (see the "rt_hrl_lmps"
feed definition at https://dataminer2.pjm.com/feed/rt_hrl_lmps/definition):

    datetime_beginning_utc / datetime_beginning_ept  -- hour start, UTC and Eastern local
    pnode_id / pnode_name                             -- which zone/hub (22 of them; PJM-RTO is the system-wide one)
    voltage / equipment / type / zone                 -- node metadata (mostly null for zone-level aggregates)
    system_energy_price_rt                             -- system marginal energy price component
    total_lmp_rt                                       -- the full real-time LMP (energy + congestion + losses)
    congestion_price_rt / marginal_loss_price_rt        -- the other two LMP components
    row_is_current / version_nbr                        -- PJM's own settlement-revision bookkeeping

Usage:
    venv/bin/python3 scripts/data_generation/combine_raw_data.py
"""

import argparse
import json
from pathlib import Path

import pandas as pd

KEEP_COLUMNS = [
    "datetime_beginning_utc", "datetime_beginning_ept",
    "pnode_id", "pnode_name", "voltage", "equipment", "type", "zone",
    "system_energy_price_rt", "total_lmp_rt", "congestion_price_rt", "marginal_loss_price_rt",
    "row_is_current", "version_nbr",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", default=[2016, 2017, 2018])
    parser.add_argument("--raw-dir", default="data/raw", help="Where the raw cached days live (default: data/raw)")
    parser.add_argument("--out", default=None, help="Override the output CSV path")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    cache_dir = raw_dir / "pjm_cache"

    day_files = []
    for year in args.years:
        day_files.extend(sorted(cache_dir.glob(f"{year}-*.json")))
    if not day_files:
        raise SystemExit(f"No cached days found in {cache_dir}/ -- run download_pjm_data.py first.")

    rows = []
    for day_file in day_files:
        payload = json.loads(day_file.read_text())
        for item in payload.get("items", []):
            rows.append({col: item.get(col) for col in KEEP_COLUMNS})

    df = pd.DataFrame(rows, columns=KEEP_COLUMNS)
    df["datetime_beginning_utc"] = pd.to_datetime(df["datetime_beginning_utc"])
    df = df.sort_values(["datetime_beginning_utc", "pnode_id"]).reset_index(drop=True)

    out_path = Path(args.out) if args.out else raw_dir / f"pjm_zone_raw_full_{min(args.years)}_{max(args.years)}.csv"
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} rows x {len(KEEP_COLUMNS)} columns to {out_path}")
    print(f"  Distinct zone/hub nodes: {df['pnode_name'].nunique()} -- {sorted(df['pnode_name'].unique())}")
    print(f"  Date range: {df['datetime_beginning_utc'].min()} -> {df['datetime_beginning_utc'].max()}")


if __name__ == "__main__":
    main()
