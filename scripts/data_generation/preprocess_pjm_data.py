"""Stage 2/3: turn the raw cached PJM daily responses into one clean hourly price series.

Reads the raw per-day JSON files download_pjm_data.py cached, extracts the
PJM-RTO row (pnode_id=1, the system-wide aggregate real-time LMP -- what the
paper's own single price series represents) from each day's ~22-zone
response, combines all years into one series, and validates the result
before writing it back out as a clean CSV. "Clean" here means standard
column names (timestamp, lmp_total) and validated structure -- it does NOT
mean adjusting PJM's own settled LMP values.

Uses `datetime_beginning_utc`, not `datetime_beginning_ept`, as the
timestamp. PJM's EPT (Eastern Prevailing Time) field is real local
wall-clock time, which is genuinely ambiguous on the November DST
"fall back" day: 1:00-2:00 AM local happens twice, and both occurrences
are labeled "01:00:00" in EPT with two different real prices. An earlier
version of this script deduplicated on the EPT string and silently
discarded one of those two real hours every year -- caught because the
per-year row-count check came up exactly 1 short each year. UTC has no
such ambiguity, so it's the only safe dedup/sort key.

Note this shifts the *reported* clock time for each row (UTC is a few
hours ahead of Eastern), so per-year row counts are checked against each
year's own source files (grouped by the EPT calendar day download_pjm_data.py
fetched them under) rather than by re-deriving "year" from the UTC
timestamp after the fact -- a UTC year boundary falls a few hours before
the EPT one, so checking post-concatenation would misattribute a handful
of hours at each year's edge to the wrong year.

Usage:
    venv/bin/python3 scripts/data_generation/preprocess_pjm_data.py
"""

import argparse
import json
from pathlib import Path

import pandas as pd

TARGET_PNODE_NAME = "PJM-RTO"
EXPECTED_HOURS_PER_YEAR = {2016: 8784, 2017: 8760, 2018: 8760}  # 2016 is a leap year


def load_raw_year(year, raw_dir):
    cache_dir = raw_dir / "pjm_cache"
    day_files = sorted(cache_dir.glob(f"{year}-*.json"))
    if not day_files:
        raise SystemExit(f"No cached days found for {year} in {cache_dir}/ -- run download_pjm_data.py first.")

    rows = []
    for day_file in day_files:
        payload = json.loads(day_file.read_text())
        for item in payload.get("items", []):
            if item.get("pnode_name") == TARGET_PNODE_NAME:
                rows.append({
                    "timestamp": item["datetime_beginning_utc"],
                    "lmp_total": float(item["total_lmp_rt"]),
                })

    if not rows:
        raise SystemExit(f"No {TARGET_PNODE_NAME} rows found in any cached day for {year}.")

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)

    expected = EXPECTED_HOURS_PER_YEAR.get(year)
    if expected is not None and len(df) != expected:
        raise SystemExit(f"{year}: expected {expected} hourly rows (fetched across the EPT calendar "
                          f"year, deduped on UTC timestamp), got {len(df)}. This is a real data gap "
                          f"or duplicate, not just a DST edge case (those are already handled) -- "
                          f"check data/raw/pjm_cache/{year}-*.json for missing or malformed days.")
    return df


def validate(df):
    problems = []

    n_dupes = df["timestamp"].duplicated().sum()
    if n_dupes:
        problems.append(f"{n_dupes} duplicate timestamps across the combined series")

    n_missing = df["lmp_total"].isna().sum()
    if n_missing:
        problems.append(f"{n_missing} missing lmp_total values")

    if problems:
        raise SystemExit("Validation failed:\n  - " + "\n  - ".join(problems))
    print(f"Validation OK: {len(df)} rows total, no duplicates, no missing values "
          f"(each source year's row count already checked individually against the expected "
          f"leap/non-leap total).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", default=[2016, 2017, 2018])
    parser.add_argument("--raw-dir", default="data/raw", help="Where the raw cached days live (default: data/raw)")
    parser.add_argument("--out", default=None, help="Override the output CSV path")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    frames = [load_raw_year(year, raw_dir) for year in args.years]
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    validate(df)

    out_path = Path(args.out) if args.out else raw_dir / f"pjm_rto_rt_hourly_lmp_{min(args.years)}_{max(args.years)}_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote cleaned series ({len(df)} rows, {df['timestamp'].min()} -> {df['timestamp'].max()}) to {out_path}")
    print("Next: scripts/data_generation/split_train_test.py")


if __name__ == "__main__":
    main()
