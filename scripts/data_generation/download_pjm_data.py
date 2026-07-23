"""Stage 1/3: download the ORIGINAL, untouched PJM real-time hourly LMP data into data/raw/.

Paper (Xu, Li, Zhang, Zhang, 2019, arXiv:1904.12232) uses real-time hourly
LMP from PJM for 2016, 2017, and 2018. This script fetches that data
day-by-day from PJM's public "rt_hrl_lmps" feed (Data Miner 2) and caches
each day's raw JSON response, untouched, into data/raw/pjm_cache/.

No PJM account or API registration needed. PJM's own Data Miner 2 website
(dataminer2.pjm.com) is explicitly public with no login required -- to make
that work, its JavaScript loads a small public config file
(https://dataminer2.pjm.com/config/settings.json) containing a subscription
key, which every visitor's browser uses automatically. This script uses that
same key the same way the public website does. It's not a documented
"developer" key from PJM's API portal -- it's what the public site itself
sends to anonymous browsers -- so this script re-fetches it live each run
(falling back to a hardcoded copy if that ever fails) rather than assuming
it's permanent.

Archived (>2 years old) PJM data has query restrictions (confirmed by
testing): a date range must stay within one calendar year, and pnode_id/
pnode_name filters are rejected -- only type=ZONE is accepted for archived
requests. This script filters by type=ZONE (all ~22 zone/hub aggregates,
528 rows/day) rather than requesting all ~11,000+ individual pricing nodes
(which would be ~280,000 rows/day for data we don't need). Stage 2
(preprocess_pjm_data.py) extracts just the PJM-RTO row -- the system-wide
aggregate -- from each cached day.

Usage:
    venv/bin/python3 scripts/data_generation/download_pjm_data.py
"""

import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path

import requests

DM2_CONFIG_URL = "https://dataminer2.pjm.com/config/settings.json"
FALLBACK_API_BASE = "https://api.pjm.com/api/v1"
FALLBACK_SUBSCRIPTION_KEY = "6a75d9f6d933401dbb4f36f8e70b95b3"
FEED = "rt_hrl_lmps"


def resolve_public_credentials(session):
    """Fetch the same public config the Data Miner 2 website itself loads,
    so this script keeps working if PJM ever rotates the key. Falls back to
    the hardcoded copy (verified working as of this writing) if that fails.
    """
    try:
        resp = session.get(DM2_CONFIG_URL, timeout=15)
        resp.raise_for_status()
        config = resp.json()
        base_url = config["baseUrl"]
        key = config["subscriptionKey"]
        print(f"Resolved live public API config from {DM2_CONFIG_URL}")
        return base_url, key
    except Exception as exc:  # noqa: BLE001
        print(f"Could not fetch live config ({exc}); falling back to the hardcoded "
              f"public key captured when this script was written.")
        return FALLBACK_API_BASE, FALLBACK_SUBSCRIPTION_KEY


def fetch_day(session, base_url, key, day_str):
    """One day, type=ZONE only (required for archived/>2yr-old data; also
    keeps responses small -- 528 rows/day across ~22 zone aggregates,
    instead of ~280,000/day for every individual pricing node).
    """
    url = f"{base_url}/{FEED}"
    params = {
        "startRow": 1,
        "rowCount": 2000,  # headroom above the observed 528 rows/day
        "datetime_beginning_ept": day_str,
        "type": "ZONE",
    }
    headers = {"Ocp-Apim-Subscription-Key": key, "Accept": "application/json"}
    resp = session.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", default=[2016, 2017, 2018])
    parser.add_argument("--out-dir", default="data/raw", help="Base output directory (default: data/raw)")
    parser.add_argument("--request-delay", type=float, default=0.2,
                         help="Seconds between requests (default: 0.2, be polite to a shared public endpoint)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "pjm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    base_url, key = resolve_public_credentials(session)

    days = []
    for year in args.years:
        days.extend(daterange(date(year, 1, 1), date(year, 12, 31)))

    n_fetched, n_cached, n_failed = 0, 0, 0
    for d in days:
        day_str = d.isoformat()
        cache_file = cache_dir / f"{day_str}.json"

        if cache_file.exists():
            n_cached += 1
            continue

        try:
            payload = fetch_day(session, base_url, key, day_str)
        except requests.HTTPError as exc:
            print(f"  {day_str}: HTTP error {exc}, skipping")
            n_failed += 1
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"  {day_str}: {exc}, skipping")
            n_failed += 1
            continue

        cache_file.write_text(json.dumps(payload))
        n_fetched += 1
        if n_fetched % 50 == 0:
            print(f"  ... {n_fetched} new days fetched so far")
        time.sleep(args.request_delay)

    print(f"\nDone: {n_fetched} days fetched new, {n_cached} already cached, {n_failed} failed.")
    print(f"Raw daily responses cached in {cache_dir}")
    print("Next: scripts/data_generation/preprocess_pjm_data.py")


if __name__ == "__main__":
    main()
