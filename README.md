# PPO + RNN Real-Time Arbitrage (PJM)

Replication of Xu, H., Li, X., Zhang, X., & Zhang, J. (2019). [*Arbitrage of
Energy Storage in Electricity Markets with Deep Reinforcement
Learning*](https://arxiv.org/abs/1904.12232). arXiv:1904.12232.

The paper controls a battery storage system for arbitrage within PJM's
real-time market. It formulates the problem as an MDP whose state includes
not just the current price and remaining battery energy, but a hidden state
extracted from the price *sequence* by an EMA filter + RNN (so the agent can
react to recent price trends, not just the instantaneous price), and solves
it with **Proximal Policy Optimization (PPO)**. The paper benchmarks this
directly against Wang & Zhang's tabular Q-learning (the paper already
replicated in `qlearning_realtime_arbitrage_2018` in this same collection),
reporting PPO+RNN beats it by roughly 40% across three separate test years
(2016, 2017, 2018).

## Status

**Data pipeline: done and verified.** Model implementation (environment,
reward, PPO+RNN agent, train/evaluate scripts) is not started yet -- this
README describes the paper and the data as they stand today; the RL parts
below are the paper's design, not yet this repo's code.

## Data

Real PJM real-time hourly LMP for the **PJM-RTO** node (the system-wide
load-weighted aggregate price -- what the paper's own single price series
represents), for 2016, 2017, and 2018 (the paper's own years).

```bash
venv/bin/pip install -r requirements.txt

venv/bin/python3 scripts/data_generation/download_pjm_data.py    # 1/3: fetch raw daily data
venv/bin/python3 scripts/data_generation/preprocess_pjm_data.py  # 2/3: extract PJM-RTO, clean, validate
venv/bin/python3 scripts/data_generation/split_train_test.py     # 3/3: split each year train/test
```

**No PJM account or API key needed.** PJM's Data Miner 2 website
(dataminer2.pjm.com) is explicitly public with no login required. To make
that work, its own JavaScript loads a small public config file
(`https://dataminer2.pjm.com/config/settings.json`) containing an API
subscription key that every visitor's browser uses automatically -- this
project's download script fetches that same config live each run (falling
back to a hardcoded copy of the key if that ever fails) and uses it exactly
the way the public website itself does, rather than assuming it's
permanent. See `scripts/data_generation/download_pjm_data.py`'s docstring
for the full explanation, and the "Finding this in the GUI" section below if
you'd rather pull it by hand.

**Data node**: PJM classifies every pricing node as `LOAD` (~11,000+
individual buses) or `ZONE` (~22 zone/hub-level aggregates, including
`PJM-RTO`). Historical (>2 years old) queries reject direct `pnode_id`/
`pnode_name` filters (tested directly, confirmed by PJM's own error
message), so the download script requests `type=ZONE` (528 rows/day across
all 22 zones) and the preprocessing step picks out just the `PJM-RTO` row --
far cheaper than requesting all ~11,000+ individual nodes (~280,000
rows/day) for data we don't need.

**Train/test split** matches the paper exactly (Sec. IV): for each of 2016,
2017, and 2018 independently, the first 9 months are training data and the
last 3 months are test data -- three separate train/test experiments, not
one combined 2016-2018 split.

### Two bugs found and fixed while building this (both from the same root cause)

PJM's `datetime_beginning_ept` field is real Eastern local wall-clock time,
which is genuinely ambiguous on the November DST "fall back" day (1:00-2:00
AM happens twice, with two different real prices, both labeled `01:00:00`).
An early version of the preprocessing script deduplicated on that ambiguous
field and silently dropped one real hour per year -- caught because the
per-year row-count validation came up exactly 1 short every time. Fixed by
using `datetime_beginning_utc` (unambiguous) as the canonical timestamp
instead. A second, subtler version of the same issue showed up in the
train/test split: splitting directly on the UTC month would have
misclassified the last few hours of each September as October, since UTC
runs a few hours ahead of Eastern time. Fixed by converting to Eastern time
only to decide which month/year an hour belongs to, while keeping the
stored data in UTC.

### Inspecting the full raw data

`scripts/data_generation/combine_raw_data.py` is a standalone, optional
script (not part of the required pipeline) that combines the raw cached
daily JSON into one CSV with **every** column and **all 23** zone/hub nodes
PJM returns -- not just the single `PJM-RTO`/`lmp_total` series used for
training. Useful for seeing what the API actually returns (energy vs.
congestion vs. loss components, PJM's own settlement-revision fields, and
how prices differ node-to-node in the same hour):

```bash
venv/bin/python3 scripts/data_generation/combine_raw_data.py
```

### Finding this data in the Data Miner 2 GUI (no script needed)

1. Go to https://dataminer2.pjm.com/feed/rt_hrl_lmps (or search "Real-Time
   Hourly LMPs" from the homepage) and click **Explore Data Set**.
2. Set the date range. PJM's own restriction on data older than 2 years:
   **the range must stay within a single calendar year** (e.g.
   `2016-01-01` to `2016-12-31`, not spanning into 2017).
3. Filter the **"Pricing Node Type"** column to **`ZONE`** -- restricts
   results to the ~22 zone/hub aggregates instead of all ~11,000+
   individual nodes (which would blow past the tool's 1-million-row CSV
   export cap for a full year).
4. Click **Submit**, then filter the **"Pricing Node Name"** column for
   **`PJM-RTO`** to isolate the system-wide aggregate.
5. Click the **CSV** button to export (one year at a time is comfortably
   under the row cap: ~528 rows/day x 365 ~ 190K rows).

No account or login is required for any of this.

## Project structure

```
scripts/
  data_generation/
    download_pjm_data.py      # 1/3: fetch raw daily PJM data -> data/raw/pjm_cache/
    preprocess_pjm_data.py    # 2/3: extract PJM-RTO, clean, validate -> data/raw/*_clean.csv
    split_train_test.py       # 3/3: split each year Jan-Sep/Oct-Dec -> data/train/, data/test/
    combine_raw_data.py       # optional: full-column, all-zone raw CSV for inspection

data/    # not tracked in git -- rebuild locally via the scripts above
  raw/   pjm_cache/ (per-day JSON), the clean combined CSV, the full raw export
  train/ pjm_rto_rt_hourly_lmp_<year>_train.csv (Jan-Sep, per year)
  test/  pjm_rto_rt_hourly_lmp_<year>_test.csv  (Oct-Dec, per year)
```

## What's next

The RL side of this replication (not yet built): the MDP environment
(state = remaining energy, average energy cost basis, current price, and
the EMA+RNN hidden state; 3-action bang-bang charge/hold/discharge, per
Lemma 1 in the earlier Wang & Zhang paper this project's data loader shares
lineage with), the PPO agent with an RNN price-sequence encoder, and
train/evaluate scripts that reproduce the paper's three-year benchmark
against tabular Q-learning.
