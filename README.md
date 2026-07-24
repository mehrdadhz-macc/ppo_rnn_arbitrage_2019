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

**Data pipeline and RL implementation both done, verified end-to-end on
real 2018 PJM data.** See "Known findings" below for actual results. Only
2018 has been trained/evaluated so far; 2016 and 2017 (the paper's other
two case-study years) are not yet run.

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

## Model

`src/environment.py` -- state `(E_t, c_t, rho_t, h_t)`: remaining battery
energy, a running average cost-basis for the energy currently stored
(Eq. 3 -- like FIFO/weighted-average-cost inventory accounting; realized
profit only shows up in the reward when discharging), the current price,
and an RNN hidden feature (see below). Bang-bang 3-action space
(charge/discharge at the max feasible rate, or hold), matching Wang &
Zhang's Lemma 1 that this paper cites directly. Reward is Eq. 4 --
economically meaningful on its own (no separate "shaped reward vs. true
profit" split needed, unlike some sibling projects in this collection: the
paper states the discharge-reward terms alone sum to total realized
arbitrage profit).

`src/price_encoder.py` -- an EMA filter (Eq. 6) feeding a single-layer RNN
(Eq. 7), trained via its own auxiliary next-price-prediction loss (Eq. 8),
entirely separate from the RL reward. Implemented with `nn.RNN` rather
than a manual per-timestep loop -- mathematically identical, but running
4000 steps of backprop-through-time over a ~6,500-hour training sequence in
a pure Python loop would be far too slow.

`src/ppo_agent.py` -- separate actor and critic networks (2 hidden layers,
128 and 32 units), GAE (Eq. 12), and the clipped surrogate objective
(Eq. 9-10) -- Algorithm 1. Each of the paper's K=200 outer updates collects
D=10 independent one-week (168h) trajectories from randomly sampled
starting points in the training data, battery reset to empty each time.

`src/qlearning_baseline.py` -- the paper's own comparison point: Wang &
Zhang's Q-learning discretized into 100 price bins x 10 energy bins (not
the 10x9 binning from `qlearning_realtime_arbitrage_2018` -- that was tuned
for a different paper's case study). Uses the identical environment/reward
as PPO, so the two are compared on equal footing.

## Train

```bash
venv/bin/python3 train.py --data data/train/pjm_rto_rt_hourly_lmp_2018_train.csv
```

Trains Q-learning, PPO, and PPO-RNN (in that order; PPO-RNN pretrains the
price encoder first). Takes roughly 35 minutes total at the paper's own
scale (4000 encoder steps, K=200 PPO updates) -- timed directly, not a
guess. Key flags: `--methods` (subset of `qlearning ppo ppo_rnn`, default
all three), `--n-updates`/`--n-trajectories`/`--traj-len` (K/D/T in
Algorithm 1), `--encoder-steps`, `--qlearning-price-bins`/
`--qlearning-energy-bins`, `--wear-cost` (beta in Eq. 4, $/MW).

`--n-trials N` runs N independent trials (each with its own seed, saved
under `trial_NN/`) and reports each method's mean +/- std profit across
trials -- see below for why single-trial numbers alone aren't trustworthy.
`--seed` doesn't feed into any trial directly; it seeds an RNG that
*generates* each trial's own seed, so the same `--seed` always reproduces
the same set of trial seeds.

## Evaluate

```bash
venv/bin/python3 evaluate.py --data data/test/pjm_rto_rt_hourly_lmp_2018_test.csv
```

Freezes every trial's trained models (greedy, no exploration) and replays
them on the held-out test split, reproducing the paper's Fig. 3-style
comparison. Reports each method's mean +/- std held-out profit across all
of the run's trials, and plots the mean curve with a +/- 1 std band per
method.

## Deviations / assumptions (where the paper is ambiguous)

- **Price encoder pretrained-then-frozen, not jointly trained with PPO.**
  Section III-A's auxiliary loss is presented independently of Section
  III-B's RL algorithm, and the paper doesn't state whether the two are
  trained end-to-end together or the encoder is fixed before RL training
  starts. Pretrain-then-freeze is the simpler reading and lets the hidden
  state be precomputed once per price series rather than recomputed inside
  every environment step (it depends only on the price sequence, never on
  the agent's actions).
- **Q-learning baseline's alpha/epsilon** aren't restated in this paper
  (only cited by reference to Wang & Zhang); this project uses that paper's
  own literal values (alpha=0.5, epsilon=0.9), but this paper's own
  gamma=0.999 (rather than Wang & Zhang's 0.9) so Q-learning and PPO
  optimize the same discounted objective on the same MDP.
- **Q-learning's price bin *method*** defaults to causal quantile bins
  (fit from a 30-day prefix), not equal-width -- the same reasoning as
  `qlearning_realtime_arbitrage_2018`'s README documents in more detail:
  real-time prices are heavy-tailed, and equal-width bins waste most of
  the table on rarely-visited spikes. `--qlearning-price-bin-method
  equal_width` is available for the more literal "100 price intervals"
  reading.
- **Seed sensitivity is real here too.** This collection's sibling project
  (`qlearning_realtime_arbitrage_2018`) rigorously confirmed with 100-trial
  paired testing that exploration/init noise can swing single-run results
  by an order of magnitude. The same pattern shows up here: re-running with
  `--seed 42` instead of the default 0 moved plain PPO's held-out profit
  from +$3,586 down to essentially $0 (its training curve broke out early
  that time, then collapsed late, instead of the reverse), and Q-learning
  from +$4,826 to +$2,514 -- on identical code and data. `train.py`/
  `evaluate.py` now support `--n-trials N` (mean +/- std across N
  independent seeded trials, saved under `trial_NN/`), matching the sibling
  project's approach; the numbers below still predate that flag (single
  trial each) and haven't yet been re-run at `--n-trials > 1`.

### Two implementation gaps found and fixed during a train/test audit

Prompted by held-out results not matching the paper's reported numbers,
both `src/price_encoder.py` and `evaluate.py` were re-audited line-by-line
against Eq. 6-8. Two real gaps (not just hyperparameter ambiguity) turned up:

- **RNN hidden state was reset to a cold start at the train/test boundary
  during evaluation.** `train.py` correctly computes the encoder's hidden
  state `h_t` continuously over the whole training price series (as it
  should -- `h_t` is meant to carry price-trend memory forward hour by
  hour). But `evaluate.py`'s `evaluate_ppo_rnn` was calling
  `compute_hidden_states` on the *test-only* price array, so the RNN
  started evaluation from `h_0` with zero memory of the 9 months of prices
  immediately preceding the test period -- a discontinuity that would never
  occur in real deployment. Fixed by computing hidden states over the
  training series concatenated with the test series, then using only the
  test-period slice to build the evaluation environment (`evaluate.py`
  now also loads the matching training CSV, inferred by filename
  convention or overridable with `--train-data`).
- **`h_0` was a hardcoded zero constant, not "randomly initialized" as the
  paper states.** `nn.RNN` silently defaults to a zero initial hidden state
  when none is passed. Fixed by making `h0` a learnable `nn.Parameter`
  (randomly initialized, then optimized jointly with the rest of the
  encoder during pretraining) -- the standard reading of "randomly
  initialized" for a network component.

Empirically, re-running training and evaluation with both fixes applied
changed PPO-RNN's held-out result from $9,635.09 to $9,287.94 (Q-learning
and plain PPO are unaffected, as expected -- neither uses the price
encoder's hidden states, and indeed their numbers were unchanged to the
cent). So these were real, worth-fixing deviations from the paper's stated
methodology, but empirically a *minor* contributor to the gap between this
replication's numbers and the paper's -- not the dominant explanation.
The likelier dominant sources of the remaining gap are the unstated
hyperparameters and single-seed noise already documented throughout this
section.

## Known findings from this replication

Trained on 2018 (Jan-Sep), evaluated on held-out 2018 (Oct-Dec), paper's
own default hyperparameters (K=200, D=10, T=168h, 4000 encoder steps,
gamma=0.999, wear cost beta=$1/MW):

|  | Training (mean profit, final updates) | Held-out test (cumulative) |
|---|---|---|
| Q-learning | -$5,590.66 (one online pass) | +$4,825.60 |
| PPO (no RNN) | ~$0 for ~160/200 updates, then breaks out to ~$340/week | +$3,586.16 |
| PPO-RNN | ~$700-1,300/week from update 10 onward | **+$9,287.94** |

(PPO-RNN's number reflects the two audit fixes above; Q-learning and PPO
are numerically unchanged from before the fixes, as expected.)

**PPO-RNN's central claim replicates cleanly**: it clearly and
substantially beats both other methods on held-out data (about 1.9x
Q-learning, 2.6x plain PPO), matching the paper's own claim that PPO-RNN
outperforms both by a wide margin. The *training-time* learning curves also
show the qualitative pattern the paper describes: plain PPO stays stuck
near zero profit for most of training before suddenly discovering a
profitable policy late (paper Fig. 2 shows the same flat-then-breakout
shape for PPO vs. Q-learning), while PPO-RNN finds a good policy almost
immediately, presumably because the RNN's price-trend feature gives it a
much easier signal to exploit than the raw instantaneous price alone.

**Where it doesn't match**: the paper's own reported numbers have plain
PPO beating Q-learning ($10,942 vs. $9,377 on the 2018 test quarter); this
run shows the reverse (Q-learning $4,826 vs. PPO $3,586). Given the
single-seed caveat above, this specific ordering flip between the two
weaker methods isn't necessarily a real disagreement with the paper --
it's exactly the kind of comparison this project's own sibling found to be
unreliable without multiple seeds. PPO-RNN's win over both is large enough
in this run to be a more believable signal on its own, but multi-seed
testing (matching `qlearning_realtime_arbitrage_2018`'s pattern) is the
natural next step before treating any of these numbers as solid.

## What's next

- Actually run `--n-trials > 1` at the paper's full scale (now supported,
  but not yet exercised beyond a quick smoke test) and update the table
  above with mean +/- std figures, before treating the Q-learning-vs-PPO
  ordering as settled either way.
- Run 2016 and 2017 (the paper's other two case-study years) the same way.
- Compare against this paper's own reported profit figures more directly
  once multiple seeds are available.
