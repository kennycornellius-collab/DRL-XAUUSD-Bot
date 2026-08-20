# DRL XAUUSD Trading Bot

A Deep Reinforcement Learning trading agent for Gold (XAUUSD) on the M15 timeframe, built using **Soft Actor-Critic (SAC)** and trained with a custom **Walk-Forward Optimization (WFO)** pipeline. Includes a live execution bridge via MetaTrader 5 and a synchronized Macroeconomic Risk Circuit Breaker.

> **Disclaimer:** This project is for educational and portfolio purposes only. It is not financial advice and should not be used with real capital without thorough due diligence.

---

## Overview

Most DRL trading projects train once on a fixed historical window and call it done. This project takes a different approach — it uses a rolling walk-forward training loop that continuously fine-tunes the agent week-by-week on fresh out-of-sample data, mimicking how a real adaptive system would behave in a live market.

The agent trades Gold (XAUUSD) on the M15 timeframe using Soft Actor-Critic (SAC) with a continuous action space — position size scales with the policy's own conviction rather than snapping to a fixed lot size — operating on a 14-dimensional observation vector combining normalized market features (price/volume, ADX, ATR, RSI, a higher-timeframe ADX, and cyclical time-of-day encoding) with live account state (position, unrealized PnL, current drawdown). A custom `WeeklyRollingBuffer` ensures the agent only trains on recent market regimes by purging the oldest week's experience at each rollover, preventing it from overfitting to stale conditions.

The reward is the trade's raw dollar PnL each step, and a 20% cumulative drawdown hard stop terminates episodes early — capital preservation is enforced structurally rather than through reward shaping (see [Reward Function](#reward-function) for why an earlier risk-adjusted-ratio reward was replaced with this simpler one).

**Bottom line: after six rounds of methodologically rigorous experimentation — hyperparameter tuning, continuous position sizing, a richer feature set, wider reward horizons, seed ensembling, and a reward-signal audit — the agent is not reliably profitable under any configuration tried.** Net profit's standard deviation across seeds consistently swamps its mean, in both validation and held-out test segments, and no configuration has come close to matching simple buy-and-hold of the underlying asset. This is formally documented as the project's current, accepted finding rather than an open problem still being chased — see [Hyperparameter Experiment Results](#hyperparameter-experiment-results) for the full investigation timeline and the reasoning behind that call.

---

## Multi-Repo Ecosystem

This bot is the execution layer of a three-repository automated quantitative system. The data extraction and sentiment analysis layers run independently via cloud automation, feeding this bot a fresh risk assessment without any manual intervention.

```
[forex-news-scraper]          (GitHub Actions — runs every Monday)
  Scrapes ForexFactory → filters high impact USD events → commits CSV to repository
       │
       ▼
[macro-analyzer]              (GitHub Actions — runs every Monday @ 00:30 UTC)
  Pulls filtered data from Repo 1
  Sends events to Google Gemini API → outputs regime.json with expected_volatility
  Commits regime.json to repository & broadcasts to Discord
       │
       ▼ (fetched via GitHub raw URL before each M15 candle decision)
[drl-xauusd]                  (this repository)
  Reads expected_volatility → if "High" or "Extreme" → CIRCUIT_BREAKER_FLAT
```

---

## The Macro Circuit Breaker

One of the most well-known failure modes of algorithmic trading is the inability to navigate unpredictable economic news releases (CPI, FOMC, NFP). These events cause sharp, erratic price spikes that violate the statistical patterns the agent was trained on.

After computing the DRL action each candle, the MT5 bridge fetches the latest `regime.json` from the macro analyzer repository. If the `expected_volatility` is `"High"` or `"Extreme"`, the bridge overrides the neural network's decision and forces a flat position.

The check runs on every M15 candle close, not just at startup. If the macro regime updates mid-session, the bot will pick it up on the next candle.

---

## Experiment: State-Space Macro Integration

**Hypothesis:** If the agent is penalized for holding trades during high-impact news windows during training, it will learn to organically flatten its position and avoid volatility spikes — producing a higher risk-adjusted return than a deterministic circuit breaker.

**Methodology:**
- Sourced historical Forex Factory data (2023–present) via Hugging Face.
- Built `merge_news.py` to convert timezones to UTC, filter for USD high-impact events, and merge a boolean `news_flag` into the M5/M15 OHLCV dataset.
- Expanded the SAC observation space to `Box(10,)` to include the macro flag.
- Modified the reward function to heavily penalize the agent for opening or holding positions during `news_flag == 1`.

**Results (Walk-Forward Out-of-Sample):**

| Setup | Net Profit | Max Drawdown |
|---|---|---|
| Baseline SAC (no macro data) | +24.65% | 11.05% |
| Experimental SAC (macro state integration) | -27.11% | 30.68% |

> Both rows use the same single-window, single-seed methodology later found unreliable
> (see [Hyperparameter Experiment Results](#hyperparameter-experiment-results)). The
> relative conclusion below (macro integration made things worse) is likely still valid,
> but neither absolute number should be read as a robust estimate of real performance.

**Conclusion:**

Integrating macro data directly into the RL state space severely degraded performance. The agent developed what could be described as "scared agent syndrome" — closing highly profitable, long-running trend trades prematurely simply because a news event appeared on the horizon. This caused significant spread bleed from paying double broker fees to re-enter interrupted trends, and distorted the reward signal.

DRL agents excel at continuous, flowing data like technical price action but struggle with sparse, binary anomalies like news drops. The signal is too rare and too irregular for the agent to learn a clean policy around it.

**The conclusion is separation of concerns.** The SAC model is left completely blind to macro events, allowing it to optimize purely for technical price action. Risk management is decoupled and handled deterministically by `mt5bridge.py`, which halts trading via API polling during high volatility windows. The experimental training script is preserved in `xau_macro.py` for reference.

---

## Architecture

### Observation Space — `Box(14,)`

| Index | Feature | Description |
|-------|---------|-------------|
| 0–8 | Market Features | `open, high, low, close, volume, adx, atr, rsi, adx_h1` — normalized via rolling 200-candle min-max to `[-1, 1]` (`adx`/`atr`/`rsi` computed on M5 and backward-merged onto M15; `adx_h1` computed on H1 bars and merged with a +1h shift so no not-yet-closed H1 data leaks in — see `features.py`) |
| 9 | Position | Continuous, `[-1, 1]` — see Action Space below |
| 10 | Unrealized PnL | Scaled by ÷100 and clamped to `[-1, 1]` |
| 11 | Drawdown % | Current drawdown from peak balance, clamped to `[-1, 1]` |
| 12–13 | Session Time | `sin`/`cos` encoding of hour-of-day, so e.g. 23:45 and 00:00 stay numerically adjacent instead of wrapping discontinuously |

### Action Space — `Box(1,)` continuous `[-1, 1]`

The action maps to position size, not just direction:
- `[-0.3, 0.3]` → **Flat** (a deadzone so the agent isn't forced to always hold a position)
- Outside the deadzone, position size tracks the action value proportionally — a barely-confident `0.31` and a maximally-confident `0.99` get proportionally different exposure, instead of both snapping to the same full-size position. At `|action| == 1.0` this is a full-size position, same as before this was added.

An earlier version hard-thresholded the action into exactly three states (full short / flat / full long), discarding the policy's own conviction signal entirely. Scaling position size continuously was the first and only Next-Priority direction (see [Hyperparameter Experiment Results](#hyperparameter-experiment-results)) that produced a standing, uncomplicated improvement — reduced seed variance and a better validation mean than the hard-thresholded baseline.

> **Note:** the live bridge (`mt5bridge.py`) does not yet mirror this — it still thresholds the action into a binary long/short/flat decision and always executes at a fixed lot size (`FIXED_LOTS`). This is a known train/live gap; see [Limitations & Future Work](#limitations--future-work).

### Reward Function

The reward is the step's raw dollar PnL:

```
reward = step_return   # price_pnl - spread_cost
```

with no per-step normalization or drawdown penalty. A **20% cumulative drawdown hard stop** (tracked against the true running peak balance across the whole walk-forward run, not reset each week) still terminates the episode early — that's the risk boundary, enforced structurally rather than through the reward.

This replaced an earlier Sortino-like ratio (`step_pnl / (downside_std + ε)`, plus a penalty when drawdown exceeded 5%) that recomputed its normalization from a sliding 50-step window every step. That formula could disagree with the week's actual PnL in both sign and magnitude — real training logs showed profitable weeks with negative reward and losing weeks with positive reward. Switching to raw PnL fixed that decoupling (verified: reward now sums to exactly the episode's net PnL by construction) but did **not** improve trading outcomes — see the reward-signal-audit entry in [Hyperparameter Experiment Results](#hyperparameter-experiment-results) for the full result and why the simpler, bug-free reward was kept anyway over reverting to the known-broken one.

---

## Walk-Forward Optimization Pipeline

```
[M5 Raw CSV]
     │
     ▼
[Data Engineering]
  • Compute ADX/ATR/RSI(14) on M5, H1 ADX on resampled H1
  • Resample M5 → M15, backward-merge indicators (no lookahead)
  • Tag ISO week labels (week-based year, not calendar year)
     │
     ▼
[Pre-Training Phase]
  • 26 weeks of history
  • SAC agent, net_arch=[128, 128]
  • Custom WeeklyRollingBuffer
     │
     ▼
[Walk-Forward Fine-Tuning]  ← loops week by week →
  • Collect experience (no gradients)
  • Maintain 4-week sliding replay buffer
  • Fine-tune with 1,000 gradient steps
  • Save model checkpoint per week
     │
     ▼
[Out-of-Sample Backtest Summary]
```

### Custom `WeeklyRollingBuffer`

A custom replay buffer that tags every experience with an **ISO week label**. At each rollover, the oldest week's data is purged — ensuring the agent only learns from recent, relevant market regimes and preventing it from overfitting to stale market conditions.

---

## Hyperparameter Experiment Results

### Final Status (2026-08-20): Not Reliably Profitable — Investigation Concluded

Starting from the original hyperparameter-tuning result below, five further experimental
directions were tried (numbered 1–4 and 6 in the [Investigation Timeline](#investigation-timeline);
#5 is the decision, formalized here, to accept this as the finding), each carried
forward only if it produced a real, standing improvement. Every
number below is a 5-seed mean ± standard deviation on the same chronological
validation/test split ([methodology](#hyperparameter-validation-methodology)):

| Era | What changed | Validation | Test |
|---|---|---|---|
| 0. Original hyperparameter tuning | 6-config sweep, old Sortino-like reward | -8.95% ± 9.66% | +0.71% ± 7.72% |
| 1. Position sizing | Continuous sizing instead of hard-thresholded ±1 | -1.58% ± 7.11% | -0.13% ± 6.91% |
| 2. Richer features | + ATR, RSI, H1 ADX, session encoding (9→14-dim obs) | **+0.77% ± 6.09%** | -3.65% ± 5.68% |
| 3. Reward horizon (200-step) | Widened the old reward's downside window | -7.63% ± 2.45% | -0.89% ± 9.31% |
| 3. Reward horizon (100-step) | Same, narrower — both retired, reverted | -1.10% ± 5.34% | -2.10% ± 4.73% |
| 6. Raw-PnL reward | Reward audit fix: reward = raw step PnL | -5.92% ± 3.82% | -6.69% ± 8.01% |
| **Buy-and-hold** (benchmark) | Just holding gold over the same window | **+64.26%** | **+35.72%** |

Era 2 (position sizing + richer features) has the best validation mean of any
configuration tried and is the current code baseline — but even there, standard
deviation is ~8x the mean, test net profit is negative, and it isn't close to
buy-and-hold. A 4th direction, seed ensembling at inference (averaging several trained
seeds' actions), reduced drawdown substantially wherever it was tried but only
inconsistently improved net profit — see the timeline below.

**Conclusion: the agent is not reliably profitable, under any configuration tried, on
this data and time horizon.** This is now formally documented as the project's current,
accepted finding (Next-Priority direction #5, "accept the finding") rather than an open
problem still being actively chased — six methodologically distinct interventions
(hyperparameters, position sizing, features, reward horizon, seed ensembling, reward
signal fidelity) were each tried in isolation and evaluated with the same
seed-variance-aware protocol, and none closed the gap to a consistently learned edge, let
alone to simple buy-and-hold. Two untried reward-shaping ideas remain on the table for
future work (a differential Sharpe/Sortino ratio, or raw PnL plus a terminal-only risk
bonus) — see [Limitations & Future Work](#limitations--future-work) — but pursuing them
is a deliberate choice to open a new, unvalidated direction, not a continuation of
anything already in progress.

### Investigation Timeline

#### 0. Original hyperparameter tuning (validation/test split + 5-seed variance)

Six candidate configs were trained and walk-forward-backtested, then ranked using only
a validation segment of the walk-forward window (see
[Hyperparameter Validation Methodology](#hyperparameter-validation-methodology) below):

| Pretrain Steps | Gradient Steps | Batch Size | Validation Net Profit | Validation Max DD |
|---|---|---|---|---|
| **150k** | **1,000** | **256** | **+8.18%** | **3.33%** |
| 100k | 1,000 | 256 | +5.78% | 2.29% |
| 50k | 1,000 | 256 | +3.77% | 3.75% |
| 150k | 1,000 | 128 | +3.42% | 2.72% |
| 150k | 2,000 | 256 | -19.80% | 19.86% |
| 150k | 500 | 256 | -19.95% | 19.95% |

`150k / 1,000 / 256` won on net profit (though not on drawdown — `100k / 1,000 / 256`
had the lowest, 2.29% — a real tradeoff, unlike an earlier version of this sweep that
had a data bug making one config dominate on both axes; see the note below). Its
held-out test-segment result: **+9.48% net profit, 5.64% max drawdown** — actually
*better* than its own validation result, a good sign against overfitting to the
validation window. That looked like a genuinely good, robust result — until it was
rerun across 5 random seeds:

| Seed | Validation Net % | Validation Max DD % | Test Net % | Test Max DD % |
|---|---|---|---|---|
| 42 | +8.18 | 3.33 | +9.48 | 5.64 |
| 44 | -6.74 | 14.03 | -4.18 | 12.44 |
| 46 | -10.55 | 11.33 | -10.60 | 13.68 |
| 45 | -15.69 | 15.72 | +8.99 | 7.42 |
| 43 | -19.94 | 19.94 | -0.16 | 0.16 |

**Mean ± std across seeds: validation -8.95% ± 9.66%, test +0.71% ± 7.72%.**

**Conclusion: this configuration — and, by extension, the current reward/observation
setup at this training scale — is not reliably profitable.** 4 of 5 seeds show a losing
validation result; only seed 42 (the one used for every other result in this project's
history) came out ahead. The test segment's mean is barely positive but its standard
deviation is an order of magnitude larger than the mean itself. Any single training
run's headline number is dominated by seed luck rather than a consistently learned
trading edge.

> **A genuine data bug was found and fixed during this investigation.**
> `xau.py`'s week-labeling used `%Y-W%V` (Gregorian year + ISO week number), which
> silently mislabels the last few days of December as week 1 of the *current* year when
> they're actually ISO week 1 of the *next* year — colliding with the real January week
> 1 under the identical label. Two week-labels in this dataset (`2024-W01`, `2025-W01`)
> each secretly spliced together dates roughly a year apart into what the walk-forward
> loop treated as one contiguous week, producing a fake single-step price jump (gold's
> real year-over-year move, ~$1,694 in one case) that dominated whichever seed's policy
> happened to be positioned on either side of it. Fixed by using `%G-W%V` instead
> (`%G` = ISO week-based year). The table and seed results above are from the corrected
> code; an earlier version of this investigation (before the fix) showed even larger,
> partly-artifactual variance across seeds and configs — this is the trustworthy version.

#### 1. Position sizing

The action-to-position mapping used to hard-threshold into exactly three states (full
short / flat / full long), discarding the policy's own conviction signal beyond that.
Scaling position size continuously with the action value beyond the ±0.3 deadzone (see
[Action Space](#action-space--box1-continuous--1-1)) reduced seed variance and improved
the validation mean over era 0: **-1.58% ± 7.11% validation, -0.13% ± 6.91% test** — a
real, standing improvement, kept as the baseline for every era after it.

#### 2. Richer feature set

Added ATR(14), RSI(14), a higher-timeframe (H1) ADX, and cyclical session/time-of-day
encoding — observation space grew from 9 to 14 dimensions, all sharing
`features.py::MARKET_FEATURE_COLUMNS` with the live bridge so training and live can't
drift apart the way plain ADX once did. Result: **+0.77% ± 6.09% validation, -3.65% ±
5.68% test** — the best validation mean and lowest validation std of any era, but a
worse test mean than sizing-only alone, a mild overfitting signal on 5 seeds. Kept as
the current baseline anyway, since era 0/1's numbers were categorically worse.

#### 3. Reward horizon — tried, not adopted

Widened the (then still Sortino-like) reward's downside-risk window from the original
50 steps, on top of eras 1+2:

- **200 steps:** worst validation mean of any era (-7.63% ± 2.45%) and the worst test
  std (9.31, -10.08% to +16.42% spread across seeds) — read as overfitting to
  validation-window-specific volatility.
- **100 steps:** better than 200 but still negative on both segments (-1.10% ± 5.34%
  validation, -2.10% ± 4.73% test), and still below era 2 on the validation metric that
  actually decides rankings.

Retired; `xau.py` reverted to the original 50-step window.

#### 4. Seed ensembling at inference

Rather than trusting one trained seed, `ensemble_eval.py` replays the walk-forward
backtest driving several already-trained seeds at once, combining each seed's
deterministic action per step by mean or median.

- On the retired 100-step era's checkpoints, **mean-combine beat the individual-seed
  mean on both segments** (+0.61% vs. -1.10% validation, -1.60% vs. -2.10% test) and cut
  max drawdown roughly 4x (~1.3–2.2% vs. ~5.2–5.4%) — an unambiguous win over the plain
  seed mean, though it didn't beat the single luckiest seed (expected — no one could have
  picked the lucky seed in advance).
- Re-tested on era 2 (the actual current-best baseline), the result was **mixed**:
  median-combine improved validation (+2.02% vs. +0.77%, lower drawdown too) but
  *worsened* test net profit (-5.21% vs. -3.65%), while still reducing test drawdown
  (7.26% vs. 8.45%). The drawdown-reduction property generalizes; the net-profit
  improvement doesn't, at least not yet.

#### 5. Accept the finding

Formally invoked as of 2026-08-20 — see [Final Status](#final-status-2026-08-20-not-reliably-profitable--investigation-concluded)
above.

#### 6. Reward signal audit — raw PnL reward, tried, not adopted

Real training logs showed weekly Reward and PnL frequently disagreeing in sign and
magnitude (e.g. a -$21.58 week logging Reward +136.83; a -$43.24 week — the batch's
biggest loss — logging Reward +1.27). Root cause: the reward's `downside_std`
normalization was recomputed every step from a sliding 50-step window, so identical
dollar moves got arbitrarily amplified or crushed by unrelated recent volatility, and a
`-0.5 * drawdown` penalty fired on every step spent underwater rather than just the step
that caused the drawdown. Fixed by making the reward plain raw step PnL (see
[Reward Function](#reward-function)) — verified to sum to exactly the episode's net PnL,
by construction, both in `smoke_test.py` and in real sweep logs.

Real-swept on top of eras 1+2: **-5.92% ± 3.82% validation, -6.69% ± 8.01% test**.
Validation std shrank ~37% versus era 2 (6.09 → 3.82) — the reward being internally
consistent with PnL does appear to reduce training-outcome noise across seeds — but the
mean shifted solidly negative rather than tightening around breakeven/profit, and test
got worse on both mean and std. **Not adopted as an improvement**, but also not reverted
to the old, confirmed-buggy formula — the current code keeps the simpler, bug-free raw-PnL
reward, since removing a real bug is worth keeping even though it didn't fix the
underlying profitability problem.

### Original Results (superseded)

The table below was this project's initial result, before the validation/test split,
multi-seed steps, and week-label fix above existed. It's kept for reference, but has
multiple known issues — see
[Hyperparameter Validation Methodology](#hyperparameter-validation-methodology) and the
data-bug note above.

| Pretrain Steps | Gradient Steps | Batch Size | End Balance | Net Profit | Max Drawdown |
|---|---|---|---|---|---|
| 150k | 500 | 256 | $10,036.18 | +0.36% | 12.17% |
| 150k | 1,000 | 256 | $12,063.55 | +20.64% | 10.60% |
| 150k | 2,000 | 256 | $5,299.15 | -47.01% | 49.82% |
| 150k | 1,000 | 128 | $4,706.40 | -52.94% | 55.49% |
| 100k | 1,000 | 256 | $9,450.09 | -5.50% | 17.36% |
| 50k | 1,000 | 256 | $12,465.32 | +24.65% | 11.05% |

#### Key Observations (from the original, single-seed, bug-contaminated sweep — superseded by the results above)
- The model appeared highly sensitive to gradient steps in this original sweep, and this one held up: in the corrected sweep, all 4 positive configs use exactly 1,000 gradient steps, while both 500 and 2,000 land at the bottom — 1,000 looks like a genuine sweet spot, not a single-seed fluke
- Batch size 128 looked catastrophic here (-52.94%), but the corrected sweep shows it competitively (+3.42%, 4th of 6) — this observation did *not* hold up, and was likely a week-label-bug artifact

---

## Hyperparameter Validation Methodology

The "Original Results" table above has a known methodology issue: all 6 configs were
trained and backtested on the *same* walk-forward window, and the best-looking one was
then reported as "the result." Once a config is chosen by comparing outcomes on a
window, that window stops being a clean estimate of unseen performance for the winner —
the selection process has already used it. DRL training is also seed-sensitive, and
every run in that table used a single fixed seed, so a result could be an outlier rather
than representative of that config's typical behavior.

The corrected process:

1. **Chronological split.** The data splits into three ordered blocks: pretraining
   (first 26 weeks), a validation window, and a held-out test window at the end.
   Walk-forward fine-tuning runs continuously through both — the split only changes
   when results get looked at, not how training happens.
2. **Rank on validation only.** Each candidate config is trained and walked forward
   through the full timeline, but configs are compared using only the validation
   segment's result.
3. **Test the winner exactly once.** Only the winning config's held-out test-segment
   result gets reported as the real out-of-sample number.
4. **Multi-seed variance.** The winning config is rerun across several seeds, and a
   mean ± standard deviation is reported instead of a single point estimate.

This is implemented by five scripts:
- `run_sweep.py` — trains and ranks several hyperparameter configs (steps 1–2 above),
  running them concurrently as separate processes. Each invocation writes to its own
  timestamped `sweep_runs/<run_id>/` directory so two runs can never collide.
- `run_seed_sweep.py` — reruns one chosen config across multiple seeds (step 4),
  reusing a result already produced by `run_sweep.py` where possible instead of
  re-running that seed. Writes to `sweep_runs_seeds/<run_id>/`.
- `sweep_common.py` — the shared process-parallel execution engine both drivers above
  use.
- `ensemble_eval.py` — replays the walk-forward backtest driving several already-trained
  seeds' checkpoints at once (no training), combining their per-step actions by mean or
  median — used for the seed-ensembling investigation (era 4 above).
- `smoke_test.py` — runs the same training pipeline against a small data slice with
  tiny step counts, to catch pipeline bugs in seconds instead of after a multi-hour
  real run.

Results from this corrected process are in
[Hyperparameter Experiment Results](#hyperparameter-experiment-results) above.

---

## Project Structure

```
drl-xauusd/
├── xau.py                   # Training pipeline (environment, WFO loop, backtest)
├── xau_macro.py             # Experimental macro state-space integration (see experiment above)
├── mt5bridge.py             # Live execution bridge with macro circuit breaker
├── features.py               # Shared M5-to-M15 feature pipeline used by both xau.py and mt5bridge.py
├── converter.py             # Utility to convert raw CSV timestamps to UTC format
├── merge_news.py            # Merges historical Hugging Face economic event data into OHLCV
├── smoke_test.py             # Fast pipeline sanity check on a small data slice
├── run_sweep.py               # Hyperparameter sweep driver (see Validation Methodology)
├── run_seed_sweep.py          # Seed-variance driver (see Validation Methodology)
├── sweep_common.py            # Shared process-parallel execution engine for the two drivers above
├── ensemble_eval.py           # Inference-time seed-ensembling evaluator (no training)
├── data/
│   ├── data.csv             # Raw M5 OHLCV (not included)
│   └── data_with_news.csv   # Merged dataset with news flags (not included)
├── models/
│   └── *.zip                # Saved weekly model checkpoints (not included)
├── sweep_runs/                # Phase 1 sweep output, gitignored (not included)
├── sweep_runs_seeds/           # Phase 2 seed-variance output, gitignored (not included)
├── archive/                    # Superseded checkpoints/results from earlier eras, gitignored (not included)
└── README.md
```

---

## Setup & Usage

### Requirements

```bash
pip install stable-baselines3 gymnasium pandas pandas-ta-classic MetaTrader5 torch requests tqdm datasets tensorboard
```

### 1. Prepare Data

Ensure your base CSV has `timestamp, open, high, low, close, volume`. Then run:

```bash
python converter.py your_data.csv
```

To reproduce the macro experiment, also run:

```bash
python merge_news.py
```

### 2. Train the Agent

```bash
python xau.py
```

This will:
- Pretrain on the first 26 weeks
- Walk-forward fine-tune week by week
- Print the out-of-sample backtest summary
- Save model checkpoints to `./models/`

### 3. Run Live (MT5 Bridge)

Configure the settings at the top of `mt5bridge.py`:
```python
SYMBOL      = "XAUUSDm"         # Your broker's gold symbol
MODEL_PATH  = "./models/..."    # Path to your chosen weekly checkpoint
DISCORD_WEBHOOK_URL = ""        # Optional: Discord alert webhook
DISCORD_USER_ID     = ""        # Optional: Discord user ID for pings
```

Then run:
```bash
python mt5bridge.py
```

The bridge waits for each M15 candle close, builds the observation vector from live MT5 data, queries the macro regime, and either executes the agent's decision or engages the circuit breaker.

---

## Tech Stack

| Component | Library |
|---|---|
| DRL Algorithm | `stable-baselines3` (SAC) |
| Environment | `gymnasium` |
| Technical Indicators | `pandas-ta-classic` |
| Live Execution | `MetaTrader5` |
| Deep Learning Backend | `PyTorch` |
| Macro Risk Filter | Gemini API (via macro-analyzer repo) |

---

## Limitations & Future Work

- **The agent is not reliably profitable, and this is now the project's accepted,
  documented conclusion rather than an open problem still being chased.** Six
  methodologically distinct interventions — hyperparameter tuning, continuous position
  sizing, a richer feature set, reward-horizon widening, seed ensembling, and a
  reward-signal-fidelity audit — were each tried in isolation and evaluated with a
  seed-variance-aware validation/test protocol (see
  [Hyperparameter Experiment Results](#hyperparameter-experiment-results)). None closed
  the gap: net profit's standard deviation swamps its mean in every era tried, and no
  configuration comes close to simple buy-and-hold of the underlying asset. Two untried
  reward-shaping ideas remain candidates for a genuinely new (not yet validated)
  direction, should this be picked back up: a differential Sharpe/Sortino ratio (an
  online-updated formulation designed to avoid the sliding-window instability the reward
  audit found and fixed), or the current raw-PnL reward plus a terminal-only
  risk-adjustment bonus computed once over the whole episode rather than recomputed every
  step
- **The live bridge doesn't mirror the simulation's continuous position sizing.**
  `mt5bridge.py` still thresholds the action into a binary long/short/flat decision and
  always trades at a fixed 0.01 lots (`FIXED_LOTS`), while `xau.py`'s simulation has since
  moved to continuous, conviction-scaled position sizing (see
  [Action Space](#action-space--box1-continuous--1-1)). Given the "not reliably
  profitable" finding above, closing this gap isn't urgent, but it's a real
  train/live mismatch worth flagging
- No per-trade risk control — the only protection is the episode-level 20% drawdown kill
  switch; there is no per-trade stop-loss/take-profit, and `mt5bridge.py` places no
  broker-side stop either, so a crashed bot process leaves any open position completely
  unprotected
- Generalization across different market regimes (trending vs. ranging) remains an open problem
- The macro circuit breaker suppresses all new trades during high impact windows — a more refined approach could reduce position size rather than stop entirely