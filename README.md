# DRL XAUUSD Trading Bot

A Deep Reinforcement Learning trading agent for Gold (XAUUSD) on the M15 timeframe, built using **Soft Actor-Critic (SAC)** and trained with a custom **Walk-Forward Optimization (WFO)** pipeline. Includes a live execution bridge via MetaTrader 5 and a synchronized Macroeconomic Risk Circuit Breaker.

> **Disclaimer:** This project is for educational and portfolio purposes only. It is not financial advice and should not be used with real capital without thorough due diligence.

---

## Overview

Most DRL trading projects train once on a fixed historical window and call it done. This project takes a different approach — it uses a rolling walk-forward training loop that continuously fine-tunes the agent week-by-week on fresh out-of-sample data, mimicking how a real adaptive system would behave in a live market.

The agent trades Gold (XAUUSD) on the M15 timeframe using Soft Actor-Critic (SAC) with a continuous action space, operating on a compact 9-dimensional observation vector that combines normalized market features with live account state — unrealized PnL and current drawdown. A custom `WeeklyRollingBuffer` ensures the agent only trains on recent market regimes by purging the oldest week's experience at each rollover, preventing it from overfitting to stale conditions.

Reward shaping uses a Sortino-like ratio with an explicit drawdown penalty, and a 20% drawdown hard stop terminates episodes early — building capital preservation directly into the training objective rather than treating it as an afterthought.

Hyperparameter experiments revealed the model is highly sensitive to the number of gradient steps per fine-tuning cycle. Too few and the agent underfits; too many and it overfits aggressively to the current week and collapses on the next. The best configuration achieved +24.65% net profit over the out-of-sample walk-forward period with an 11% max drawdown — though generalizing consistently across different market regimes remains an open problem.

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

**Conclusion:**

Integrating macro data directly into the RL state space severely degraded performance. The agent developed what could be described as "scared agent syndrome" — closing highly profitable, long-running trend trades prematurely simply because a news event appeared on the horizon. This caused significant spread bleed from paying double broker fees to re-enter interrupted trends, and distorted the reward signal.

DRL agents excel at continuous, flowing data like technical price action but struggle with sparse, binary anomalies like news drops. The signal is too rare and too irregular for the agent to learn a clean policy around it.

**The conclusion is separation of concerns.** The SAC model is left completely blind to macro events, allowing it to optimize purely for technical price action. Risk management is decoupled and handled deterministically by `mt5bridge.py`, which halts trading via API polling during high volatility windows. The experimental training script is preserved in `xau_macro.py` for reference.

---

## Architecture

### Observation Space — `Box(9,)`

| Index | Feature | Description |
|-------|---------|-------------|
| 0–5 | Market Features | `open, high, low, close, volume, ADX(14)` — normalized via rolling 200-candle min-max to `[-1, 1]` |
| 6 | Position | `1.0` = Long, `-1.0` = Short, `0.0` = Flat |
| 7 | Unrealized PnL | Scaled by ÷100 and clamped to `[-1, 1]` |
| 8 | Drawdown % | Current drawdown from peak balance, clamped to `[-1, 1]` |

### Action Space — `Box(1,)` continuous `[-1, 1]`

The continuous action is mapped to discrete positions:
- `< -0.3` → **Short**
- `> 0.3` → **Long**
- `[-0.3, 0.3]` → **Flat / Hold**

### Reward Function

The reward is a modified **Sortino-like ratio** computed at each step:

```
reward = step_pnl / (downside_std + ε)
```

With an additional **drawdown penalty** if drawdown exceeds 5%:
```
reward -= 0.5 * drawdown  (if drawdown > 5%)
```

A **20% drawdown hard stop** terminates the episode early.

---

## Walk-Forward Optimization Pipeline

```
[M5 Raw CSV]
     │
     ▼
[Data Engineering]
  • Compute ADX(14) on M5
  • Resample M5 → M15
  • Tag ISO week labels
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

All results are **out-of-sample** walk-forward backtests on XAUUSD M15 starting from $10,000.

| Pretrain Steps | Gradient Steps | Batch Size | End Balance | Net Profit | Max Drawdown |
|---|---|---|---|---|---|
| 150k | 500 | 256 | $10,036.18 | +0.36% | 12.17% |
| 150k | 1,000 | 256 | $12,063.55 | +20.64% | 10.60% |
| 150k | 2,000 | 256 | $5,299.15 | -47.01% | 49.82% |
| 150k | 1,000 | 128 | $4,706.40 | -52.94% | 55.49% |
| 100k | 1,000 | 256 | $9,450.09 | -5.50% | 17.36% |
| **50k** | **1,000** | **256** | **$12,465.32** | **+24.65%** | **11.05%** |

### Key Observations
- The model is **highly sensitive to gradient steps** — too few underfit, too many overfit aggressively to the current week and collapse on the next
- **Batch size matters significantly** — 128 produced severe overfitting compared to 256
- **More pretraining isn't always better** — 50k pretrain with 1k gradient steps outperformed 150k pretrain with the same fine-tuning config, suggesting that a lighter pretrain leaves the policy more adaptable during the walk-forward phase
- The best config (+24.65%) is still not consistently profitable across all market regimes, indicating the agent hasn't solved the generalization problem — a known hard challenge in financial DRL

---

## Project Structure

```
drl-xauusd/
├── xau.py                   # Training pipeline (environment, WFO loop, backtest)
├── xau_macro.py             # Experimental macro state-space integration (see experiment above)
├── mt5bridge.py             # Live execution bridge with macro circuit breaker
├── converter.py             # Utility to convert raw CSV timestamps to UTC format
├── merge_news.py            # Merges historical Hugging Face economic event data into OHLCV
├── data/
│   ├── data.csv             # Raw M5 OHLCV (not included)
│   └── data_with_news.csv   # Merged dataset with news flags (not included)
├── models/
│   └── *.zip                # Saved weekly model checkpoints (not included)
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

- The agent uses only 6 market features — adding more indicators (RSI, Bollinger Bands, session filters) may improve signal quality
- No position sizing — fixed 0.01 lots; Kelly Criterion or volatility-based sizing could improve risk-adjusted returns
- The reward function is simple; exploring reward shaping with Sharpe ratio or risk-adjusted PnL over longer windows is worth investigating
- Generalization across different market regimes (trending vs. ranging) remains an open problem
- The macro circuit breaker suppresses all new trades during high impact windows — a more refined approach could reduce position size rather than stop entirely