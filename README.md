# DRL XAUUSD Trading Bot

A Deep Reinforcement Learning trading agent for Gold (XAUUSD) on the M15 timeframe, built using **Soft Actor-Critic (SAC)** and trained with a custom **Walk-Forward Optimization (WFO)** pipeline. Includes a live execution bridge via MetaTrader 5.

> **Disclaimer:** This project is for educational and portfolio purposes only. It is not financial advice and should not be used with real capital without thorough due diligence.

---

## Overview
Most DRL trading projects train once on a fixed historical window and call it done. This project takes a different approach — it uses a rolling walk-forward training loop that continuously fine-tunes the agent week-by-week on fresh out-of-sample data, mimicking how a real adaptive system would behave in a live market.

The agent trades Gold (XAUUSD) on the M15 timeframe using Soft Actor-Critic (SAC) with a continuous action space, operating on a compact 9-dimensional observation vector that combines normalized market features with live account state — unrealized PnL and current drawdown. A custom WeeklyRollingBuffer ensures the agent only trains on recent market regimes by purging the oldest week's experience at each rollover, preventing it from overfitting to stale conditions.

Reward shaping uses a Sortino-like ratio with an explicit drawdown penalty, and a 20% drawdown hard stop terminates episodes early — building capital preservation directly into the training objective rather than treating it as an afterthought.
Hyperparameter experiments revealed the model is highly sensitive to the number of gradient steps per fine-tuning cycle. Too few and the agent underfits; too many and it overfits aggressively to the current week and collapses on the next. The best configuration achieved +24.65% net profit over the out-of-sample walk-forward period with an 11% max drawdown — though generalizing consistently across different market regimes remains an open problem.
---

## Architecture

### Observation Space — `Box(9,)` 
| Index | Feature | Description |
|-------|---------|-------------|
| 0–5 | Market Features | `open, high, low, close, volume, ADX(14)` — normalized via rolling 200-candle min-max to `[-1, 1]` |
| 6 | Position | `1.0` = Long, `-1.0` = Short, `0.0` = Flat |
| 7 | Unrealized PnL | Scaled and clamped to `[-1, 1]` |
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
  • Fine-tune with N gradient steps
  • Save model checkpoint per week
     │
     ▼
[Out-of-Sample Backtest Summary]
```

### Custom `WeeklyRollingBuffer`
A key component of this pipeline is a custom replay buffer that tags every experience with an **ISO week label**. At each rollover, the oldest week's data is purged — ensuring the agent only learns from recent, relevant market regimes. This prevents the agent from overfitting to stale market conditions.

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
├── xau.py              # Training pipeline (environment, WFO loop, backtest)
├── mt5bridge.py        # Live execution bridge via MetaTrader 5
├── converter.py        # Utility to convert raw CSV timestamps to UTC format
├── data/
│   └── data.csv        # Your M5 OHLCV CSV (not included)
├── models/
│   └── *.zip           # Saved weekly model checkpoints (not included)
└── README.md
```

---

## Setup & Usage

### Requirements
```bash
pip install stable-baselines3 gymnasium pandas pandas-ta-classic MetaTrader5 torch requests tqdm
```

### 1. Prepare Data
Your CSV should have the columns: `timestamp, open, high, low, close, volume`

If timestamps are missing seconds/timezone:
```bash
python converter.py your_data.csv
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
SYMBOL = "XAUUSDm"            # Your broker's symbol
MODEL_PATH = "./models/..."   # Path to your chosen checkpoint
DISCORD_WEBHOOK_URL = ""      # Optional: Discord alert webhook
DISCORD_USER_ID = ""          # Optional: Your Discord user ID for pings
```

Then run:
```bash
python mt5bridge.py
```
The bridge syncs to M15 candle closes and executes trades based on the agent's decisions.

---

## Tech Stack

| Component | Library |
|---|---|
| DRL Algorithm | `stable-baselines3` (SAC) |
| Environment | `gymnasium` |
| Technical Indicators | `pandas-ta-classic` |
| Live Execution | `MetaTrader5` |
| Deep Learning Backend | `PyTorch` |

---

## Limitations & Future Work

- The agent uses only 6 market features — adding more indicators (RSI, Bollinger Bands, session filters) may improve signal quality
- No position sizing — fixed 0.01 lots; Kelly Criterion or volatility-based sizing could improve risk-adjusted returns
- The reward function is simple; exploring reward shaping with Sharpe ratio or risk-adjusted PnL over longer windows is worth investigating
- Generalization across different market regimes (trending vs. ranging) remains an open problem
