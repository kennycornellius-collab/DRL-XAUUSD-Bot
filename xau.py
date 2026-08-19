import os
import time
import json
import threading
import collections
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import gymnasium as gym
from gymnasium import spaces

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from features import compute_m15_features, MARKET_FEATURE_COLUMNS


def prepare_data(csv_path: str) -> pd.DataFrame:
    print(f"Loading data from {csv_path}...")
    df_m5 = pd.read_csv(csv_path)
    df_m5['timestamp'] = pd.to_datetime(df_m5['timestamp'], utc=True)

    df_m15 = compute_m15_features(df_m5)
    # %G (ISO week-based year), not %Y (Gregorian year) - %Y-W%V mislabels the last
    # few days of December as week 1 of the *current* year when they're actually ISO
    # week 1 of the *next* year, colliding with the real week 1 in January and silently
    # splicing two dates ~1 year apart into a single "week".
    df_m15['week_label'] = df_m15['timestamp'].dt.strftime("%G-W%V")

    print(f"Data prepped. M15 Shape: {df_m15.shape}")
    return df_m15

class XAUEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0, initial_peak_balance: Optional[float] = None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.max_steps = len(self.df) - 1
        self.initial_balance = initial_balance
        self.initial_peak_balance = initial_peak_balance if initial_peak_balance is not None else initial_balance

        # len(MARKET_FEATURE_COLUMNS) window-normalized market features (see _get_obs)
        # + 5 state features appended directly: position, unrealized PnL, drawdown,
        # session_sin, session_cos (the latter two are already bounded in [-1, 1] by
        # construction, so they skip the rolling-window normalization the market
        # features go through).
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(len(MARKET_FEATURE_COLUMNS) + 5,), dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.start_index = 0

        # Pre-extracted numpy arrays instead of repeated df.iloc[...] lookups per step -
        # .iloc row/cell access re-does pandas type-checking/index-alignment on every call,
        # which dominates per-step cost at this env's scale (measured ~590x slower than
        # numpy indexing for the same lookups). _close deliberately keeps its native
        # float64 dtype (no cast) since it feeds PnL/balance/reward arithmetic directly -
        # casting it would silently change every dollar figure the pipeline produces.
        self._close = self.df['close'].to_numpy()
        self._week_label = self.df['week_label'].to_numpy()
        self._market_features = self.df[MARKET_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        self._session_sin = self.df['session_sin'].to_numpy()
        self._session_cos = self.df['session_cos'].to_numpy()

        # Precomputed rolling min/max for _get_obs's window normalization, replacing a
        # collections.deque that used to get rebuilt into a fresh array every single call.
        # Every episode always starts at self.start_index == 0 and steps forward
        # monotonically through this same df, so "the window visible at row k" is a pure
        # function of row position - identical across every episode/reset this instance
        # ever runs, so it's safe (and much cheaper) to compute once here rather than
        # maintain live per-episode state. Cast to float32 BEFORE rolling, matching
        # _append_history's cast-then-reduce order so the result is bitwise identical to
        # the old deque-of-float32-rows approach.
        feat_df = self.df[MARKET_FEATURE_COLUMNS].astype(np.float32)
        roll = feat_df.rolling(window=200, min_periods=1)
        self._roll_min = roll.min().to_numpy(dtype=np.float32)
        self._roll_max = roll.max().to_numpy(dtype=np.float32)
        self._hist_step = 0  # index of the last row _append_history() actually wrote

    def current_week_label(self) -> str:
        if self.current_step >= len(self._week_label):
            return self._week_label[-1]
        return self._week_label[self.current_step]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.start_index
        
        self.position = 0.0
        self.entry_price = 0.0
        self.balance = self.initial_balance
        self.peak_balance = self.initial_peak_balance
        self.current_unrealized_pnl = 0.0
        self.current_dd = 0.0

        self._append_history()
        return self._get_obs(), {"week_label": self.current_week_label()}

    def _append_history(self):
        self._hist_step = self.current_step

    def step(self, action: np.ndarray):
        act_val = float(np.clip(action[0], -1.0, 1.0))
        position_before = self.position
        dir_before = np.sign(position_before)

        # Beyond the +-0.3 deadzone, position size tracks the raw action value instead of
        # snapping to +-1 - a barely-confident 0.31 and a maximally-confident 0.99 now get
        # proportionally different exposure. At |action|==1.0 this reproduces the old
        # always-full-size behavior exactly.
        if act_val < -0.3 or act_val > 0.3:
            new_pos = act_val
        else:
            new_pos = 0.0
        dir_after = np.sign(new_pos)

        # Cost is only charged on the same "events" as before (opening from flat, or
        # reversing direction) - resizing within the same direction and closing to flat
        # both stay free, matching the original model. Now proportional to the size of the
        # position being entered rather than a flat fee, so it still charges exactly $0.30
        # at full size but less for a partial-conviction entry.
        spread_cost = 0.0
        if dir_after != 0 and dir_after != dir_before:
            spread_cost = 0.30 * abs(new_pos)

        current_close = self._close[self.current_step]

        self.current_step += 1
        done = False
        truncated = False

        if self.current_step >= self.max_steps:
            done = True
            next_close = current_close
        else:
            next_close = self._close[self.current_step]

        price_pnl = position_before * (next_close - current_close)
        step_return = price_pnl - spread_cost
        
        self.balance += step_return

        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        self.current_dd = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0.0

        # Raw dollar step_return - no rolling downside_std normalization and no per-step
        # drawdown penalty. Both were found to decouple a week's summed reward from its
        # actual net PnL (sometimes flipping the sign entirely): downside_std was
        # recomputed every step from a sliding 50-step window, so identical $ moves got
        # arbitrarily amplified or crushed depending on unrelated recent volatility, and
        # the old `-0.5 * current_dd` term fired on every step spent underwater
        # regardless of that step's own outcome (e.g. an intra-week dip-then-recovery
        # netting a small loss could still rack up a huge cumulative penalty). Reward now
        # sums to exactly the episode's net PnL by construction, so the training signal
        # can't disagree with the metric actually being optimized for.
        reward = step_return

        self.position = new_pos

        if dir_after != 0:
            # Reset the entry reference on a fresh entry *or* a direct reversal (dir
            # changes without passing through exactly 0) - previously only the "was 0.0"
            # check ran, so a reversal kept the old (now-closed) trade's entry price and
            # computed unrealized PnL against the wrong reference.
            if dir_after != dir_before:
                self.entry_price = current_close
            self.current_unrealized_pnl = self.position * (next_close - self.entry_price)
        else:
            self.entry_price = 0.0
            self.current_unrealized_pnl = 0.0

        if self.current_dd > 0.20:
            done = True

        if not done:
            self._append_history()

        info = {
            "week_label": self.current_week_label() if not done else self._week_label[-1],
            "step_pnl": step_return,
            "balance": self.balance,
            "peak_balance": self.peak_balance,
            # Exposed for trade-level tearsheet stats (win rate, profit factor, duration)
            # built in run_wfo_pipeline - position_before/price_pnl attribute this bar's
            # price move to the trade that was actually open during it, while
            # position_after/spread_cost attribute the entry fee (if any) to the new trade.
            "position_before": position_before,
            "position_after": new_pos,
            "price_pnl": price_pnl,
            "spread_cost": spread_cost,
        }
        return self._get_obs(), float(reward), done, truncated, info

    def _get_obs(self) -> np.ndarray:
        # mins/maxs/current_features all come from _hist_step (the last row
        # _append_history() actually wrote), NOT self.current_step directly - on a
        # terminating step the append is skipped (see step()'s `if not done:` guard), so
        # self.current_step can point one row past what's actually been observed.
        mins = self._roll_min[self._hist_step]
        maxs = self._roll_max[self._hist_step]
        ranges = maxs - mins
        ranges[ranges == 0] = 1e-8

        current_features = self._market_features[self._hist_step]
        scaled = (current_features - mins) / ranges
        scaled = (scaled * 2.0) - 1.0
        scaled = np.clip(scaled, -1.0, 1.0)

        pos = float(self.position)
        ur_pnl = np.clip(self.current_unrealized_pnl / 100.0, -1.0, 1.0)
        dd = np.clip(self.current_dd, -1.0, 1.0)

        # session_sin/session_cos are computed straight from the row's own timestamp in
        # features.py and already lie in [-1, 1] - read directly, no rolling-window
        # normalization needed (or wanted - it would distort their cyclical meaning).
        # Deliberately keyed on self.current_step (not _hist_step) - matches the original
        # code's existing behavior, including its pre-existing terminal-step
        # inconsistency with the market features above.
        row_idx = min(self.current_step, len(self._session_sin) - 1)
        session_sin = float(self._session_sin[row_idx])
        session_cos = float(self._session_cos[row_idx])

        return np.concatenate([scaled, [pos, ur_pnl, dd, session_sin, session_cos]]).astype(np.float32)


class WeeklyRollingBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid_mask = np.zeros(self.buffer_size, dtype=bool)
        self.week_index_map = collections.defaultdict(list)
        # Reverse lookup (slot -> owning week), built after super().__init__() since SB3
        # may rewrite self.buffer_size (e.g. // n_envs) - sized to match valid_mask.
        # Lets add() find which week owns a slot about to be overwritten in O(1) instead
        # of scanning every tracked week's index list.
        self.slot_week: List[Optional[str]] = [None] * self.buffer_size

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray,
            reward: np.ndarray, done: np.ndarray, infos: List[Dict[str, Any]]) -> None:

        week_label = infos[0]['week_label']
        idx = self.pos

        old_week = self.slot_week[idx]
        if old_week is not None:
            # .get(), never bracket access - old_week may have already been fully purged
            # by purge_oldest_week(), and bracket access on this defaultdict would
            # silently resurrect it as an empty list (then .remove() would raise, since
            # idx was never actually in it).
            inds = self.week_index_map.get(old_week)
            if inds is not None and idx in inds:
                inds.remove(idx)
                if not inds:
                    del self.week_index_map[old_week]

        self.week_index_map[week_label].append(idx)
        self.slot_week[idx] = week_label
        self.valid_mask[idx] = True

        super().add(obs, next_obs, action, reward, done, infos)

    def purge_oldest_week(self):
        if not self.week_index_map:
            return

        oldest_week = sorted(self.week_index_map.keys())[0]
        inds_to_purge = self.week_index_map.pop(oldest_week)

        for idx in inds_to_purge:
            self.rewards[idx] = 0.0
            self.dones[idx] = 0.0
            self.timeouts[idx] = 0.0
            self.valid_mask[idx] = False
            self.slot_week[idx] = None  # keep the reverse lookup consistent after a purge

    def sample(self, batch_size: int, env: Optional[DummyVecEnv] = None):
        valid_indices = np.where(self.valid_mask)[0]

        if len(valid_indices) == 0:
            valid_indices = np.arange(self.buffer_size)

        replace = len(valid_indices) < batch_size
        batch_inds = np.random.choice(valid_indices, size=batch_size, replace=replace)
        
        return self._get_samples(batch_inds, env=env)


class ThreadAllocPoller:
    """Background daemon thread that watches a shared JSON status file (written by the
    sweep driver's parent process) and re-calls torch.set_num_threads() whenever the
    recommended thread count for this worker changes - lets a straggler config absorb
    logical cores freed up by sibling configs finishing early, instead of being stuck
    with whatever thread count was decided when the sweep was first launched. Runs
    independently of the main thread's own work (pretraining vs. the walk-forward loop),
    so one mechanism covers both phases without needing separate polling hooks in each.
    A no-op entirely if thread_alloc_path is None (the single, non-sweep run case)."""

    def __init__(self, thread_alloc_path: Optional[str], poll_interval: float = 3.0):
        self.thread_alloc_path = thread_alloc_path
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.thread_alloc_path:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                with open(self.thread_alloc_path, "r", encoding="utf-8") as f:
                    alloc = json.load(f)
                target = int(alloc["threads_per_worker"])
                if target != torch.get_num_threads():
                    torch.set_num_threads(target)
                    tqdm.write(f"[ThreadAlloc] Adjusted to {target} torch thread(s) "
                               f"({alloc.get('active', '?')} worker(s) still active).")
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pass  # file not written yet, or caught mid-write - just retry next cycle
            self._stop_event.wait(self.poll_interval)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_interval + 1)


class WeekRolloverCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.current_week = None

    def _on_step(self) -> bool:

        week_labels = self.training_env.env_method("current_week_label")
        week_label = week_labels[0]

        if self.current_week is None:
            self.current_week = week_label
        elif self.current_week != week_label:
            self.model.replay_buffer.purge_oldest_week()
            valid_count = self.model.replay_buffer.valid_mask.sum()
            tqdm.write(f"[Callback] Rolled over to {week_label}. Purged oldest week. Valid Buffer Size: {valid_count}")
            self.current_week = week_label
        return True


def compute_backtest_metrics(equity_curve: List[float]) -> Dict[str, float]:
    eq_arr = np.array(equity_curve)
    start_bal = eq_arr[0]
    end_bal = eq_arr[-1]
    net_profit = end_bal - start_bal
    ret_pct = (net_profit / start_bal) * 100 if start_bal else 0.0

    peaks = np.maximum.accumulate(eq_arr)
    drawdowns = (peaks - eq_arr) / peaks
    max_dd = np.max(drawdowns) * 100 if len(eq_arr) > 1 else 0.0

    return {
        "start_balance": float(start_bal),
        "end_balance": float(end_bal),
        "net_profit": float(net_profit),
        "net_profit_pct": float(ret_pct),
        "max_drawdown_pct": float(max_dd),
    }


def compute_trade_stats(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    """Trades are built in run_wfo_pipeline from consecutive same-direction position
    bars (see XAUEnv.step's position_before/position_after/price_pnl/spread_cost info
    fields) - each dict here is {"week", "direction", "pnl", "duration"}."""
    if not trades:
        return {
            "trade_count": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "avg_trade_duration_bars": 0.0,
            "avg_trade_duration_hours": 0.0,
        }

    pnls = np.array([t["pnl"] for t in trades])
    durations = np.array([t["duration"] for t in trades])

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else 0.0

    return {
        "trade_count": int(len(pnls)),
        "win_rate_pct": float((pnls > 0).mean() * 100),
        "profit_factor": float(profit_factor),
        "avg_trade_duration_bars": float(durations.mean()),
        "avg_trade_duration_hours": float(durations.mean() * 0.25),  # M15 bars -> hours
    }


def compute_sharpe_sortino(weekly_returns: List[float], periods_per_year: int = 52) -> Dict[str, float]:
    """weekly_returns are fractional (ep_pnl / start-of-week balance), one per processed
    walk-forward week - annualized with sqrt(52) since that's the native resolution the
    walk-forward loop and replay buffer already operate at."""
    if len(weekly_returns) < 2:
        return {"sharpe": 0.0, "sortino": 0.0}

    returns = np.array(weekly_returns)
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = float(mean_ret / std_ret * np.sqrt(periods_per_year)) if std_ret > 0 else 0.0

    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) >= 2 else 0.0
    sortino = float(mean_ret / downside_std * np.sqrt(periods_per_year)) if downside_std > 0 else 0.0

    return {"sharpe": sharpe, "sortino": sortino}


def compute_weekly_pnl_distribution(weekly_pnl: List[float]) -> Dict[str, float]:
    if not weekly_pnl:
        return {
            "weekly_pnl_mean": 0.0,
            "weekly_pnl_std": 0.0,
            "pct_weeks_profitable": 0.0,
            "best_week_pnl": 0.0,
            "worst_week_pnl": 0.0,
        }

    pnl_arr = np.array(weekly_pnl)
    return {
        "weekly_pnl_mean": float(pnl_arr.mean()),
        "weekly_pnl_std": float(pnl_arr.std()),
        "pct_weeks_profitable": float((pnl_arr > 0).mean() * 100),
        "best_week_pnl": float(pnl_arr.max()),
        "worst_week_pnl": float(pnl_arr.min()),
    }


def compute_buy_hold_metrics(start_price: float, end_price: float, strategy_net_profit_pct: float) -> Dict[str, float]:
    """The bar the strategy actually needs to clear: simply buying and holding the same
    instrument over the same window, with zero trades and zero effort."""
    buy_hold_return_pct = (end_price - start_price) / start_price * 100 if start_price else 0.0
    return {
        "buy_hold_return_pct": float(buy_hold_return_pct),
        "vs_buy_hold_pct": float(strategy_net_profit_pct - buy_hold_return_pct),
    }


def build_segment_metrics(
    equity_curve: List[float],
    weekly_returns: List[float],
    weekly_pnl: List[float],
    trades: List[Dict[str, Any]],
    start_price: Optional[float] = None,
    end_price: Optional[float] = None,
) -> Dict[str, float]:
    """Combines the equity-curve-level metrics (net profit, max DD) with trade-level
    stats, risk-adjusted return, weekly P&L distribution, and (when start_price/end_price
    are given) the buy-and-hold comparison into one flat dict, so callers still get the
    same start_balance/end_balance/... keys smoke_test.py already asserts on, plus the
    new tearsheet fields alongside them."""
    metrics = compute_backtest_metrics(equity_curve)
    metrics.update(compute_trade_stats(trades))
    metrics.update(compute_sharpe_sortino(weekly_returns))
    metrics.update(compute_weekly_pnl_distribution(weekly_pnl))
    if start_price is not None and end_price is not None:
        metrics.update(compute_buy_hold_metrics(start_price, end_price, metrics["net_profit_pct"]))
    return metrics


def print_backtest_summary(title: str, metrics: Dict[str, float]) -> None:
    print("\n" + "="*50)
    print(title)
    print("="*50)
    print(f"Starting Balance:  ${metrics['start_balance']:,.2f}")
    print(f"Ending Balance:    ${metrics['end_balance']:,.2f}")
    print(f"Net Profit:        ${metrics['net_profit']:,.2f} ({metrics['net_profit_pct']:+.2f}%)")
    print(f"Max Drawdown:      {metrics['max_drawdown_pct']:.2f}%")
    if "sharpe" in metrics:
        print(f"Sharpe (weekly):   {metrics['sharpe']:.2f}")
        print(f"Sortino (weekly):  {metrics['sortino']:.2f}")
    if "trade_count" in metrics:
        pf = metrics['profit_factor']
        pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
        print(f"Trades:            {metrics['trade_count']} | Win Rate: {metrics['win_rate_pct']:.1f}% | Profit Factor: {pf_str}")
        print(f"Avg Trade Duration: {metrics['avg_trade_duration_bars']:.1f} bars (~{metrics['avg_trade_duration_hours']:.1f}h)")
    if "weekly_pnl_mean" in metrics:
        print(f"Weekly PnL:        mean ${metrics['weekly_pnl_mean']:,.2f} | std ${metrics['weekly_pnl_std']:,.2f} | "
              f"{metrics['pct_weeks_profitable']:.1f}% weeks profitable")
        print(f"Best/Worst Week:   ${metrics['best_week_pnl']:,.2f} / ${metrics['worst_week_pnl']:,.2f}")
    if "buy_hold_return_pct" in metrics:
        print(f"Buy & Hold:        {metrics['buy_hold_return_pct']:+.2f}% (strategy vs. buy & hold: {metrics['vs_buy_hold_pct']:+.2f} pts)")
    print("="*50 + "\n")


def rollout_week(vec_env, predict_fn, week_label: str, on_transition=None) -> Dict[str, Any]:
    """One deterministic pass over a single week's episode, stepping `vec_env` end-to-end
    via predict_fn(obs) -> action. Groups steps into trades by position *direction*
    changes (not exact position equality) - continuous sizing changes the raw position
    almost every step even while direction is held, so grouping by sign is what keeps "a
    trade" meaning a directional hold instead of fragmenting into one degenerate trade per
    step. Mirrors the dir_before/dir_after boundary XAUEnv.step() itself uses for entry
    costs. Shared by run_wfo_pipeline's training loop (which also feeds transitions to its
    replay buffer via on_transition) and any inference-only replay - e.g. ensemble_eval.py
    - that doesn't train, so the two paths can't silently drift apart."""
    obs = vec_env.reset()
    done = False
    ep_reward = 0.0
    ep_pnl = 0.0
    trades: List[Dict[str, Any]] = []
    current_trade: Optional[Dict[str, Any]] = None
    last_info: Dict[str, Any] = {}

    while not done:
        action = predict_fn(obs)
        next_obs, rewards, dones, infos = vec_env.step(action)
        if on_transition is not None:
            on_transition(obs, next_obs, action, rewards, dones, infos)

        ep_reward += rewards[0]
        ep_pnl += infos[0]['step_pnl']

        pos_before = infos[0]['position_before']
        pos_after = infos[0]['position_after']
        dir_before = np.sign(pos_before)
        dir_after = np.sign(pos_after)
        if dir_before != 0 and current_trade is not None:
            current_trade['pnl'] += infos[0]['price_pnl']
            current_trade['duration'] += 1
        if dir_after != dir_before:
            if current_trade is not None:
                trades.append(current_trade)
                current_trade = None
            if dir_after != 0:
                current_trade = {"week": week_label, "direction": int(dir_after), "pnl": -infos[0]['spread_cost'], "duration": 0}

        obs = next_obs
        done = dones[0]
        last_info = infos[0]

    if current_trade is not None:
        trades.append(current_trade)

    return {"ep_reward": ep_reward, "ep_pnl": ep_pnl, "trades": trades, "last_info": last_info}


def run_wfo_pipeline(
    csv_path: str,
    pretrain_weeks_count: int = 26,
    val_weeks_count: int = 79,
    pretrain_timesteps: int = 50_000,
    gradient_steps: int = 1000,
    batch_size: int = 256,
    buffer_size: int = 1920,
    replay_window_weeks: int = 4,
    min_week_rows: int = 50,
    seed: int = 42,
    model_dir: str = "./models",
    tb_log_name: str = "SAC_Pretrain",
    num_torch_threads: Optional[int] = None,
    thread_alloc_path: Optional[str] = None,
) -> Dict[str, Any]:
    if num_torch_threads is not None:
        torch.set_num_threads(num_torch_threads)

    # If given (by a sweep driver running several configs in parallel), watches
    # thread_alloc_path for updated thread-count recommendations as sibling configs
    # finish, so this run can absorb logical cores they free up instead of being stuck at
    # its launch-time allocation for its entire (possibly much longer) remaining runtime.
    thread_poller = ThreadAllocPoller(thread_alloc_path)
    thread_poller.start()

    os.makedirs(model_dir, exist_ok=True)
    df = prepare_data(csv_path)

    weeks = sorted(df['week_label'].unique())
    if len(weeks) < pretrain_weeks_count + 1:
        raise ValueError(f"Not enough data. Needed >{pretrain_weeks_count} weeks, got {len(weeks)}.")

    pretrain_weeks = weeks[:pretrain_weeks_count]
    walk_forward_weeks = weeks[pretrain_weeks_count:]
    validation_weeks = set(walk_forward_weeks[:val_weeks_count])
    test_weeks = set(walk_forward_weeks[val_weeks_count:])

    print(f"Pretraining on {len(pretrain_weeks)} weeks: {pretrain_weeks[0]} to {pretrain_weeks[-1]}")
    pretrain_df = df[df['week_label'].isin(pretrain_weeks)].copy()

    pretrain_env = DummyVecEnv([lambda: XAUEnv(pretrain_df)])

    model = SAC(
        "MlpPolicy",
        pretrain_env,
        policy_kwargs=dict(net_arch=[128, 128]),
        replay_buffer_class=WeeklyRollingBuffer,
        buffer_size=buffer_size,
        seed=seed,
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )

    cb = WeekRolloverCallback(verbose=1)
    start_time = time.time()

    model.learn(total_timesteps=pretrain_timesteps, callback=cb, tb_log_name=tb_log_name)
    print(f"Pre-training complete. Wall time: {time.time() - start_time:.2f}s")
    model.save(os.path.join(model_dir, "sac_xauusd_pretrained.zip"))

    oos_equity_curve = [10000.0]
    equity_curve_weeks: List[str] = []
    running_peak_balance = 10000.0

    # Trade-level and weekly-return records for the tearsheet (compute_trade_stats /
    # compute_sharpe_sortino / compute_weekly_pnl_distribution below). Positions never
    # carry across week boundaries (each week gets a fresh XAUEnv), so trades are always
    # opened and closed within a single week's loop iteration.
    trades: List[Dict[str, Any]] = []
    weekly_pnl: List[float] = []
    weekly_returns: List[float] = []
    # Per-week open/close gold price, parallel to equity_curve_weeks - lets the buy-and-hold
    # benchmark be computed over the exact same windows as the strategy's own metrics.
    week_open_prices: List[float] = []
    week_close_prices: List[float] = []

    for step_idx, w in enumerate(tqdm(walk_forward_weeks, desc="WFO Progress", unit="week")):
        w_df = df[df['week_label'] == w].copy()

        if len(w_df) < min_week_rows:
            continue

        # Carry balance/peak forward instead of resetting to $10k each week, so the
        # env's own drawdown penalty and 20% hard stop react to true cumulative
        # equity (matching mt5bridge.py's PEAK_BALANCE) instead of each week in isolation.
        wf_env = DummyVecEnv([lambda: XAUEnv(w_df, initial_balance=oos_equity_curve[-1], initial_peak_balance=running_peak_balance)])
        loop_start = time.time()

        rollout = rollout_week(
            wf_env,
            predict_fn=lambda obs: model.predict(obs, deterministic=True)[0],
            week_label=w,
            on_transition=lambda obs, next_obs, action, rewards, dones, infos: model.replay_buffer.add(obs, next_obs, action, rewards, dones, infos),
        )
        ep_reward = rollout["ep_reward"]
        ep_pnl = rollout["ep_pnl"]
        trades.extend(rollout["trades"])
        last_info = rollout["last_info"]

        start_of_week_balance = oos_equity_curve[-1]
        weekly_pnl.append(ep_pnl)
        weekly_returns.append(ep_pnl / start_of_week_balance if start_of_week_balance else 0.0)
        week_open_prices.append(float(w_df.iloc[0]['open']))
        week_close_prices.append(float(w_df.iloc[-1]['close']))

        new_balance = oos_equity_curve[-1] + ep_pnl
        oos_equity_curve.append(new_balance)
        equity_curve_weeks.append(w)
        running_peak_balance = last_info['peak_balance']

        if len(model.replay_buffer.week_index_map) > replay_window_weeks:
            model.replay_buffer.purge_oldest_week()

        model.train(gradient_steps=gradient_steps, batch_size=batch_size)

        occupancy = model.replay_buffer.valid_mask.sum()

        model.logger.record("wfo/episode_reward", ep_reward)
        model.logger.record("wfo/episode_pnl", ep_pnl)
        model.logger.record("wfo/buffer_occupancy", occupancy)
        model.logger.dump(step=step_idx)

        model.save(os.path.join(model_dir, f"sac_xauusd_week_{w}.zip"))

        elapsed = time.time() - loop_start
        tqdm.write(f"| WFO Complete: {w} | Reward: {ep_reward:7.2f} | PnL: ${ep_pnl:7.2f} | Buffer Occ: {occupancy}/{buffer_size} | Time: {elapsed:5.2f}s |")

    full_metrics = build_segment_metrics(
        oos_equity_curve, weekly_returns, weekly_pnl, trades,
        start_price=week_open_prices[0] if week_open_prices else None,
        end_price=week_close_prices[-1] if week_close_prices else None,
    )
    print_backtest_summary("WALK-FORWARD OUT-OF-SAMPLE BACKTEST RESULTS (FULL)", full_metrics)

    # Split the equity curve (and the parallel weekly-return/PnL/trade/price records) at
    # the validation/test boundary using the weeks that were actually processed (some
    # weeks may have been skipped via min_week_rows), so config selection can be based on
    # the validation segment alone without touching test.
    val_curve = [oos_equity_curve[0]]
    val_returns: List[float] = []
    val_pnl: List[float] = []
    val_open_price: Optional[float] = None
    val_close_price: Optional[float] = None
    test_curve: Optional[List[float]] = None
    test_returns: List[float] = []
    test_pnl: List[float] = []
    test_open_price: Optional[float] = None
    test_close_price: Optional[float] = None
    for w, bal, wret, wpnl, wopen, wclose in zip(
        equity_curve_weeks, oos_equity_curve[1:], weekly_returns, weekly_pnl, week_open_prices, week_close_prices
    ):
        if w in validation_weeks:
            val_curve.append(bal)
            val_returns.append(wret)
            val_pnl.append(wpnl)
            if val_open_price is None:
                val_open_price = wopen
            val_close_price = wclose
        elif w in test_weeks:
            if test_curve is None:
                test_curve = [val_curve[-1]]
                test_open_price = wopen
            test_curve.append(bal)
            test_returns.append(wret)
            test_pnl.append(wpnl)
            test_close_price = wclose

    val_trades = [t for t in trades if t["week"] in validation_weeks]
    test_trades = [t for t in trades if t["week"] in test_weeks]

    results: Dict[str, Any] = {
        "oos_equity_curve": oos_equity_curve,
        "equity_curve_weeks": equity_curve_weeks,
        "weekly_pnl": weekly_pnl,
        "trades": trades,
        "full": full_metrics,
    }

    if len(val_curve) > 1:
        val_metrics = build_segment_metrics(val_curve, val_returns, val_pnl, val_trades, val_open_price, val_close_price)
        print_backtest_summary("VALIDATION SEGMENT RESULTS", val_metrics)
        results["validation"] = val_metrics

    if test_curve and len(test_curve) > 1:
        test_metrics = build_segment_metrics(test_curve, test_returns, test_pnl, test_trades, test_open_price, test_close_price)
        print_backtest_summary("TEST SEGMENT RESULTS (only inspect once, for the already-chosen winner)", test_metrics)
        results["test"] = test_metrics

    thread_poller.stop()
    return results

def generate_dummy_csv(path="dummy_xauusd.csv"):
    if not os.path.exists(path):
        print("Generating mock data to allow execution...")
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=60000, freq="5min", tz="UTC")
        close = 2000.0 + np.random.randn(60000).cumsum()
        df = pd.DataFrame({
            "timestamp": dates,
            "open": close + np.random.randn(60000) * 0.5,
            "high": close + np.random.rand(60000) * 2,
            "low": close - np.random.rand(60000) * 2,
            "close": close,
            "volume": np.random.randint(100, 1000, size=60000)
        })
        df.to_csv(path, index=False)
    return path

if __name__ == "__main__":
    test_csv = "data/data.csv"
    run_wfo_pipeline(test_csv)