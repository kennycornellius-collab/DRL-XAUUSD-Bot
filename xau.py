import os
import time
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

from features import compute_m15_features


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

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.start_index = 0

        self.history_200 = collections.deque(maxlen=200)
        self.returns_50 = collections.deque(maxlen=50)

    def current_week_label(self) -> str:
        if self.current_step >= len(self.df):
            return self.df.iloc[-1]['week_label']
        return self.df.iloc[self.current_step]['week_label']

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.start_index
        
        self.position = 0
        self.entry_price = 0.0
        self.balance = self.initial_balance
        self.peak_balance = self.initial_peak_balance
        self.current_unrealized_pnl = 0.0
        self.current_dd = 0.0

        self.history_200.clear()
        self.returns_50.clear()

        self._append_history()
        return self._get_obs(), {"week_label": self.current_week_label()}

    def _append_history(self):
        row = self.df.iloc[self.current_step]
        feats = np.array([
            row['open'], row['high'], row['low'], 
            row['close'], row['volume'], row['adx']
        ], dtype=np.float32)
        self.history_200.append(feats)

    def step(self, action: np.ndarray):
        act_val = action[0]

        if act_val < -0.3:
            new_pos = -1
        elif act_val > 0.3:
            new_pos = 1
        else:
            new_pos = 0

        spread_cost = 0.0
        if new_pos != 0 and new_pos != self.position:
            spread_cost = 0.30

        current_close = self.df.iloc[self.current_step]['close']

        self.current_step += 1
        done = False
        truncated = False

        if self.current_step >= self.max_steps:
            done = True
            next_close = current_close
        else:
            next_close = self.df.iloc[self.current_step]['close']

        step_pnl = self.position * (next_close - current_close)
        step_return = step_pnl - spread_cost
        
        self.returns_50.append(step_return)
        self.balance += step_return
        
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        self.current_dd = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0.0
        
        neg_returns = [r for r in self.returns_50 if r < 0]
        downside_std = np.std(neg_returns) if len(neg_returns) >= 2 else 1.0

        reward = step_return / (downside_std + 1e-8)
        if self.current_dd > 0.05:
            reward -= 0.5 * self.current_dd

        self.position = new_pos
        
        if self.position != 0:
            if self.entry_price == 0.0:
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
            "week_label": self.current_week_label() if not done else self.df.iloc[-1]['week_label'],
            "step_pnl": step_return,
            "balance": self.balance,
            "peak_balance": self.peak_balance
        }
        return self._get_obs(), float(reward), done, truncated, info

    def _get_obs(self) -> np.ndarray:
        hist = np.array(self.history_200)

        mins = hist.min(axis=0)
        maxs = hist.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1e-8

        current_features = hist[-1]
        scaled = (current_features - mins) / ranges
        scaled = (scaled * 2.0) - 1.0
        scaled = np.clip(scaled, -1.0, 1.0)

        pos = float(self.position)
        ur_pnl = np.clip(self.current_unrealized_pnl / 100.0, -1.0, 1.0)
        dd = np.clip(self.current_dd, -1.0, 1.0)

        return np.concatenate([scaled, [pos, ur_pnl, dd]]).astype(np.float32)


class WeeklyRollingBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid_mask = np.zeros(self.buffer_size, dtype=bool)
        self.week_index_map = collections.defaultdict(list)

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, 
            reward: np.ndarray, done: np.ndarray, infos: List[Dict[str, Any]]) -> None:
        
        week_label = infos[0]['week_label']
        idx = self.pos

        for w, inds in list(self.week_index_map.items()):
            if idx in inds:
                inds.remove(idx)
                if not inds:
                    del self.week_index_map[w]
                break

        self.week_index_map[week_label].append(idx)
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

    def sample(self, batch_size: int, env: Optional[DummyVecEnv] = None):
        valid_indices = np.where(self.valid_mask)[0]

        if len(valid_indices) == 0:
            valid_indices = np.arange(self.buffer_size)

        replace = len(valid_indices) < batch_size
        batch_inds = np.random.choice(valid_indices, size=batch_size, replace=replace)
        
        return self._get_samples(batch_inds, env=env)


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


def print_backtest_summary(title: str, metrics: Dict[str, float]) -> None:
    print("\n" + "="*50)
    print(title)
    print("="*50)
    print(f"Starting Balance:  ${metrics['start_balance']:,.2f}")
    print(f"Ending Balance:    ${metrics['end_balance']:,.2f}")
    print(f"Net Profit:        ${metrics['net_profit']:,.2f} ({metrics['net_profit_pct']:+.2f}%)")
    print(f"Max Drawdown:      {metrics['max_drawdown_pct']:.2f}%")
    print("="*50 + "\n")


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
) -> Dict[str, Any]:
    if num_torch_threads is not None:
        torch.set_num_threads(num_torch_threads)

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

    for step_idx, w in enumerate(tqdm(walk_forward_weeks, desc="WFO Progress", unit="week")):
        w_df = df[df['week_label'] == w].copy()

        if len(w_df) < min_week_rows:
            continue

        # Carry balance/peak forward instead of resetting to $10k each week, so the
        # env's own drawdown penalty and 20% hard stop react to true cumulative
        # equity (matching mt5bridge.py's PEAK_BALANCE) instead of each week in isolation.
        wf_env = DummyVecEnv([lambda: XAUEnv(w_df, initial_balance=oos_equity_curve[-1], initial_peak_balance=running_peak_balance)])
        obs = wf_env.reset()
        done = False

        ep_reward = 0.0
        ep_pnl = 0.0
        loop_start = time.time()

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, rewards, dones, infos = wf_env.step(action)
            model.replay_buffer.add(obs, next_obs, action, rewards, dones, infos)

            ep_reward += rewards[0]
            ep_pnl += infos[0]['step_pnl']
            obs = next_obs
            done = dones[0]

        new_balance = oos_equity_curve[-1] + ep_pnl
        oos_equity_curve.append(new_balance)
        equity_curve_weeks.append(w)
        running_peak_balance = infos[0]['peak_balance']

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

    full_metrics = compute_backtest_metrics(oos_equity_curve)
    print_backtest_summary("WALK-FORWARD OUT-OF-SAMPLE BACKTEST RESULTS (FULL)", full_metrics)

    # Split the equity curve at the validation/test boundary using the weeks that were
    # actually processed (some weeks may have been skipped via min_week_rows), so config
    # selection can be based on the validation segment alone without touching test.
    val_curve = [oos_equity_curve[0]]
    test_curve: Optional[List[float]] = None
    for w, bal in zip(equity_curve_weeks, oos_equity_curve[1:]):
        if w in validation_weeks:
            val_curve.append(bal)
        elif w in test_weeks:
            if test_curve is None:
                test_curve = [val_curve[-1]]
            test_curve.append(bal)

    results: Dict[str, Any] = {
        "oos_equity_curve": oos_equity_curve,
        "equity_curve_weeks": equity_curve_weeks,
        "full": full_metrics,
    }

    if len(val_curve) > 1:
        val_metrics = compute_backtest_metrics(val_curve)
        print_backtest_summary("VALIDATION SEGMENT RESULTS", val_metrics)
        results["validation"] = val_metrics

    if test_curve and len(test_curve) > 1:
        test_metrics = compute_backtest_metrics(test_curve)
        print_backtest_summary("TEST SEGMENT RESULTS (only inspect once, for the already-chosen winner)", test_metrics)
        results["test"] = test_metrics

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