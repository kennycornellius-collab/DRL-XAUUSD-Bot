import os
import time
import collections
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pandas_ta_classic as ta
from tqdm import tqdm

import gymnasium as gym
from gymnasium import spaces

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv


def prepare_data(csv_path: str) -> pd.DataFrame:
    print(f"Loading data from {csv_path}...")
    df_m5 = pd.read_csv(csv_path)
    df_m5['timestamp'] = pd.to_datetime(df_m5['timestamp'], utc=True)
    df_m5.sort_values('timestamp', ascending=True, inplace=True)
    df_m5.reset_index(drop=True, inplace=True)

    adx_df = df_m5.ta.adx(length=14)
    if adx_df is not None and 'ADX_14' in adx_df.columns:
         df_m5['adx'] = adx_df['ADX_14']
    else:
         
         df_m5['adx'] = np.nan

    
    df_m5.set_index('timestamp', inplace=True)
    
    
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'high_impact_news': 'max'
    }
    
    df_m15 = df_m5.resample('15min', closed='left', label='left').agg(agg_dict)
    df_m15.reset_index(inplace=True)

    df_m5_adx = df_m5[['adx']].dropna().reset_index()
    df_m15 = pd.merge_asof(
        df_m15.sort_values('timestamp'), 
        df_m5_adx.sort_values('timestamp'), 
        on='timestamp', 
        direction='backward'
    )

    df_m15.dropna(inplace=True)
    df_m15.reset_index(drop=True, inplace=True)
   
    df_m15['week_label'] = df_m15['timestamp'].dt.strftime("%Y-W%V")
    
    print(f"Data prepped. M15 Shape: {df_m15.shape}")
    return df_m15

class XAUEnv(gym.Env):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.max_steps = len(self.df) - 1

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)

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
        self.balance = 10000.0  
        self.peak_balance = 10000.0
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

        current_news_flag = self.df.iloc[self.current_step]['high_impact_news']
        if current_news_flag == 1:
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
            "balance": self.balance
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

        news_flag = float(self.df.iloc[self.current_step]['high_impact_news'])
        return np.concatenate([scaled, [pos, ur_pnl, dd, news_flag]]).astype(np.float32)


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


def run_wfo_pipeline(csv_path: str):
    os.makedirs("./models", exist_ok=True)
    df = prepare_data(csv_path)

    weeks = sorted(df['week_label'].unique())
    if len(weeks) < 27:
        raise ValueError(f"Not enough data. Needed >26 weeks, got {len(weeks)}.")

    pretrain_weeks = weeks[:26]
    walk_forward_weeks = weeks[26:]

    print(f"Pretraining on {len(pretrain_weeks)} weeks: {pretrain_weeks[0]} to {pretrain_weeks[-1]}")
    pretrain_df = df[df['week_label'].isin(pretrain_weeks)].copy()

    pretrain_env = DummyVecEnv([lambda: XAUEnv(pretrain_df)])
    

    MAX_BUFFER_SIZE = 1920
    
    model = SAC(
        "MlpPolicy", 
        pretrain_env, 
        policy_kwargs=dict(net_arch=[128, 128]),
        replay_buffer_class=WeeklyRollingBuffer,
        buffer_size=MAX_BUFFER_SIZE,
        seed=42,
        verbose=1, 
        tensorboard_log="./tensorboard_logs/"
    )

    cb = WeekRolloverCallback(verbose=1)
    start_time = time.time()

    model.learn(total_timesteps=50_000, callback=cb, tb_log_name="SAC_Pretrain")
    print(f"Pre-training complete. Wall time: {time.time() - start_time:.2f}s")
    model.save("./models/sac_xauusd_pretrained.zip")

    oos_equity_curve = [10000.0]

    for step_idx, w in enumerate(tqdm(walk_forward_weeks, desc="WFO Progress", unit="week")):
        w_df = df[df['week_label'] == w].copy()

        if len(w_df) < 50: 
            continue
            
        wf_env = DummyVecEnv([lambda: XAUEnv(w_df)])
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

        if len(model.replay_buffer.week_index_map) > 4:
            model.replay_buffer.purge_oldest_week()

        model.train(gradient_steps=1000, batch_size=256)
        
        occupancy = model.replay_buffer.valid_mask.sum()

        model.logger.record("wfo/episode_reward", ep_reward)
        model.logger.record("wfo/episode_pnl", ep_pnl)
        model.logger.record("wfo/buffer_occupancy", occupancy)
        model.logger.dump(step=step_idx) 

        model.save(f"./models/sac_xauusd_week_{w}.zip")
        
        elapsed = time.time() - loop_start
        tqdm.write(f"| WFO Complete: {w} | Reward: {ep_reward:7.2f} | PnL: ${ep_pnl:7.2f} | Buffer Occ: {occupancy}/{MAX_BUFFER_SIZE} | Time: {elapsed:5.2f}s |")


    print("\n" + "="*50)
    print("WALK-FORWARD OUT-OF-SAMPLE BACKTEST RESULTS")
    print("="*50)
    start_bal = oos_equity_curve[0]
    end_bal = oos_equity_curve[-1]
    net_profit = end_bal - start_bal
    ret_pct = (net_profit / start_bal) * 100
    
    eq_arr = np.array(oos_equity_curve)
    peaks = np.maximum.accumulate(eq_arr)
    drawdowns = (peaks - eq_arr) / peaks
    max_dd = np.max(drawdowns) * 100

    print(f"Starting Balance:  ${start_bal:,.2f}")
    print(f"Ending Balance:    ${end_bal:,.2f}")
    print(f"Net Profit:        ${net_profit:,.2f} ({ret_pct:+.2f}%)")
    print(f"Max Drawdown:      {max_dd:.2f}%")
    print("="*50 + "\n")

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
    test_csv = "data/data_merged.csv"
    run_wfo_pipeline(test_csv)