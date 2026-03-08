import os
import sys
import time
import requests
import datetime
import numpy as np
import pandas as pd
import pandas_ta_classic as ta
import MetaTrader5 as mt5

from stable_baselines3 import SAC

SYMBOL = "XAUUSDm"
TIMEFRAME = mt5.TIMEFRAME_M15 
MAGIC_NUMBER = 777777
FIXED_LOTS = 0.01  
DEVIATION = 20    


MODEL_PATH = "./model1/sac_xauusd_week_2026-W09.zip" # Change to your best week's zip file!

DISCORD_WEBHOOK_URL = ""
DISCORD_USER_ID = ""


PEAK_BALANCE = 0.0

def connect_mt5():
    if not mt5.initialize():
        print("MT5 Initialization Failed")
        sys.exit()
    
    if not mt5.symbol_select(SYMBOL, True):
        print(f"Failed to select {SYMBOL}. Check your broker symbol name.")
        sys.exit()
        
    print(f"MT5 Connected. Tracking {SYMBOL} on M15.")

def send_discord_alert(message, is_alert=False):
    if not DISCORD_WEBHOOK_URL: return
    content = f"<@{DISCORD_USER_ID}> {message}" if (is_alert and DISCORD_USER_ID) else message
    try: 
        requests.post(DISCORD_WEBHOOK_URL, json={"content": content, "username": "DRL Bot"})
    except: 
        pass


def get_drl_observation():
    global PEAK_BALANCE
    
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 250)
    if rates is None:
        print("Failed to get MT5 rates.")
        return None
        
    df = pd.DataFrame(rates)
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    
    adx_df = df.ta.adx(length=14)
    if adx_df is not None and 'ADX_14' in adx_df.columns:
         df['adx'] = adx_df['ADX_14']
    else:
         df['adx'] = np.nan
         
    df.dropna(inplace=True)
    
    hist_df = df[['open', 'high', 'low', 'close', 'volume', 'adx']].tail(200)
    if len(hist_df) < 200:
        print("Not enough clean data. Waiting...")
        return None
        
    hist = hist_df.values.astype(np.float32)

    mins = hist.min(axis=0)
    maxs = hist.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1e-8

    current_features = hist[-1] 
    scaled = (current_features - mins) / ranges
    scaled = (scaled * 2.0) - 1.0
    scaled = np.clip(scaled, -1.0, 1.0)

    account_info = mt5.account_info()
    if account_info is None:
        return None
        
    balance = account_info.balance
    equity = account_info.equity
    
    if balance > PEAK_BALANCE:
        PEAK_BALANCE = balance
        
    current_dd = (PEAK_BALANCE - equity) / PEAK_BALANCE if PEAK_BALANCE > 0 else 0.0

    positions = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_NUMBER)
    pos_val = 0.0
    ur_pnl = 0.0
    
    if positions:
        pos = positions[0] 
        if pos.type == mt5.POSITION_TYPE_BUY:
            pos_val = 1.0
        elif pos.type == mt5.POSITION_TYPE_SELL:
            pos_val = -1.0
            
        raw_pnl = pos.profit
        ur_pnl = raw_pnl / 100.0 

    ur_pnl = np.clip(ur_pnl, -1.0, 1.0)
    dd = np.clip(current_dd, -1.0, 1.0)

    obs = np.concatenate([scaled, [pos_val, ur_pnl, dd]]).astype(np.float32)
    return obs, int(pos_val)

def close_all_positions():
    positions = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_NUMBER)
    if not positions:
        return True
        
    for pos in positions:
        tick = mt5.symbol_info_tick(SYMBOL)
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
        
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": pos.ticket,
            "symbol": SYMBOL,
            "volume": pos.volume,
            "type": order_type,
            "price": price,
            "deviation": DEVIATION,
            "magic": MAGIC_NUMBER,
            "comment": "DRL Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(req)
        if res.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to close position {pos.ticket}: {res.comment}")
            return False
            
    send_discord_alert("Closed existing positions to flip/flatten.")
    return True

def execute_market_order(direction):
    tick = mt5.symbol_info_tick(SYMBOL)
    if direction == 1:
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
        word = "LONG"
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
        word = "SHORT"
        
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": FIXED_LOTS,
        "type": order_type,
        "price": price,
        "deviation": DEVIATION,
        "magic": MAGIC_NUMBER,
        "comment": "DRL Open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    res = mt5.order_send(req)
    if res.retcode == mt5.TRADE_RETCODE_DONE:
        msg = f"Executed **{word}** order. Price: {price} | Lot: {FIXED_LOTS}"
        print(msg)
        send_discord_alert(msg, is_alert=True)
    else:
        print(f"Failed to open {word}: {res.comment}")

def main():
    connect_mt5()
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}")
        sys.exit()
        
    print(f"Loading SAC Agent...")
    model = SAC.load(MODEL_PATH)
    send_discord_alert("**DRL Bot Online & Monitoring M15 Candles**")

    last_candle_time = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 1)[0]['time']
    print("Waiting for the close of the current 15m candle to start execution...")

    while True:
        try:
            current_rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 1)
            if current_rates is None: continue
            current_candle_time = current_rates[0]['time']
            
            if current_candle_time == last_candle_time:
                time.sleep(1)
                continue
                
            last_candle_time = current_candle_time
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{now_str}] New M15 Candle Formed. Calculating DRL State...")

            state_data = get_drl_observation()
            if state_data is None: continue
            
            obs, current_pos = state_data
            
            action, _ = model.predict(obs, deterministic=True)
            act_val = action[0]
            
            if act_val < -0.3:
                target_pos = -1
                action_str = "SHORT"
            elif act_val > 0.3:
                target_pos = 1
                action_str = "LONG"
            else:
                target_pos = 0
                action_str = "HOLD/FLAT"
                
            print(f"NN Action Value: {act_val:+.3f} -> Target: {action_str} | Current: {current_pos}")

            if target_pos != current_pos:
                if current_pos != 0:
                    closed_ok = close_all_positions()
                    if not closed_ok:
                        continue 
                if target_pos != 0:
                    execute_market_order(target_pos)
            else:
                print("No change required. Holding state.")

        except KeyboardInterrupt:
            print("\nShutting down DRL Bot...")
            send_discord_alert("**DRL Bot Manually Shut Down**")
            mt5.shutdown()
            sys.exit()
        except Exception as e:
            print(f"Runtime Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()