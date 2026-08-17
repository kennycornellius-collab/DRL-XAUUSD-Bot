import numpy as np
import pandas as pd
import pandas_ta_classic as ta  # noqa: F401  (registers the .ta accessor used below)

# The columns both xau.py (training) and mt5bridge.py (live) window-normalize together
# for the observation vector. A single shared list so the two can't drift apart the way
# ADX once did before compute_m15_features existed - both files import this instead of
# hardcoding their own copy.
MARKET_FEATURE_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'adx', 'atr', 'rsi', 'adx_h1']


def compute_m15_features(df_m5: pd.DataFrame) -> pd.DataFrame:
    """Shared by xau.py (training) and mt5bridge.py (live) so features can't drift
    between the two paths again: every indicator here is computed once, in this one
    place, and both callers read the same output columns.

    df_m5 must have columns: timestamp, open, high, low, close, volume.

    Columns produced (beyond open/high/low/close/volume/timestamp):
    - adx: ADX(14) computed on M5 bars, resampled to M15 and backward-merged (never
      computed directly on M15 - this is the original train/live mismatch fix).
    - atr, rsi: ATR(14)/RSI(14), computed and merged the same M5-then-merge way as adx.
    - adx_h1: ADX(14) computed on H1 bars derived from the M15 series (a higher-timeframe
      trend-strength signal alongside the M15-native adx). Merged back with a +1h shift
      so an M15 row only ever matches an H1 bar that has ALREADY closed by that row's own
      timestamp - an H1 bar labeled e.g. 10:00 spans [10:00, 11:00) and isn't actually
      complete until 11:00, so matching without the shift would leak up to ~45 minutes of
      not-yet-happened price data into M15 rows inside that hour.
    - session_sin, session_cos: cyclical (sin/cos) encoding of hour-of-day, computed
      directly from each row's own timestamp - no merge, so no lookahead risk, and it
      keeps e.g. 23:45 and 00:00 numerically adjacent instead of wrapping discontinuously.
    """
    df_m5 = df_m5.copy()
    df_m5.sort_values('timestamp', ascending=True, inplace=True)
    df_m5.reset_index(drop=True, inplace=True)

    adx_df = df_m5.ta.adx(length=14)
    df_m5['adx'] = adx_df['ADX_14'] if adx_df is not None and 'ADX_14' in adx_df.columns else np.nan
    df_m5['atr'] = df_m5.ta.atr(length=14)
    df_m5['rsi'] = df_m5.ta.rsi(length=14)

    df_m5.set_index('timestamp', inplace=True)

    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    df_m15 = df_m5.resample('15min', closed='left', label='left').agg(agg_dict)
    df_m15.reset_index(inplace=True)

    df_m5_indicators = df_m5[['adx', 'atr', 'rsi']].dropna().reset_index()
    df_m15 = pd.merge_asof(
        df_m15.sort_values('timestamp'),
        df_m5_indicators.sort_values('timestamp'),
        on='timestamp',
        direction='backward'
    )

    # Higher-timeframe trend strength: resample the M15 series (not M5) up to H1, compute
    # ADX(14) there, then merge back - see the "adx_h1" note in the docstring for why the
    # merge key is shifted by +1h.
    df_h1 = df_m15.set_index('timestamp')[['open', 'high', 'low', 'close', 'volume']] \
        .resample('1h', closed='left', label='left').agg(agg_dict)
    h1_adx_df = df_h1.ta.adx(length=14)
    df_h1['adx_h1'] = h1_adx_df['ADX_14'] if h1_adx_df is not None and 'ADX_14' in h1_adx_df.columns else np.nan
    df_h1 = df_h1[['adx_h1']].dropna().reset_index()
    df_h1['available_at'] = df_h1['timestamp'] + pd.Timedelta(hours=1)

    df_m15 = pd.merge_asof(
        df_m15.sort_values('timestamp'),
        df_h1[['available_at', 'adx_h1']].sort_values('available_at'),
        left_on='timestamp',
        right_on='available_at',
        direction='backward'
    )
    df_m15.drop(columns=['available_at'], inplace=True)

    hour_frac = df_m15['timestamp'].dt.hour + df_m15['timestamp'].dt.minute / 60.0
    angle = 2 * np.pi * hour_frac / 24.0
    df_m15['session_sin'] = np.sin(angle)
    df_m15['session_cos'] = np.cos(angle)

    df_m15.dropna(inplace=True)
    df_m15.reset_index(drop=True, inplace=True)
    return df_m15
