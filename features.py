import numpy as np
import pandas as pd
import pandas_ta_classic as ta  # noqa: F401  (registers the .ta accessor used below)


def compute_m15_features(df_m5: pd.DataFrame) -> pd.DataFrame:
    """Shared by xau.py (training) and mt5bridge.py (live) so ADX can't drift
    between the two paths again: ADX(14) is always computed on M5 bars, then
    resampled to M15 and backward-merged, never computed directly on M15.

    df_m5 must have columns: timestamp, open, high, low, close, volume.
    """
    df_m5 = df_m5.copy()
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
        'volume': 'sum'
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
    return df_m15
