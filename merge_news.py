import pandas as pd
import numpy as np

def integrate_macro_data(price_csv="data/data.csv", news_csv="raw_historical_news.csv", output_csv="data_with_news.csv"):
    print("Loading historical M5 price data")
    df_price = pd.read_csv(price_csv)
    df_price['timestamp'] = pd.to_datetime(df_price['timestamp'], utc=True)

    print("Loading raw Forex Factory data")
    df_news = pd.read_csv(news_csv)

    print("Filtering and converting timezones")
    df_news = df_news[
        (df_news['Currency'] == 'USD') & 
        (df_news['Impact'] == 'High Impact Expected')
    ].copy()
    df_news['timestamp'] = pd.to_datetime(df_news['DateTime'], utc=True)
    df_news['timestamp'] = df_news['timestamp'].dt.floor('5min')
    flags = pd.DataFrame({'timestamp': df_news['timestamp'].unique(), 'high_impact_news': 1})
    df_merged = pd.merge(df_price, flags, on='timestamp', how='left')
    df_merged['high_impact_news'] = df_merged['high_impact_news'].fillna(0).astype(int)

    total_events = df_merged['high_impact_news'].sum()
    print(f"Merge Complete! Found {total_events} M5 candles containing High-Impact USD news.")
    df_merged.to_csv(output_csv, index=False)
    print(f"Saved fully aligned dataset to: {output_csv}")

if __name__ == "__main__":
    integrate_macro_data()