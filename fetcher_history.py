import pandas as pd
from datasets import load_dataset

def fetch_and_inspect():
    try:
        dataset = load_dataset("Ehsanrs2/Forex_Factory_Calendar", split="train")
        df = pd.DataFrame(dataset)
        
        print("\nDataset successfully loaded into memory!")
        print(f"Total Events Downloaded: {len(df):,}")
        
        print("\n--- Column Names ---")
        print(df.columns.tolist())
        
        print("\n--- Data Sample (First 3 Rows) ---")
        print(df.head(3).to_string())
        df.to_csv("raw_historical_news.csv", index=False)
        print("\nSaved to 'raw_historical_news.csv'")
        
    except Exception as e:
        print(f"Error fetching dataset: {e}")

if __name__ == "__main__":
    fetch_and_inspect()