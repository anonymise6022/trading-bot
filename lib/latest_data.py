import pandas as pd
import numpy as np
import ccxt
import requests

# -------------------------------------------------------
# 1. FUNCTION TO DOWNLOAD HISTORICAL DATA FROM BINANCE
# -------------------------------------------------------

def download_his_data():
    exchange = ccxt.binance()
    
    symbol = 'BTC/USDT'
    timeframe = '1h'
    limit = 1000  # Number of candles to fetch

    print(f"Fetching historical data for {symbol}...")

    raw_candles = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(raw_candles, columns=columns)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Calculate log return
    df['close_log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Lag it from 1 to 6 candles
    for lag in range(1, 7):
        df[f'close_log_return_lag_{lag}'] = df['close_log_return'].shift(lag)

    # Remove rows with NaN values created by shift()
    df = df.dropna()

# -------------------------------------------------------
# 2. DOWNLOAD OPEN INTEREST DATA FROM BINANCE
# -------------------------------------------------------
    
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    params = {
    "symbol": "BTCUSDT",
    "period": "1h",
    "limit": 1000
    }

    data = requests.get(url, params=params).json() 
    oi_df = pd.DataFrame(data)

    # Convert timestamp
    oi_df["timestamp"] = pd.to_datetime(oi_df["timestamp"], unit="ms")

    # Rename the useful columns
    oi_df = oi_df.rename(columns={
    "sumOpenInterest": "open_interest",
    "sumOpenInterestValue": "open_interest_value"
    })

    # Keep only what I need
    oi_df = oi_df[
    ["timestamp", "open_interest", "open_interest_value"]
    ]   

# -------------------------------------------------------
# 3. MERGE THE DATAFRAMES
# -------------------------------------------------------

    merged = df.merge(oi_df, on="timestamp", how="left")
    merged = merged.dropna(subset=["open_interest"]) 

    # Convert the columns to numeric, coercing errors to NaN
    merged["open_interest"] = pd.to_numeric(merged["open_interest"], errors="coerce")
    merged["open_interest_value"] = pd.to_numeric(merged["open_interest_value"], errors="coerce")

# -------------------------------------------------------
# 4. NOTIONAL OI AND LAGS
# -------------------------------------------------------
    
# calculating notional open interest
    merged["notional_oi"] = (merged["open_interest"] * merged["close"])

    # calculating the momentum of notional open interest
    merged["oi_momentum"] = (merged["notional_oi"] / merged["notional_oi"].shift(1)) - 1

    # creating lags
    for lag in range(1, 7):
        merged[f"oi_momentum_lag_{lag}"] = (
            merged["oi_momentum"].shift(lag)
        )   
    merged = merged.dropna()

# -------------------------------------------------------
# 5. SAVE THE MERGED DATA TO CSV
# -------------------------------------------------------

    merged.to_csv('historical_data.csv', index=False)

    print(f"Historical data for {symbol} saved to 'historical_data.csv'.")

if __name__ == "__main__":
    download_his_data()