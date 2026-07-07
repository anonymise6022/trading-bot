import pandas as pd
import numpy as np
import ccxt

def download_his_data():
    exchange = ccxt.binance()

    symbol = 'BTC/USDT'
    timeframe = '1h'
    limit = 1000

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

    df.to_csv('historical_data.csv', index=False)

    print(f"Historical data for {symbol} saved to 'historical_data.csv'.")

if __name__ == "__main__":
    download_his_data()