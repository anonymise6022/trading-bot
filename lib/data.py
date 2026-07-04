import pandas as pd
import ccxt

def download_his_data():
    exchange = ccxt.binance()

    symbol = 'BTC/USDT'
    timeframe = '1h'  # Hourly data
    limit = 1000  # Number of data points to fetch

    print(f"fetching historical data for {symbol}...")

    raw_candles = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(raw_candles, columns=columns)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    df.to_csv('historical_data.csv', index=False)
    print(f"Historical data for {symbol} saved to 'historical_data.csv'.")

if __name__ == "__main__":
    download_his_data()