import pandas as pd
import numpy as np

symbol = 'BTC/USDT'
url = 'BTCUSDT-1h.csv'

df = pd.read_csv(url)

# Calculate log return (target)
df['close_log_return'] = np.log(df['close'] / df['close'].shift(1))

# Previous returns as features
for lag in range(1, 7):
    df[f'close_log_return_lag_{lag}'] = (
        df['close_log_return'].shift(lag)
    )

# Calculate moving averages
df["sma_20"] = df["close"].rolling(20).mean()
df["sma_50"] = df["close"].rolling(50).mean()

# Previous candle values only
df["sma_20_prev"] = df["sma_20"].shift(1)
df["sma_50_prev"] = df["sma_50"].shift(1)

# Detect crossover
df["sma_cross"] = 0

# bullish crossover
df.loc[(df["sma_20_prev"] <= df["sma_50_prev"]) & (df["sma_20"] > df["sma_50"]),"sma_cross"] = 1

# bearish crossover
df.loc[(df["sma_20_prev"] >= df["sma_50_prev"]) & (df["sma_20"] < df["sma_50"]),"sma_cross"] = -1

df["sma_trend_strength"] = (df["sma_20"].shift(1) -df["sma_50"].shift(1)) / df["sma_50"].shift(1)

# Remove missing rows
df = df.dropna()

df.to_csv('BTCUSDT-1hNEW.csv', index=False)

print(f"Historical data for {symbol} saved to BTCUSDT-1h.csv")