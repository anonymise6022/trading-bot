import pandas as pd
import numpy as np

symbol = 'BTC/USDT'
url = 'BTCUSDT-1h.csv'

df = pd.read_csv(url)

# Calculate close log return
df['close_log_return'] = np.log(df['close'] / df['close'].shift(1))

# Calculate future log return (target)
df['future_close_log_return'] = np.log(
    df['close'].shift(-24) / df['close']
)

# Previous returns as features
for lag in range(1, 7):
    df[f'close_log_return_lag_{lag}'] = (
        df['close_log_return'].shift(lag)
    )

# Calculate the 20-period EMA of the close price, where ewm is the exponential weighted function in pandas. The span parameter controls the decay, with a higher span giving more weight to recent prices. The adjust parameter is set to False to use the simple moving average formula.
df['ema5'] = df['close'].ewm(span=5, adjust=False).mean()
df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()

# Previous candle values only
df["ema_5_prev"] = df["ema5"].shift(1)
df["ema_20_prev"] = df["ema20"].shift(1)

# Detect crossover
#df["ema_cross"] = 0

# bullish crossover
#df.loc[(df["ema_5_prev"] <= df["ema_20_prev"]) & (df["ema5"] > df["ema20"]),"ema_cross"] = 1

# bearish crossover
#df.loc[(df["ema_5_prev"] >= df["ema_20_prev"]) & (df["ema5"] < df["ema20"]),"ema_cross"] = -1

df["ema_trend_strength"] = (df["ema5"].shift(1) -df["ema20"].shift(1)) / df["ema20"].shift(1)

# EMA distance
df['ema_distance'] = (df['close'] - df['ema20']) / df['ema20']

# Remove missing rows
df = df.dropna()
#df = df.drop(columns = ['ema5', 'ema20', 'ema_5_prev', 'ema_20_prev', 'ema_trend_strength', 'ema_cross'])

df.to_csv('BTCUSDT-1hNEW.csv', index=False)

print(f"Historical data for {symbol} saved to BTCUSDT-1hNEW.csv")