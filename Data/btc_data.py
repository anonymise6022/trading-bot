import pandas as pd
import numpy as np

symbol = 'BTC/USDT'
url = 'BTCUSDT-1h.csv'

df = pd.read_csv(url)

# Current return
df["close_log_return"] = np.log(df["close"] / df["close"].shift(1))

# Future 6-hour return
df["future_close_log_return"] = np.log(df["close"].shift(-6) / df["close"])

df["signal"] = 0

# Strong upward movement
df.loc[df["future_close_log_return"] > 0.004,"signal"] = 1

# Strong downward movement
df.loc[df["future_close_log_return"] < -0.004,"signal"] = -1

df["signal"] = df["signal"] + 1

# -------------------------------------------------------
# 1. EMA TREND FEATURES
# -------------------------------------------------------

df["ema5"] = df["close"].ewm(span=5,adjust=False).mean()

df["ema20"] = df["close"].ewm(span=20,adjust=False).mean()

# Positive = bullish EMA trend
# Negative = bearish EMA trend
df["ema_trend_strength"] = (df["ema5"].shift(1) - df["ema20"].shift(1)) / df["ema20"].shift(1)

# Price distance from the EMA
# Positive = price above EMA
# Negative = price below EMA
df["ema_distance"] = (df["close"].shift(1) - df["ema20"].shift(1)) / df["ema20"].shift(1)

# -------------------------------------------------------
# 2. MOMENTUM
# -------------------------------------------------------

# 12-hour momentum
# Positive = price higher than 12 hours ago
# Negative = price lower than 12 hours ago
df["momentum_12"] = (df["close"].shift(1)/ df["close"].shift(13)) - 1

# -------------------------------------------------------
# 3. RSI-14
# -------------------------------------------------------

price_change = df["close"].diff()

gain = price_change.clip(lower=0)

loss = -price_change.clip(upper=0)

average_gain = gain.ewm(alpha=1 / 14,adjust=False).mean()

average_loss = loss.ewm(alpha=1 / 14,adjust=False).mean()

rs = average_gain / average_loss

df["rsi_14"] = (100 - 100 / (1 + rs))

# Shift it so the feature only uses
# information available before the prediction candle
df["rsi_14"] = (df["rsi_14"].shift(1))

df["rsi_normalized"] = (df["rsi_14"] - 50) / 50

# -------------------------------------------------------
# 4. RELATIVE VOLUME
# -------------------------------------------------------

volume_average = (df["volume"].rolling(20).mean())

df["relative_volume_20"] = (df["volume"].shift(1) / volume_average.shift(1))
df["volume_change"] = (df["volume"].shift(1) /df["volume"].shift(2))

# -------------------------------------------------------
# 5. ATR
# -------------------------------------------------------

high_low = df["high"] - df["low"]

high_close = abs(df["high"] - df["close"].shift(1))

low_close = abs(df["low"] - df["close"].shift(1))

tr = pd.concat([high_low, high_close, low_close],axis=1).max(axis=1)

df["atr_14"] = (tr.rolling(14).mean().shift(1))

df["atr_percent"] = (df["atr_14"] / df["close"].shift(1))

# -------------------------------------------------------
# 6. Volatility
# -------------------------------------------------------

df["volatility_20"] = (df["close_log_return"].rolling(20).std().shift(1))

# -------------------------------------------------------
# 7. ema crossover
# -------------------------------------------------------

df["ema_cross"] = 0

df.loc[(df["ema5"].shift(2) < df["ema20"].shift(2))&(df["ema5"].shift(1) > df["ema20"].shift(1)),"ema_cross"] = 1

df.loc[(df["ema5"].shift(2) > df["ema20"].shift(2))&(df["ema5"].shift(1) < df["ema20"].shift(1)),"ema_cross"] = -1

# -------------------------------------------------------
# 8. candle strength
# -------------------------------------------------------

df["candle_strength"] = ((df["close"] - df["open"]) /(df["high"] - df["low"]).replace(0,np.nan)).shift(1)

# -------------------------------------------------------
# 9. trend regime
# -------------------------------------------------------

df["trend_regime"] = (df["close"].shift(1) >df["ema20"].shift(1)).astype(int)

# Replace infinite values caused by division by zero
df = df.replace([np.inf, -np.inf], np.nan)

# Remove missing rows
df = df.dropna()
#df = df.drop(columns = ['ema5', 'ema20', 'ema_5_prev', 'ema_20_prev', 'ema_trend_strength', 'ema_cross'])

df.to_csv('BTCUSDT-1hNEW.csv', index=False)

print(f"Historical data for {symbol} saved to BTCUSDT-1hNEW.csv")

print(df["signal"].value_counts())