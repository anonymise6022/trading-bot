import pandas as pd
import numpy as np

symbol = 'BTC/USDT'
url = 'BTCUSDT-1h.csv'  # Replace with the actual URL of your CSV file
df = pd.read_csv(url) #reading the data from the url

# Calculate log return
df['close_log_return'] = np.log(df['close'] / df['close'].shift(1))

# Lag it from 1 to 6 candles and add it to csv file
for lag in range(1, 7):
    df[f'close_log_return_lag_{lag}'] = df['close_log_return'].shift(lag)

# Remove rows with NaN values created by shift()
df = df.dropna()

#save it and overwrite the csv file with the new columns
df.to_csv('BTCUSDT-1h.csv', index=False)
print(f"Historical data for {symbol} saved to 'BTCUSDT-1h.csv'.")


