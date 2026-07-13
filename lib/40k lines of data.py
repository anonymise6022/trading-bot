import pandas as pd
import numpy as np

symbol = 'BTC/USDT'
url = 'https://drive.google.com/uc?export=download&id=1qnX9GpiL5Ii1FEnHTIAzWnxNejWnilKp'
df = pd.read_csv(url) #reading the data from the url

# Calculate log return
df['close_log_return'] = np.log(df['close'] / df['close'].shift(1))

# Lag it from 1 to 6 candles
for lag in range(1, 7):
    df[f'close_log_return_lag_{lag}'] = df['close_log_return'].shift(lag)

# Remove rows with NaN values created by shift()
df = df.dropna()

# add the new 6 lags to the csv file

#save it to a new csv file 
df.to_csv('GOOGLE DRIVE 40K DATA.csv', index=False)

print(f"Historical data for {symbol} saved to 'GOOGLE DRIVE 40K DATA.csv'.")


