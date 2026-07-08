from training import model
import numpy as np
import pandas as pd

df = pd.read_csv('historical_data.csv')

weights = model.weight.data
print(f"weights: {weights}")
biases = model.bias.data
print(f"biases: {biases}")

lags = df.iloc[:, -6:].values

for lag, weights in zip(lags, weights):
    pred += lag * weights
    print(f"Prediction for{pred}")