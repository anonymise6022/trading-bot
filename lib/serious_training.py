import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ask which data to use: old static one or new latest 1000 candles
user_input = input("Which dataset do you want to use? 40k or latest?: ")

# if user wants to use the latest 1000 candles dataset
if user_input == "latest":
    btcusdt = pd.read_csv('historical_data.csv', parse_dates=["timestamp"], index_col='timestamp')

# if user wants to use the old static dataset
else:
     btcusdt = pd.read_csv('BTCUSDT-1hNEW.csv', parse_dates=["open_time"], index_col='open_time')

df = pd.DataFrame()

def time_split(x, train_size = 0.75):
    i = int(len(x) * train_size)
    return x[:i].copy(), x[i:].copy()

btcusdt_train, btcusdt_test = time_split(btcusdt, train_size = 0.7)

# -------------------------------------------------------
# 0. REPRODUCIBILITY SETTINGS
# -------------------------------------------------------
SEED = 99

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -------------------------------------------------------
# Train one model for each lag and oi
# -------------------------------------------------------

target = "signal"

use_oi = False
feature_log_cols = [
    "ema_trend_strength",
    "ema_distance",
    "momentum_12",
    "rsi_normalized",
    "relative_volume_20",
    "volume_change",
    "atr_percent",
    "volatility_20",
    "ema_cross",
    "candle_strength",
    "trend_regime"
]

feature_oi_cols = [
    'oi_momentum_lag_1',
    'oi_momentum_lag_2',
    'oi_momentum_lag_3',
    'oi_momentum_lag_4',
    'oi_momentum_lag_5',
]

#check if user wants to use open interest features
if use_oi == True:
    feature_cols = feature_log_cols + feature_oi_cols
else:
    feature_cols = feature_log_cols

# 2. Extract them from your train/test DataFrames as a NumPy array
X_train_np = btcusdt_train[feature_cols].values
X_test_np  = btcusdt_test[feature_cols].values

print(f"\n========== Training model  ==========")

# -------------------------------------------------------
# 1. CREATE TENSORS FROM DATAFRAME
# -------------------------------------------------------

scaler = StandardScaler()

# fit ONLY on training data
X_train_scaled = scaler.fit_transform(btcusdt_train[feature_cols])

# use the same scaler on test data
X_test_scaled = scaler.transform(btcusdt_test[feature_cols])

# convert to tensors
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)

model_size = len(feature_cols)
model = nn.Sequential(
    nn.Linear(11,16),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(16,3)
)

y_train = torch.tensor(btcusdt_train[target].values,dtype=torch.long)
y_test = torch.tensor(btcusdt_test[target].values,dtype=torch.long)

    # -------------------------------------------------------
    # 2. DEFINE MODEL
    # -------------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

    # -------------------------------------------------------
    # 3. TRAINING LOOP
    # -------------------------------------------------------

for epoch in range(1000):

        optimizer.zero_grad()

        logits = model(X_train)

        loss = criterion(logits, y_train)

        loss.backward()

        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch: {epoch} | Loss: {loss.item()}")

    # -------------------------------------------------------
    # 4. SAVE MODEL
    # -------------------------------------------------------

for name, param in model.named_parameters():
    print(name, param.data)


if use_oi == True:
    torch.save({
        'model_state_dict': model.state_dict(),
        'features': feature_cols,
        'model_size': model_size
    }, "model_price_oi.pth")
    print("Model saved as model_price_oi.pth")
else:
    torch.save({
        'model_state_dict': model.state_dict(),
        'features': feature_cols,
        'model_size': model_size
    }, "model_price_only.pth")
    print("Model saved as model_price_only.pth")

    # -------------------------------------------------------
    # 5. CHECK TRAINED PARAMETERS
    # -------------------------------------------------------

model.eval()

with torch.no_grad():

    # Raw model output
    test_logits = model(X_test)

    # Classification loss
    test_loss = criterion(test_logits,y_test)

    # Convert logits into probabilities
    predicted_class = torch.argmax(test_logits, dim=1)

    accuracy = (predicted_class == y_test).float().mean()

print(f"Test Loss: {test_loss.item()}")

accuracy = (predicted_class == y_test).float().mean()
print("Classification Accuracy:",accuracy.item())

baseline = y_test.bincount().float().max() / len(y_test)
print("Baseline Accuracy:", baseline.item())

pred_np = (predicted_class.cpu().numpy().flatten())

actual_np = (y_test.cpu().numpy().flatten())

# Accuracy when the model predicts UP
long_predictions = (pred_np == 2)

short_predictions = (pred_np == 0)

neutral_predictions = (pred_np == 1)

if long_predictions.sum()>0:

    long_precision = (actual_np[long_predictions] == 2).mean()

    print("Long precision:",long_precision)

if short_predictions.sum()>0:

    short_precision = (actual_np[short_predictions] == 0).mean()

    print("Short precision:",short_precision)

print("Long Signals:",long_predictions.sum())

print("Short Signals:",short_predictions.sum())

print(btcusdt["signal"].value_counts())

probabilities = torch.softmax(test_logits, dim=1)

for i in range(10):
    print(probabilities[i])