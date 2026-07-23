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

target = 'close_log_return'

use_oi = False
feature_log_cols = [
    'close_log_return_lag_1',
    'close_log_return_lag_2',
    'close_log_return_lag_3',
    'close_log_return_lag_4',
    'close_log_return_lag_5',
    'close_log_return_lag_6',
    'sma_cross',
    'sma_trend_strength',
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
model = nn.Linear(model_size, 1)

y_train = torch.tensor(btcusdt_train[target].values, dtype=torch.float32).unsqueeze(1)
y_test  = torch.tensor(btcusdt_test[target].values, dtype=torch.float32).unsqueeze(1)

    # -------------------------------------------------------
    # 2. DEFINE MODEL
    # -------------------------------------------------------

criterion = nn.HuberLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)

    # -------------------------------------------------------
    # 3. TRAINING LOOP
    # -------------------------------------------------------

for epoch in range(5000):

        optimizer.zero_grad()

        y_pred = model(X_train)

        loss = criterion(y_pred, y_train)

        loss.backward()

        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch: {epoch} | Loss: {loss.item()}")

    # -------------------------------------------------------
    # 4. SAVE MODEL
    # -------------------------------------------------------

print("Final weight:", model.weight.data)
print("Final bias:", model.bias.data)


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
    test_pred = model(X_test)
    test_loss = criterion(test_pred, y_test)

print(f"Test Loss: {test_loss.item()}")

pred_np = test_pred.numpy().flatten()
actual_np = y_test.numpy().flatten()

corr = np.corrcoef(pred_np, actual_np)[0, 1]
print("Correlation:", corr)

direction_correct = (
    (pred_np > 0) == (actual_np > 0)
).mean()

print("Directional Accuracy:", direction_correct)

print("Baseline accuracy:", (actual_np > 0).mean())