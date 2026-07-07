import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import pandas as pd

btcusdt = pd.read_csv('historical_data.csv', parse_dates=["timestamp"], index_col='timestamp')

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
# Train one model for each lag
# -------------------------------------------------------

target = 'close_log_return'

for lag in range(1, 7):

    print(f"\n========== Training Lag {lag} ==========")

    # -------------------------------------------------------
    # 1. CREATE TENSORS FROM DATAFRAME
    # -------------------------------------------------------

    features = [f'close_log_return_lag_{lag}']

    X_train = torch.tensor(btcusdt_train[features].values, dtype=torch.float32)
    X_test  = torch.tensor(btcusdt_test[features].values, dtype=torch.float32)

    y_train = torch.tensor(btcusdt_train[target].values, dtype=torch.float32).unsqueeze(1)
    y_test  = torch.tensor(btcusdt_test[target].values, dtype=torch.float32).unsqueeze(1)

    # -------------------------------------------------------
    # 2. DEFINE MODEL
    # -------------------------------------------------------

    no_features = len(features)

    model = nn.Linear(no_features, 1)

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
            print(f"Lag {lag} | Epoch: {epoch} | Loss: {loss.item()}")

    # -------------------------------------------------------
    # 4. SAVE MODEL
    # -------------------------------------------------------

    torch.save(model.state_dict(), f"model_lag_{lag}.pth")

    # -------------------------------------------------------
    # 5. CHECK TRAINED PARAMETERS
    # -------------------------------------------------------

    print("Final weight:", model.weight.data)
    print("Final bias:", model.bias.data)
    print(f"Model saved as model_lag_{lag}.pth")