import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

def time_split(x, train_size = 0.75):
  i = int(len(x) * train_size)
  return x[:i].copy(), x[i:].copy()

btcusdt_train, btcusdt_test = time_split(btcusdt, train_size = 0.7)

# -------------------------------------------------------
# 0. REPRODUCIBILITY SETTINGS
# -------------------------------------------------------
SEED = 99

# Ensure Python’s hash-based operations are deterministic
os.environ["PYTHONHASHSEED"] = str(SEED)

# Set seeds for Python's built-in RNG, NumPy, and PyTorch
random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)          # For single-GPU setups
torch.cuda.manual_seed_all(SEED)      # For multi-GPU setups


# -------------------------------------------------------
# 1. CREATE TENSORS FROM DATAFRAME **AFTER SETTING SEEDS**
# -------------------------------------------------------

# Input features (model predictors)
features = ['close_log_return_lag_3']

# Target variable (model output)
target = 'close_log_return'

# Convert train/test splits into PyTorch tensors
X_train = torch.tensor(btcusdt_train[features].values, dtype=torch.float32)
X_test  = torch.tensor(btcusdt_test[features].values, dtype=torch.float32)

# Create target tensors and add a column dimension (N → N×1)
y_train = torch.tensor(btcusdt_train[target].values, dtype=torch.float32).unsqueeze(1)
y_test  = torch.tensor(btcusdt_test[target].values, dtype=torch.float32).unsqueeze(1)


# -------------------------------------------------------
# 2. DEFINE MODEL
# -------------------------------------------------------

# Number of input features (1 in this case)
no_features = len(features)

# Simple linear regression model: y = Wx + b
model = nn.Linear(no_features, 1)

# Huber loss (robust to outliers compared to MSE)
criterion = nn.HuberLoss()

# Stochastic Gradient Descent optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)


# -------------------------------------------------------
# 3. TRAINING LOOP (FULL-BATCH GRADIENT DESCENT)
# -------------------------------------------------------
for epoch in range(5000):

    # Clear previously stored gradients (they accumulate by default)
    optimizer.zero_grad()

    # Forward pass: compute predictions
    y_pred = model(X_train)

    # Compute loss between predictions and true values
    loss = criterion(y_pred, y_train)

    # Backpropagation: compute gradients of loss w.r.t. parameters
    loss.backward()

    # Update model parameters using the computed gradients
    optimizer.step()

    # Print loss every 500 epochs
    if epoch % 500 == 0:
        print("Epoch:", epoch, "Loss:", loss.item())


# -------------------------------------------------------
# 4. CHECK TRAINED PARAMETERS
# -------------------------------------------------------
print("Final weight:", model.weight.data)
print("Final bias:", model.bias.data)