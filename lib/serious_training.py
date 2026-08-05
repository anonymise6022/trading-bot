import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# =======================================================
# 1. LOAD DATA
# =======================================================

user_input = input("Which dataset do you want to use? 40k or latest?: ").strip().lower()

if user_input == "latest":
    btcusdt = pd.read_csv(
        "historical_data.csv",
        parse_dates=["timestamp"],
        index_col="timestamp"
    )
else:
    btcusdt = pd.read_csv(
        "BTCUSDT-1hNEW.csv",
        parse_dates=["open_time"],
        index_col="open_time"
    )


# =======================================================
# 2. TIME-BASED TRAIN / TEST SPLIT
# =======================================================

def time_split(data, train_size=0.7):
    split_index = int(len(data) * train_size)
    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()
    return train_data, test_data

btcusdt_train, btcusdt_test = time_split(btcusdt, train_size=0.7)

print("\n========== DATA INFORMATION ==========")
print("Total rows:", len(btcusdt))
print("Training rows:", len(btcusdt_train))
print("Testing rows:", len(btcusdt_test))

# =======================================================
# 3. REPRODUCIBILITY SETTINGS
# =======================================================

SEED = 99

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# =======================================================
# 4. TARGET AND FEATURES
# =======================================================

target = "signal"

USE_OI = False

# Price-based features
feature_log_cols = [
    "relative_volume_20",
    "atr_percent",
    "volatility_20",
]

# Open-interest features
feature_oi_cols = [
    "oi_momentum_lag_1",
    "oi_momentum_lag_2",
    "oi_momentum_lag_3",
    "oi_momentum_lag_4",
    "oi_momentum_lag_5"
]

# Select the features
if USE_OI:
    feature_cols = feature_log_cols + feature_oi_cols
else:
    feature_cols = feature_log_cols

print("\n========== FEATURES ==========")
for feature in feature_cols:
    print(feature)

# =======================================================
# 5. CHECK FOR MISSING DATA
# =======================================================

required_columns = feature_cols + [target]
missing_columns = [column for column in required_columns if column not in btcusdt.columns]

if len(missing_columns) > 0:
    raise ValueError(f"\nMissing columns in the dataset:\n{missing_columns}")

if btcusdt[required_columns].isna().sum().sum() > 0:
    print("\nWarning: Missing values were found.")
    print(btcusdt[required_columns].isna().sum())
    print("\nRemoving rows containing missing values...")

    btcusdt = btcusdt.dropna(subset=required_columns)
    btcusdt_train, btcusdt_test = time_split(btcusdt, train_size=0.7)

# =======================================================
# 6. SCALE FEATURES
# =======================================================

scaler = StandardScaler()

# Fit the scaler ONLY using training data
X_train_scaled = scaler.fit_transform(btcusdt_train[feature_cols])

# Apply the same training scaler to testing data
X_test_scaled = scaler.transform(btcusdt_test[feature_cols])

# Convert feature data to PyTorch tensors
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)

# Convert target values to PyTorch tensors
y_train = torch.tensor(btcusdt_train[target].values, dtype=torch.long)
y_test = torch.tensor(btcusdt_test[target].values, dtype=torch.long)

# =======================================================
# 7. CREATE THE NEURAL NETWORK
# =======================================================

model_size = len(feature_cols)

model = nn.Sequential(
    nn.Linear(model_size, 16),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(16, 3)
)

print("\n========== TRAINING MODEL ==========")
print("Number of input features:", model_size)
print("\nModel architecture:")
print(model)

# =======================================================
# 8. CLASS WEIGHTS
# =======================================================

class_counts = torch.bincount(y_train, minlength=3).float()
class_weights = class_counts.sum() / (class_counts * len(class_counts))

print("\nClass counts (train):", class_counts.tolist())
print("Class weights:", class_weights.tolist())

# =======================================================
# 9. LOSS FUNCTION AND OPTIMIZER
# =======================================================

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =======================================================
# 10. TRAIN THE MODEL
# =======================================================

EPOCHS = 1000

for epoch in range(EPOCHS):
    # Put the model into training mode
    model.train()

    # Remove old gradients
    optimizer.zero_grad()

    # Get the model's raw predictions
    logits = model(X_train)

    # Calculate the loss
    loss = criterion(logits, y_train)

    # Calculate gradients
    loss.backward()

    # Update the model
    optimizer.step()

    # Print progress every 100 epochs
    if epoch % 100 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch: {epoch} | Loss: {loss.item():.6f}")

# =======================================================
# 11. SAVE THE MODEL
# =======================================================

save_data = {
    "model_state_dict": model.state_dict(),
    "features": feature_cols,
    "model_size": model_size,
    "scaler_mean": scaler.mean_,
    "scaler_scale": scaler.scale_
}

if USE_OI:
    model_filename = "model_price_oi.pth"
else:
    model_filename = "model_price_only.pth"

torch.save(save_data, model_filename)

print(f"\nModel saved as {model_filename}")

# =======================================================
# 12. TEST THE MODEL
# =======================================================

model.eval()

with torch.no_grad():
    # Get raw test predictions
    test_logits = model(X_test)

    # Calculate test loss
    test_loss = criterion(test_logits, y_test)

    # Convert logits into class predictions
    predicted_class = torch.argmax(test_logits, dim=1)

    # Calculate classification accuracy
    accuracy = (predicted_class == y_test).float().mean()

print("\n========== TEST RESULTS ==========")
print(f"Test Loss: {test_loss.item():.6f}")
print(f"Classification Accuracy: {accuracy.item():.4%}")

# =======================================================
# 13. BASELINE ACCURACY
# =======================================================

baseline = y_test.bincount().float().max() / len(y_test)

print(f"Baseline Accuracy: {baseline.item():.4%}")
print(f"Improvement over baseline: {(accuracy - baseline).item():.4%}")

# =======================================================
# 14. CONVERT RESULTS TO NUMPY
# =======================================================

pred_np = predicted_class.cpu().numpy().flatten()
actual_np = y_test.cpu().numpy().flatten()

# =======================================================
# 15. SIGNAL PRECISION
# =======================================================

long_predictions = pred_np == 2
short_predictions = pred_np == 0
neutral_predictions = pred_np == 1

print("\n========== SIGNAL PRECISION ==========")

if long_predictions.sum() > 0:
    long_precision = (actual_np[long_predictions] == 2).mean()
    print(f"Long precision: {long_precision:.4%}")

if short_predictions.sum() > 0:
    short_precision = (actual_np[short_predictions] == 0).mean()
    print(f"Short precision: {short_precision:.4%}")

if neutral_predictions.sum() > 0:
    neutral_precision = (actual_np[neutral_predictions] == 1).mean()
    print(f"Neutral precision: {neutral_precision:.4%}")

print("\nLong signals:", long_predictions.sum())
print("Short signals:", short_predictions.sum())
print("Neutral signals:", neutral_predictions.sum())

# =======================================================
# 16. ACTUAL SIGNAL DISTRIBUTION
# =======================================================

print("\n========== FULL DATA SIGNAL COUNTS ==========")
print(btcusdt[target].value_counts().sort_index())

print("\n========== TEST DATA SIGNAL COUNTS ==========")
print(btcusdt_test[target].value_counts().sort_index())

# =======================================================
# 17. CONFUSION MATRIX
# =======================================================

confusion = np.zeros((3, 3), dtype=int)

for actual_value, predicted_value in zip(actual_np, pred_np):
    confusion[actual_value, predicted_value] += 1

print("\n========== CONFUSION MATRIX ==========")
print("Rows = actual")
print("Columns = predicted")
print("0 = short")
print("1 = neutral")
print("2 = long\n")
print(confusion)

# =======================================================
# 18. MODEL PROBABILITIES
# =======================================================

probabilities = torch.softmax(test_logits, dim=1)

print("\n========== FIRST 10 MODEL PROBABILITIES ==========")
print("Order: [short, neutral, long]\n")

for i in range(min(10, len(probabilities))):
    print(f"Test row {i}: ", probabilities[i].cpu().numpy())

# =======================================================
# 19. INDIVIDUAL FEATURE ANALYSIS
# =======================================================

print("\n========== INDIVIDUAL FEATURE ANALYSIS ==========")
print("\nEach feature is divided into:")
print("LOW = bottom 33%")
print("MEDIUM = middle 33%")
print("HIGH = top 33%")
print("\nCounts and percentages show")
print("the ACTUAL future signal.")
print("Order: [short, neutral, long]")

for feature in feature_cols:
    print("\n" + "=" * 60)
    print(f"FEATURE: {feature.upper()}")
    print("=" * 60)

    # Find the lower and upper thirds
    low_cutoff = btcusdt_test[feature].quantile(1 / 3)
    high_cutoff = btcusdt_test[feature].quantile(2 / 3)

    # Create low / medium / high groups
    low_mask = btcusdt_test[feature] <= low_cutoff
    medium_mask = (btcusdt_test[feature] > low_cutoff) & (btcusdt_test[feature] < high_cutoff)
    high_mask = btcusdt_test[feature] >= high_cutoff

    # Store each group
    feature_groups = {
        "LOW": low_mask,
        "MEDIUM": medium_mask,
        "HIGH": high_mask
    }

    # Analyze every group
    for group_name, group_mask in feature_groups.items():
        group_actual = actual_np[group_mask.to_numpy()]
        group_counts = np.bincount(group_actual, minlength=3)
        total_group_rows = group_counts.sum()

        if total_group_rows == 0:
            print(f"\n{group_name}:")
            print("No rows")
            continue

        group_percentages = group_counts / total_group_rows * 100

        print(f"\n{group_name}")
        print(
            f"Feature range: {btcusdt_test.loc[group_mask, feature].min():.6f} "
            f"to {btcusdt_test.loc[group_mask, feature].max():.6f}"
        )
        print(f"Total rows: {total_group_rows}")
        print("Counts:")
        print(f"  Short:   {group_counts[0]}")
        print(f"  Neutral: {group_counts[1]}")
        print(f"  Long:    {group_counts[2]}")
        print("Percentages:")
        print(f"  Short:   {group_percentages[0]:.2f}%")
        print(f"  Neutral: {group_percentages[1]:.2f}%")
        print(f"  Long:    {group_percentages[2]:.2f}%")

# =======================================================
# 20. FINISHED
# =======================================================

print("\n========== ANALYSIS COMPLETE ==========")