import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import copy
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

# =======================================================
# 1. LOAD DATA
# =======================================================

user_input = input("Which dataset do you want to use? 40k or latest?: ").strip().lower()

if user_input == "latest":
    btcusdt = pd.read_csv(
        #"historical_data.csv",
        "Jacob_data_daily.csv",
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
# 2. TARGET AND BASE FEATURES
# =======================================================

target = "signal"

USE_OI = False

# Price-based features
feature_log_cols = [
    "relative_volume_20",
    "atr_percent",
    "volatility_20",
    "momentum_12",
    "candle_strength",
    "ema_cross",
]

# Open-interest features
feature_oi_cols = [
    "oi_momentum_lag_1",
    "oi_momentum_lag_2",
    "oi_momentum_lag_3",
    "oi_momentum_lag_4",
    "oi_momentum_lag_5"
]

if USE_OI:
    base_feature_cols = feature_log_cols + feature_oi_cols
else:
    base_feature_cols = feature_log_cols

# =======================================================
# 3. ENGINEERED INTERACTION FEATURES
# =======================================================
# The pairwise combo analysis (section 20 below) showed real, repeatable
# edges when two features are HIGH/LOW together (e.g. atr_percent HIGH +
# momentum_12 LOW -> 48% long vs a 40.5% base rate). A single hidden layer
# can *theoretically* learn these crosses, but in practice it wasn't --
# the model's output probabilities were sitting close to uniform after
# training. Feeding the products directly as input features removes that
# burden so the model spends its capacity ranking, not discovering.
#
# Add/remove pairs here as you find new edges in the pairwise analysis.
ADD_INTERACTIONS = True

interaction_pairs = [
    ("atr_percent", "momentum_12"),
    ("atr_percent", "volatility_20"),
    ("relative_volume_20", "volatility_20"),
]

interaction_cols = []

if ADD_INTERACTIONS:
    for feat_a, feat_b in interaction_pairs:
        if feat_a in btcusdt.columns and feat_b in btcusdt.columns:
            col_name = f"{feat_a}_x_{feat_b}"
            btcusdt[col_name] = btcusdt[feat_a] * btcusdt[feat_b]
            interaction_cols.append(col_name)

feature_cols = base_feature_cols + interaction_cols

print("\n========== FEATURES ==========")
for feature in feature_cols:
    tag = " (interaction)" if feature in interaction_cols else ""
    print(feature + tag)

# =======================================================
# 4. CHECK FOR MISSING DATA
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

# =======================================================
# 5. TIME-BASED TRAIN / VAL / TEST SPLIT
# =======================================================
# Three-way split, all time-ordered (no shuffling -- this is a time series).
# TRAIN_FRAC trains the weights, VAL_FRAC is used only for early stopping /
# model selection, TEST is held out completely until final evaluation.

TRAIN_FRAC = 0.60
VAL_FRAC = 0.15
# remaining ~0.25 goes to test

def time_split_3way(data, train_frac, val_frac):
    n = len(data)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train_data = data.iloc[:train_end].copy()
    val_data = data.iloc[train_end:val_end].copy()
    test_data = data.iloc[val_end:].copy()
    return train_data, val_data, test_data

btcusdt_train, btcusdt_val, btcusdt_test = time_split_3way(btcusdt, TRAIN_FRAC, VAL_FRAC)

print("\n========== DATA INFORMATION ==========")
print("Total rows:", len(btcusdt))
print("Training rows:", len(btcusdt_train))
print("Validation rows:", len(btcusdt_val))
print("Testing rows:", len(btcusdt_test))

# =======================================================
# 6. REPRODUCIBILITY SETTINGS
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
# 7. SCALE FEATURES
# =======================================================

scaler = StandardScaler()

# Fit the scaler ONLY on the training split (not val, not test)
X_train_scaled = scaler.fit_transform(btcusdt_train[feature_cols])
X_val_scaled = scaler.transform(btcusdt_val[feature_cols])
X_test_scaled = scaler.transform(btcusdt_test[feature_cols])

X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val = torch.tensor(X_val_scaled, dtype=torch.float32)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)

y_train = torch.tensor(btcusdt_train[target].values, dtype=torch.long)
y_val = torch.tensor(btcusdt_val[target].values, dtype=torch.long)
y_test = torch.tensor(btcusdt_test[target].values, dtype=torch.long)

# =======================================================
# 8. CREATE THE NEURAL NETWORK
# =======================================================
# Slightly deeper than before (two hidden layers + BatchNorm) so the model
# has enough capacity to actually use the interaction features instead of
# collapsing to near-uniform outputs.

model_size = len(feature_cols)

model = nn.Sequential(
    nn.Linear(model_size, 32),
    nn.BatchNorm1d(32),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(32, 16),
    nn.BatchNorm1d(16),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(16, 3)
)

print("\n========== TRAINING MODEL ==========")
print("Number of input features:", model_size)
print("\nModel architecture:")
print(model)

# =======================================================
# 9. CLASS WEIGHTS (softened)
# =======================================================
# Raw inverse-frequency weighting was pushing the model to massively
# over-predict the minority "neutral" class (7,902 neutral predictions vs
# only 2,924 actual neutral rows), which is why accuracy fell BELOW the
# naive baseline. Taking the square root of the inverse-frequency weight
# keeps some correction for imbalance without letting it dominate the loss.

class_counts = torch.bincount(y_train, minlength=3).float()
raw_class_weights = class_counts.sum() / (class_counts * len(class_counts))
class_weights = torch.sqrt(raw_class_weights)

print("\nClass counts (train):", class_counts.tolist())
print("Raw inverse-frequency weights:", raw_class_weights.tolist())
print("Softened (sqrt) weights used:", class_weights.tolist())

# =======================================================
# 10. LOSS FUNCTION, OPTIMIZER, SCHEDULER
# =======================================================

criterion = nn.CrossEntropyLoss(weight=class_weights)

# Small weight_decay for extra regularization now that the model is bigger
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Reduce LR when validation loss stalls
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=30
)

# =======================================================
# 11. TRAIN THE MODEL (with validation + early stopping)
# =======================================================

MAX_EPOCHS = 3000
PATIENCE = 100          # stop if val macro-F1 doesn't improve for this many epochs
EVAL_EVERY = 10         # how often to run validation + print progress

best_val_f1 = -1.0
best_state = copy.deepcopy(model.state_dict())
epochs_without_improvement = 0

for epoch in range(MAX_EPOCHS):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train)
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()

    if epoch % EVAL_EVERY == 0 or epoch == MAX_EPOCHS - 1:
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val)
            val_pred = torch.argmax(val_logits, dim=1)
            val_f1 = f1_score(y_val.numpy(), val_pred.numpy(), average="macro")

        scheduler.step(val_loss)

        print(f"Epoch: {epoch:4d} | Train Loss: {loss.item():.4f} | "
              f"Val Loss: {val_loss.item():.4f} | Val Macro-F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += EVAL_EVERY

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no val Macro-F1 improvement for {PATIENCE} epochs).")
            break

# Restore the best checkpoint seen during training (by validation macro-F1)
model.load_state_dict(best_state)
print(f"\nBest validation Macro-F1: {best_val_f1:.4f}")

# =======================================================
# 12. SAVE THE MODEL
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
# 13. TEST THE MODEL
# =======================================================

model.eval()

with torch.no_grad():
    test_logits = model(X_test)
    test_loss = criterion(test_logits, y_test)
    predicted_class = torch.argmax(test_logits, dim=1)
    accuracy = (predicted_class == y_test).float().mean()
    test_macro_f1 = f1_score(y_test.numpy(), predicted_class.numpy(), average="macro")

print("\n========== TEST RESULTS ==========")
print(f"Test Loss: {test_loss.item():.6f}")
print(f"Classification Accuracy: {accuracy.item():.4%}")
print(f"Test Macro-F1: {test_macro_f1:.4f}")

# =======================================================
# 14. BASELINE ACCURACY
# =======================================================

baseline = y_test.bincount().float().max() / len(y_test)

print(f"Baseline Accuracy: {baseline.item():.4%}")
print(f"Improvement over baseline: {(accuracy - baseline).item():.4%}")

# =======================================================
# 15. CONVERT RESULTS TO NUMPY
# =======================================================

pred_np = predicted_class.cpu().numpy().flatten()
actual_np = y_test.cpu().numpy().flatten()

# =======================================================
# 16. SIGNAL PRECISION
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
# 17. ACTUAL SIGNAL DISTRIBUTION
# =======================================================

print("\n========== FULL DATA SIGNAL COUNTS ==========")
print(btcusdt[target].value_counts().sort_index())

print("\n========== TEST DATA SIGNAL COUNTS ==========")
print(btcusdt_test[target].value_counts().sort_index())

# =======================================================
# 18. CONFUSION MATRIX
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
# 19. MODEL PROBABILITIES
# =======================================================

probabilities = torch.softmax(test_logits, dim=1)

print("\n========== FIRST 10 MODEL PROBABILITIES ==========")
print("Order: [short, neutral, long]\n")

for i in range(min(10, len(probabilities))):
    print(f"Test row {i}: ", probabilities[i].cpu().numpy())

# =======================================================
# 20. PAIRWISE FEATURE COMBINATION ANALYSIS
# =======================================================

MIN_GROUP_SIZE = 15
INCLUDE_MEDIUM = False
CLASS_NAMES = ["Short", "Neutral", "Long"]


def get_terciles(df, feature, include_medium=False):
    """Return {label: boolean_mask} splitting `feature` into thirds."""
    low_cutoff = df[feature].quantile(1 / 3)
    high_cutoff = df[feature].quantile(2 / 3)

    groups = {
        "LOW": df[feature] <= low_cutoff,
        "HIGH": df[feature] >= high_cutoff,
    }

    if include_medium:
        groups["MEDIUM"] = (df[feature] > low_cutoff) & (df[feature] < high_cutoff)

    return groups


print("\n\n========== PAIRWISE FEATURE ANALYSIS ==========")
print("\nEach feature split into LOW (bottom 33%) / HIGH (top 33%)"
      + (" / MEDIUM (middle 33%)" if INCLUDE_MEDIUM else ", middle third skipped"))
print("Percentages show the ACTUAL future signal within each combo.")
print(f"Combinations with fewer than {MIN_GROUP_SIZE} test rows are hidden.")

combo_results = []

# Only run this over the BASE features -- running it over engineered
# interaction terms too would just duplicate what the base pairs already show.
for feat_a, feat_b in combinations(base_feature_cols, 2):
    groups_a = get_terciles(btcusdt_test, feat_a, INCLUDE_MEDIUM)
    groups_b = get_terciles(btcusdt_test, feat_b, INCLUDE_MEDIUM)

    print("\n" + "=" * 60)
    print(f"{feat_a.upper()}  +  {feat_b.upper()}")
    print("=" * 60)

    any_printed = False

    for label_a, mask_a in groups_a.items():
        for label_b, mask_b in groups_b.items():
            combo_mask = (mask_a & mask_b).to_numpy()
            group_actual = actual_np[combo_mask]
            total = len(group_actual)

            if total < MIN_GROUP_SIZE:
                continue

            counts = np.bincount(group_actual, minlength=3)
            pct = counts / total * 100

            dominant_idx = int(counts.argmax())
            dominant_name = CLASS_NAMES[dominant_idx]
            dominant_pct = pct[dominant_idx]

            print(f"\n{feat_a} {label_a} + {feat_b} {label_b}  (n={total})")
            print(f"  {dominant_name} {dominant_pct:.1f}%   "
                  f"[Short {pct[0]:.1f}% / Neutral {pct[1]:.1f}% / Long {pct[2]:.1f}%]")

            any_printed = True

            combo_results.append({
                "combo": f"{feat_a} {label_a} + {feat_b} {label_b}",
                "n": total,
                "dominant": dominant_name,
                "dominant_pct": dominant_pct,
            })

    if not any_printed:
        print("\n  (no combination met the minimum group size)")

if combo_results:
    print("\n\n========== TOP COMBINATIONS BY SIGNAL STRENGTH ==========")
    top_combos = sorted(combo_results, key=lambda r: r["dominant_pct"], reverse=True)[:15]
    for r in top_combos:
        print(f"{r['combo']:<55} {r['dominant']} {r['dominant_pct']:.1f}%  (n={r['n']})")

# =======================================================
# 21. FINISHED
# =======================================================

print("\n========== ANALYSIS COMPLETE ==========")