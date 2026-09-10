import numpy as np
import pandas as pd
import torch
from itertools import combinations
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

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
# This model only ever sees rows the regime model would have flagged as
# "tradeable" (i.e. not neutral). Its only job is short vs. long.

target = "signal"

FILTER_MODE = "ground_truth"     # "ground_truth" or "regime_model"
REGIME_MODEL_PATH = "model_xgb_price_only.pth"
REGIME_NEUTRAL_THRESHOLD = 0.4

feature_log_cols = [
    "relative_volume_20",
    "atr_percent",
    "volatility_20",
    "momentum_12",
    "candle_strength",
    "ema_cross",
]

feature_oi_cols = [
    "oi_momentum_lag_1",
    "oi_momentum_lag_2",
    "oi_momentum_lag_3",
    "oi_momentum_lag_4",
    "oi_momentum_lag_5"
]

USE_OI = False

# Placeholder slot for new directional features once validated via the
# pairwise analysis at the bottom of this script (RSI, MACD hist, MA
# distance, higher-timeframe trend, etc.).
feature_directional_cols = []

if USE_OI:
    base_feature_cols = feature_log_cols + feature_oi_cols + feature_directional_cols
else:
    base_feature_cols = feature_log_cols + feature_directional_cols

# =======================================================
# 3. ENGINEERED INTERACTION FEATURES
# =======================================================

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
# 5. FILTER OUT NEUTRAL ROWS
# =======================================================

if FILTER_MODE == "ground_truth":
    print("\nFilter mode: ground_truth (dropping rows where actual signal == neutral)")
    direction_df = btcusdt[btcusdt[target] != 1].copy()

elif FILTER_MODE == "regime_model":
    print(f"\nFilter mode: regime_model (loading {REGIME_MODEL_PATH})")

    # Reload the regime model from its .pth checkpoint: pull the raw
    # booster bytes back out and hand them to a fresh XGBClassifier.
    regime_checkpoint = torch.load(REGIME_MODEL_PATH, weights_only=False)
    regime_model = XGBClassifier()
    regime_model.load_model(bytearray(regime_checkpoint["booster_bytes"]))
    regime_feature_cols = regime_checkpoint["features"]

    # NOTE: uses the regime model's OWN feature list (saved in its
    # checkpoint), not this script's feature_cols, since the two models
    # are allowed to diverge on inputs.
    regime_probs = regime_model.predict_proba(btcusdt[regime_feature_cols].values)
    p_neutral = regime_probs[:, 1]

    tradeable_mask = p_neutral < REGIME_NEUTRAL_THRESHOLD
    direction_df = btcusdt[tradeable_mask].copy()
    print(f"Regime model flagged {tradeable_mask.sum()} / {len(btcusdt)} rows as tradeable "
          f"(P(neutral) < {REGIME_NEUTRAL_THRESHOLD})")

else:
    raise ValueError(f"Unknown FILTER_MODE: {FILTER_MODE}")

direction_df["direction_label"] = (direction_df[target] == 2).astype(int)

print("\nDirection dataset size after filtering:", len(direction_df))
print("Class balance (0=short, 1=long):")
print(direction_df["direction_label"].value_counts().sort_index())

# =======================================================
# 6. TIME-BASED TRAIN / VAL / TEST SPLIT
# =======================================================

TRAIN_FRAC = 0.60
VAL_FRAC = 0.15


def time_split_3way(data, train_frac, val_frac):
    n = len(data)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train_data = data.iloc[:train_end].copy()
    val_data = data.iloc[train_end:val_end].copy()
    test_data = data.iloc[val_end:].copy()
    return train_data, val_data, test_data


direction_train, direction_val, direction_test = time_split_3way(direction_df, TRAIN_FRAC, VAL_FRAC)

print("\n========== DATA INFORMATION ==========")
print("Total tradeable rows:", len(direction_df))
print("Training rows:", len(direction_train))
print("Validation rows:", len(direction_val))
print("Testing rows:", len(direction_test))

# =======================================================
# 7. REPRODUCIBILITY SETTINGS
# =======================================================

SEED = 99
np.random.seed(SEED)

# =======================================================
# 8. BUILD FEATURE / LABEL ARRAYS
# =======================================================

X_train = direction_train[feature_cols].values
X_val = direction_val[feature_cols].values
X_test = direction_test[feature_cols].values

y_train = direction_train["direction_label"].values.astype(int)
y_val = direction_val["direction_label"].values.astype(int)
y_test = direction_test["direction_label"].values.astype(int)

# =======================================================
# 9. CLASS WEIGHTS
# =======================================================

train_counts = np.bincount(y_train, minlength=2).astype(float)
print("\nClass counts (train, 0=short / 1=long):", train_counts.tolist())
print("Ratio (long / short):", train_counts[1] / train_counts[0])

# sample_weight_train = compute_sample_weight("balanced", y_train)  # enable if skewed

# =======================================================
# 10. TRAIN THE MODEL
# =======================================================

model = XGBClassifier(
    objective="binary:logistic",
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="logloss",
    early_stopping_rounds=30,
    random_state=SEED,
)

print("\n========== TRAINING MODEL ==========")
print("Number of input features:", len(feature_cols))

model.fit(
    X_train, y_train,
    # sample_weight=sample_weight_train,  # enable if class counts above are skewed
    eval_set=[(X_val, y_val)],
    verbose=50,
)

print(f"\nBest iteration: {model.best_iteration}")

# =======================================================
# 11. FEATURE IMPORTANCE
# =======================================================

importances = model.feature_importances_
importance_pairs = sorted(zip(feature_cols, importances), key=lambda p: p[1], reverse=True)

print("\n========== FEATURE IMPORTANCE ==========")
for name, score in importance_pairs:
    tag = " (interaction)" if name in interaction_cols else ""
    print(f"{name:<35} {score:.4f}{tag}")

# =======================================================
# 12. SAVE THE MODEL (single .pth file via torch.save)
# =======================================================
# Same approach as the regime script: grab the booster's raw bytes and
# pickle them (via torch.save) alongside all feature metadata into one
# .pth file, so there's a single artifact per model instead of a
# model.json + model_features.json pair.

model_filename = "model_xgb_direction.pth"

booster_bytes = model.get_booster().save_raw(raw_format="json")

checkpoint = {
    "booster_bytes": bytes(booster_bytes),
    "xgb_params": model.get_params(),
    "features": feature_cols,
    "base_features": base_feature_cols,
    "interaction_features": interaction_cols,
    "interaction_pairs": interaction_pairs,
    "filter_mode": FILTER_MODE,
    "regime_neutral_threshold": REGIME_NEUTRAL_THRESHOLD if FILTER_MODE == "regime_model" else None,
    "best_iteration": model.best_iteration,
    "seed": SEED,
    "target": target,
}

torch.save(checkpoint, model_filename)

print(f"\nModel + feature metadata saved as {model_filename}")

# --- how to reload later ---
# checkpoint = torch.load("model_xgb_direction.pth", weights_only=False)
# model = XGBClassifier()
# model.load_model(bytearray(checkpoint["booster_bytes"]))
# feature_cols = checkpoint["features"]

# =======================================================
# 13. TEST THE MODEL
# =======================================================

predicted_class = model.predict(X_test)
probabilities = model.predict_proba(X_test)  # [:, 0] = P(short), [:, 1] = P(long)

accuracy = (predicted_class == y_test).mean()
test_f1 = f1_score(y_test, predicted_class)
test_auc = roc_auc_score(y_test, probabilities[:, 1])

print("\n========== TEST RESULTS ==========")
print(f"Classification Accuracy: {accuracy:.4%}")
print(f"Test F1 (long as positive class): {test_f1:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# =======================================================
# 14. CONDITIONAL BASELINE
# =======================================================

baseline = np.bincount(y_test).max() / len(y_test)

print(f"Conditional Baseline Accuracy (majority class, non-neutral only): {baseline:.4%}")
print(f"Improvement over baseline: {(accuracy - baseline):.4%}")

# =======================================================
# 15. SIGNAL PRECISION
# =======================================================

pred_np = predicted_class.flatten()
actual_np = y_test.flatten()

long_predictions = pred_np == 1
short_predictions = pred_np == 0

print("\n========== SIGNAL PRECISION ==========")

if long_predictions.sum() > 0:
    long_precision = (actual_np[long_predictions] == 1).mean()
    print(f"Long precision: {long_precision:.4%}")

if short_predictions.sum() > 0:
    short_precision = (actual_np[short_predictions] == 0).mean()
    print(f"Short precision: {short_precision:.4%}")

print("\nLong signals:", long_predictions.sum())
print("Short signals:", short_predictions.sum())

# =======================================================
# 16. ACTUAL DIRECTION DISTRIBUTION
# =======================================================

print("\n========== FULL FILTERED DATA DIRECTION COUNTS ==========")
print(direction_df["direction_label"].value_counts().sort_index())

print("\n========== TEST DATA DIRECTION COUNTS ==========")
print(direction_test["direction_label"].value_counts().sort_index())

# =======================================================
# 17. CONFUSION MATRIX
# =======================================================

confusion = np.zeros((2, 2), dtype=int)

for actual_value, predicted_value in zip(actual_np, pred_np):
    confusion[actual_value, predicted_value] += 1

print("\n========== CONFUSION MATRIX ==========")
print("Rows = actual")
print("Columns = predicted")
print("0 = short")
print("1 = long\n")
print(confusion)

# =======================================================
# 18. MODEL PROBABILITIES
# =======================================================

print("\n========== FIRST 10 MODEL PROBABILITIES ==========")
print("Order: [short, long]\n")

for i in range(min(10, len(probabilities))):
    print(f"Test row {i}: ", probabilities[i])

# =======================================================
# 19. PAIRWISE FEATURE COMBINATION ANALYSIS (DIRECTION VERSION)
# =======================================================

MIN_GROUP_SIZE = 15
CLASS_NAMES = ["Short", "Long"]


def get_terciles(df, feature):
    low_cutoff = df[feature].quantile(1 / 3)
    high_cutoff = df[feature].quantile(2 / 3)
    return {
        "LOW": df[feature] <= low_cutoff,
        "HIGH": df[feature] >= high_cutoff,
    }


print("\n\n========== PAIRWISE FEATURE ANALYSIS (DIRECTION) ==========")
print("\nEach feature split into LOW (bottom 33%) / HIGH (top 33%), middle third skipped")
print("Percentages show the ACTUAL direction within each combo (non-neutral rows only).")
print(f"Combinations with fewer than {MIN_GROUP_SIZE} test rows are hidden.")

combo_results = []

for feat_a, feat_b in combinations(base_feature_cols, 2):
    groups_a = get_terciles(direction_test, feat_a)
    groups_b = get_terciles(direction_test, feat_b)

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

            counts = np.bincount(group_actual, minlength=2)
            pct = counts / total * 100

            dominant_idx = int(counts.argmax())
            dominant_name = CLASS_NAMES[dominant_idx]
            dominant_pct = pct[dominant_idx]

            print(f"\n{feat_a} {label_a} + {feat_b} {label_b}  (n={total})")
            print(f"  {dominant_name} {dominant_pct:.1f}%   "
                  f"[Short {pct[0]:.1f}% / Long {pct[1]:.1f}%]")

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
# 20. FINISHED
# =======================================================

print("\n========== ANALYSIS COMPLETE ==========")