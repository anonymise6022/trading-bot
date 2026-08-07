import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

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
# NOTE: with XGBoost this section is arguably no longer load-bearing --
# trees naturally split on feature combinations (e.g. "atr_percent HIGH
# AND momentum_12 LOW") without needing the product spelled out as its
# own column. Left ADD_INTERACTIONS = True below so you can A/B it via
# feature_importances_ (see section 10) rather than assuming either way.

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
# Same 3-way, time-ordered split as the neural net version. VAL is used
# for early stopping (XGBoost watches it directly via eval_set), TEST
# stays untouched until final evaluation.

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
np.random.seed(SEED)

# =======================================================
# 7. BUILD FEATURE / LABEL ARRAYS
# =======================================================
# No StandardScaler here -- tree splits don't care about feature scale,
# so scaling would do nothing but add a moving part. Keeping raw values
# also makes feature_importances_ / SHAP easier to read later, since the
# numbers still mean what they meant in the original dataframe.

X_train = btcusdt_train[feature_cols].values
X_val = btcusdt_val[feature_cols].values
X_test = btcusdt_test[feature_cols].values

y_train = btcusdt_train[target].values.astype(int)
y_val = btcusdt_val[target].values.astype(int)
y_test = btcusdt_test[target].values.astype(int)

# =======================================================
# 8. CLASS WEIGHTS (softened, same philosophy as the NN version)
# =======================================================
# XGBoost wants a per-ROW weight array (sample_weight), not a per-class
# vector like CrossEntropyLoss took. compute_sample_weight('balanced', ...)
# gives the raw inverse-frequency weight per row; we then take sqrt to
# soften it, exactly like the NN file did, so the model doesn't overcorrect
# into over-predicting the minority "neutral" class.

class_counts = np.bincount(y_train, minlength=3).astype(float)
raw_class_weights = class_counts.sum() / (class_counts * len(class_counts))
softened_class_weights = np.sqrt(raw_class_weights)

print("\nClass counts (train):", class_counts.tolist())
print("Raw inverse-frequency weights:", raw_class_weights.tolist())
print("Softened (sqrt) weights used:", softened_class_weights.tolist())

# map the per-class weight onto each row based on its label
sample_weight_train = softened_class_weights[y_train]

# =======================================================
# 9. TRAIN THE MODEL
# =======================================================
# Trees are built sequentially -- each new tree focuses on correcting the
# mistakes of the trees before it. early_stopping_rounds plays the same
# role PATIENCE did in the NN loop: stop once the validation set stops
# improving, and keep the best iteration rather than the last one.

model = XGBClassifier(
    objective="multi:softprob",   # replaces the manual torch.softmax step
    num_class=3,
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="mlogloss",
    early_stopping_rounds=30,
    random_state=SEED,
)

print("\n========== TRAINING MODEL ==========")
print("Number of input features:", len(feature_cols))

model.fit(
    X_train, y_train,
    sample_weight=sample_weight_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)

print(f"\nBest iteration: {model.best_iteration}")

# =======================================================
# 10. FEATURE IMPORTANCE
# =======================================================
# This is the thing you can't easily get from the neural net -- a direct
# read on which features the model actually leaned on. Worth checking
# whether the hand-built interaction columns from section 3 rank highly;
# if they don't, the trees are likely finding those crosses on their own.

importances = model.feature_importances_
importance_pairs = sorted(zip(feature_cols, importances), key=lambda p: p[1], reverse=True)

print("\n========== FEATURE IMPORTANCE ==========")
for name, score in importance_pairs:
    tag = " (interaction)" if name in interaction_cols else ""
    print(f"{name:<35} {score:.4f}{tag}")

# =======================================================
# 11. SAVE THE MODEL
# =======================================================
# XGBoost's own save_model handles the tree structure; feature_cols has
# to be saved alongside it separately since it's not part of that file.

if USE_OI:
    model_filename = "model_xgb_price_oi.json"
else:
    model_filename = "model_xgb_price_only.json"

model.save_model(model_filename)

feature_meta = {
    "features": feature_cols,
    "base_features": base_feature_cols,
    "interaction_features": interaction_cols,
    "interaction_pairs": interaction_pairs,
}
pd.Series(feature_meta, dtype=object).to_json(model_filename.replace(".json", "_features.json"))

print(f"\nModel saved as {model_filename}")
print(f"Feature metadata saved as {model_filename.replace('.json', '_features.json')}")

# =======================================================
# 12. TEST THE MODEL
# =======================================================

predicted_class = model.predict(X_test)
probabilities = model.predict_proba(X_test)

accuracy = (predicted_class == y_test).mean()
test_macro_f1 = f1_score(y_test, predicted_class, average="macro")

print("\n========== TEST RESULTS ==========")
print(f"Classification Accuracy: {accuracy:.4%}")
print(f"Test Macro-F1: {test_macro_f1:.4f}")

# =======================================================
# 13. BASELINE ACCURACY
# =======================================================

baseline = np.bincount(y_test).max() / len(y_test)

print(f"Baseline Accuracy: {baseline:.4%}")
print(f"Improvement over baseline: {(accuracy - baseline):.4%}")

# =======================================================
# 14. CONVERT RESULTS TO NUMPY
# =======================================================

pred_np = predicted_class.flatten()
actual_np = y_test.flatten()

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

print("\n========== FIRST 10 MODEL PROBABILITIES ==========")
print("Order: [short, neutral, long]\n")

for i in range(min(10, len(probabilities))):
    print(f"Test row {i}: ", probabilities[i])

# =======================================================
# 19. PAIRWISE FEATURE COMBINATION ANALYSIS
# =======================================================
# Unchanged from the NN version -- this is pure data analysis on the
# test split, independent of which model produced actual_np/pred_np.

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
# 20. FINISHED
# =======================================================

print("\n========== ANALYSIS COMPLETE ==========")
