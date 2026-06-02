"""
Traffic Demand Prediction -- Submission Generator
=================================================
Dataset path change from previous version:
  BEFORE: Two sources were used:
    - training.csv  (large, 4 cols: geohash6/day/timestamp/demand)
                     -> geo-temporal aggregate lookup tables
    - dataset/train.csv (11 cols, full features)
                     -> model training + road/weather aggregates

  NOW (this version): Single source -- dataset/train.csv only
    - dataset/train.csv (11 cols: Index/geohash/day/timestamp/demand/
                          RoadType/NumberofLanes/LargeVehicles/Landmarks/
                          Temperature/Weather)
                     -> ALL aggregates + model training
    - dataset/test.csv  -> inference (unchanged)

  All other pipeline steps are IDENTICAL:
    same feature engineering, same encoding, same 3-model ensemble
    (LightGBM + XGBoost + HistGBR), same KFold CV (5 folds),
    same hyperparameters, same ensemble weight optimization.

Evaluation metric: score = max(0, 100 * R2_score(actual, predicted))
Submission format: Index, demand  --  41778 x 2
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb

SEED = 42
np.random.seed(SEED)

# ============================================================
# STEP 1: Load Data
# ============================================================
print("=" * 65)
print("STEP 1: Loading data...")
print("=" * 65)

# --- PATH CHANGE vs. previous version ---
# Previously: train_large = pd.read_csv('./training.csv')   [4-col, large]
#             train       = pd.read_csv('./dataset/train.csv')
# Now:        train       = pd.read_csv('./dataset/train.csv')  [single source]
# test path is unchanged.

print("  Loading dataset/train.csv...", flush=True)
train = pd.read_csv('./dataset/train.csv')
print(f"  dataset/train.csv shape:   {train.shape}")
print(f"  Columns: {train.columns.tolist()}")
print(f"  Unique geohashes: {train['geohash'].nunique()}")

print("  Loading dataset/test.csv...", flush=True)
test = pd.read_csv('./dataset/test.csv')
print(f"  dataset/test.csv shape:    {test.shape}")
print()

# ============================================================
# STEP 2: Parse timestamps
# ============================================================
print("=" * 65)
print("STEP 2: Parsing timestamps...")
print("=" * 65)

def parse_timestamp(df):
    """Convert 'H:MM' string into hour, minute, time_slot (0-95)."""
    df = df.copy()
    ts = df['timestamp'].astype(str)
    df['hour']      = ts.apply(lambda x: int(x.split(':')[0]))
    df['minute']    = ts.apply(lambda x: int(x.split(':')[1]))
    df['time_slot'] = df['hour'] * 4 + df['minute'] // 15
    return df

train = parse_timestamp(train)
test  = parse_timestamp(test)

print(f"  time_slot range: {train['time_slot'].min()} to {train['time_slot'].max()}")
print(f"  day range:       {train['day'].min()} to {train['day'].max()}")
print()

# ============================================================
# STEP 3: Compute ALL aggregate features from dataset/train.csv
#         (previously split across training.csv + dataset/train.csv;
#          now all come from dataset/train.csv)
# ============================================================
print("=" * 65)
print("STEP 3: Computing aggregate features from dataset/train.csv...")
print("=" * 65)

# --- Geo-temporal aggregates (same as before, now from dataset/train.csv) ---

# 3a. (geohash, time_slot) -- MOST IMPORTANT FEATURE
geo_ts_agg = (
    train.groupby(['geohash', 'time_slot'])['demand']
    .agg(['mean', 'std', 'median', 'min', 'max', 'count'])
    .reset_index()
    .rename(columns={
        'mean':   'geo_ts_mean',
        'std':    'geo_ts_std',
        'median': 'geo_ts_median',
        'min':    'geo_ts_min',
        'max':    'geo_ts_max',
        'count':  'geo_ts_count',
    })
)

# 3b. Overall geohash statistics
geo_agg = (
    train.groupby('geohash')['demand']
    .agg(['mean', 'std', 'median', 'min', 'max', 'count'])
    .reset_index()
    .rename(columns={
        'mean':   'geo_mean',
        'std':    'geo_std',
        'median': 'geo_median',
        'min':    'geo_min',
        'max':    'geo_max',
        'count':  'geo_count',
    })
)

# 3c. (geohash, hour) -- location-specific daily cycle
geo_hour_agg = (
    train.groupby(['geohash', 'hour'])['demand']
    .agg(['mean', 'std'])
    .reset_index()
    .rename(columns={'mean': 'geo_hour_mean', 'std': 'geo_hour_std'})
)

# 3d. (geohash, day) -- location ? day pattern
geo_day_agg = (
    train.groupby(['geohash', 'day'])['demand']
    .mean()
    .reset_index()
    .rename(columns={'demand': 'geo_day_mean'})
)

# 3e. Time slot statistics
ts_agg = (
    train.groupby('time_slot')['demand']
    .agg(['mean', 'std'])
    .reset_index()
    .rename(columns={'mean': 'ts_mean', 'std': 'ts_std'})
)

# 3f. Hour-level statistics
hour_agg = (
    train.groupby('hour')['demand']
    .mean()
    .reset_index()
    .rename(columns={'demand': 'hour_mean'})
)

# 3g. Day-level statistics
day_agg = (
    train.groupby('day')['demand']
    .agg(['mean', 'std'])
    .reset_index()
    .rename(columns={'mean': 'day_mean', 'std': 'day_std'})
)

# 3h. Geohash prefix fallbacks (for unseen / sparse geohashes)
train['geo_prefix_5'] = train['geohash'].str[:5]
train['geo_prefix_4'] = train['geohash'].str[:4]

geo5_ts_agg = (
    train.groupby(['geo_prefix_5', 'time_slot'])['demand']
    .mean()
    .reset_index()
    .rename(columns={'demand': 'geo5_ts_mean'})
)
geo4_ts_agg = (
    train.groupby(['geo_prefix_4', 'time_slot'])['demand']
    .mean()
    .reset_index()
    .rename(columns={'demand': 'geo4_ts_mean'})
)
geo5_agg = (
    train.groupby('geo_prefix_5')['demand']
    .agg(['mean', 'std'])
    .reset_index()
    .rename(columns={'mean': 'geo5_mean', 'std': 'geo5_std'})
)
geo4_agg = (
    train.groupby('geo_prefix_4')['demand']
    .agg(['mean', 'std'])
    .reset_index()
    .rename(columns={'mean': 'geo4_mean', 'std': 'geo4_std'})
)

# --- Road/weather aggregates (same as before, dataset/train.csv) ---

# RoadType aggregates
road_agg = (
    train.groupby('RoadType')['demand']
    .agg(['mean', 'std'])
    .reset_index()
    .rename(columns={'mean': 'roadtype_mean', 'std': 'roadtype_std'})
)

# (RoadType, time_slot) interaction
road_ts_agg = (
    train.groupby(['RoadType', 'time_slot'])['demand']
    .mean()
    .reset_index()
    .rename(columns={'demand': 'road_ts_mean'})
)

# (RoadType, hour) interaction
road_hour_agg = (
    train.groupby(['RoadType', 'hour'])['demand']
    .mean()
    .reset_index()
    .rename(columns={'demand': 'road_hour_mean'})
)

# (NumberofLanes, RoadType) interaction
lanes_road_agg = (
    train.groupby(['NumberofLanes', 'RoadType'])['demand']
    .mean()
    .reset_index()
    .rename(columns={'demand': 'lanes_road_mean'})
)

# Weather aggregates
weather_agg = (
    train.groupby('Weather')['demand']
    .mean()
    .reset_index()
    .rename(columns={'demand': 'weather_mean'})
)

# (geohash, RoadType) combo
geo_road_agg = (
    train.groupby(['geohash', 'RoadType'])['demand']
    .mean()
    .reset_index()
    .rename(columns={'demand': 'geo_road_mean'})
)

print(f"  geo_ts_agg:       {geo_ts_agg.shape}")
print(f"  geo_agg:          {geo_agg.shape}")
print(f"  geo_hour_agg:     {geo_hour_agg.shape}")
print(f"  road_agg:         {road_agg.shape}")
print(f"  geo_road_agg:     {geo_road_agg.shape}")
print()

# ============================================================
# STEP 4: Add prefix columns to train and test, then merge features
# ============================================================
print("=" * 65)
print("STEP 4: Merging all features...")
print("=" * 65)

GLOBAL_MEAN = train['demand'].mean()
print(f"  Global mean demand (dataset/train.csv): {GLOBAL_MEAN:.6f}")

def add_geo_prefix_cols(df):
    df = df.copy()
    df['geo_prefix_5'] = df['geohash'].str[:5]
    df['geo_prefix_4'] = df['geohash'].str[:4]
    return df

# train already has prefix cols from Step 3; apply to test
test = add_geo_prefix_cols(test)

def merge_all_features(df):
    """Merge all pre-computed aggregate features (no data leakage)."""
    df = df.copy()

    # Geo-temporal (from dataset/train.csv)
    df = df.merge(geo_ts_agg,   on=['geohash', 'time_slot'],      how='left')
    df = df.merge(geo_agg,      on='geohash',                     how='left')
    df = df.merge(geo_hour_agg, on=['geohash', 'hour'],           how='left')
    df = df.merge(geo_day_agg,  on=['geohash', 'day'],            how='left')
    df = df.merge(ts_agg,       on='time_slot',                   how='left')
    df = df.merge(hour_agg,     on='hour',                        how='left')
    df = df.merge(day_agg,      on='day',                         how='left')
    df = df.merge(geo5_ts_agg,  on=['geo_prefix_5', 'time_slot'], how='left')
    df = df.merge(geo4_ts_agg,  on=['geo_prefix_4', 'time_slot'], how='left')
    df = df.merge(geo5_agg,     on='geo_prefix_5',                how='left')
    df = df.merge(geo4_agg,     on='geo_prefix_4',                how='left')

    # Road/weather (from dataset/train.csv)
    df = df.merge(road_agg,       on='RoadType',                    how='left')
    df = df.merge(road_ts_agg,    on=['RoadType', 'time_slot'],     how='left')
    df = df.merge(road_hour_agg,  on=['RoadType', 'hour'],          how='left')
    df = df.merge(lanes_road_agg, on=['NumberofLanes', 'RoadType'], how='left')
    df = df.merge(weather_agg,    on='Weather',                     how='left')
    df = df.merge(geo_road_agg,   on=['geohash', 'RoadType'],       how='left')

    return df

train_feat = merge_all_features(train)
test_feat  = merge_all_features(test)

print(f"  train_feat shape: {train_feat.shape}")
print(f"  test_feat shape:  {test_feat.shape}")
print()

# ============================================================
# STEP 5: Encode categoricals and engineer final features
# ============================================================
print("=" * 65)
print("STEP 5: Encoding categoricals & engineering features...")
print("=" * 65)

ROADTYPE_MAP      = {'Residential': 0, 'Street': 1, 'Highway': 2}
LARGEVEHICLES_MAP = {'Not Allowed': 0, 'Allowed': 1}
LANDMARKS_MAP     = {'No': 0, 'Yes': 1}
WEATHER_MAP       = {'Sunny': 0, 'Rainy': 1, 'Foggy': 2, 'Snowy': 3}

for df in [train_feat, test_feat]:
    df['RoadType_enc']      = df['RoadType'].map(ROADTYPE_MAP).fillna(-1)
    df['LargeVehicles_enc'] = df['LargeVehicles'].map(LARGEVEHICLES_MAP).fillna(-1)
    df['Landmarks_enc']     = df['Landmarks'].map(LANDMARKS_MAP).fillna(-1)
    df['Weather_enc']       = df['Weather'].map(WEATHER_MAP).fillna(-1)

# Fill Temperature with training median
TEMP_MEDIAN = train['Temperature'].median()
for df in [train_feat, test_feat]:
    df['Temperature'] = df['Temperature'].fillna(TEMP_MEDIAN)

# Cyclical time encodings
for df in [train_feat, test_feat]:
    df['hour_sin']   = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']   = np.cos(2 * np.pi * df['hour'] / 24)
    df['ts_sin']     = np.sin(2 * np.pi * df['time_slot'] / 96)
    df['ts_cos']     = np.cos(2 * np.pi * df['time_slot'] / 96)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['day_sin']    = np.sin(2 * np.pi * df['day'] / 7)
    df['day_cos']    = np.cos(2 * np.pi * df['day'] / 7)

# Derived ratio features
for df in [train_feat, test_feat]:
    df['geo_ts_vs_geo']  = df['geo_ts_mean']  / (df['geo_mean']   + 1e-8)
    df['geo_ts_cv']      = df['geo_ts_std']   / (df['geo_ts_mean'] + 1e-8)
    df['road_vs_global'] = df['roadtype_mean'] / (GLOBAL_MEAN      + 1e-8)
    df['geo_hour_ratio'] = df['geo_hour_mean'] / (df['hour_mean']  + 1e-8)

print("  Categorical encoding done.")
print()

# ============================================================
# STEP 6: Define feature set and prepare arrays
#         (identical to previous version)
# ============================================================
FEATURES = [
    # === GEO-TEMPORAL (MOST IMPORTANT) ===
    'geo_ts_mean',
    'geo_ts_std',
    'geo_ts_median',
    'geo_ts_min',
    'geo_ts_max',
    'geo_ts_count',

    # === GEOHASH AGGREGATES ===
    'geo_mean',
    'geo_std',
    'geo_median',
    'geo_min',
    'geo_max',
    'geo_count',

    # === LOCATION ? TIME INTERACTIONS ===
    'geo_hour_mean',
    'geo_hour_std',
    'geo_day_mean',

    # === SPATIAL FALLBACKS (prefix-based) ===
    'geo5_ts_mean',
    'geo4_ts_mean',
    'geo5_mean',
    'geo5_std',
    'geo4_mean',
    'geo4_std',

    # === TEMPORAL AGGREGATES ===
    'ts_mean',
    'ts_std',
    'hour_mean',
    'day_mean',
    'day_std',

    # === ROAD/WEATHER INTERACTIONS ===
    'roadtype_mean',
    'roadtype_std',
    'road_ts_mean',
    'road_hour_mean',
    'lanes_road_mean',
    'weather_mean',
    'geo_road_mean',

    # === RATIO / DERIVED FEATURES ===
    'geo_ts_vs_geo',
    'geo_ts_cv',
    'road_vs_global',
    'geo_hour_ratio',

    # === RAW TIME FEATURES ===
    'time_slot',
    'hour',
    'minute',
    'day',

    # === CYCLICAL ENCODINGS ===
    'hour_sin', 'hour_cos',
    'ts_sin',   'ts_cos',
    'minute_sin', 'minute_cos',
    'day_sin',  'day_cos',

    # === ENCODED CATEGORICALS ===
    'RoadType_enc',
    'LargeVehicles_enc',
    'Landmarks_enc',
    'Weather_enc',

    # === NUMERICAL ===
    'Temperature',
    'NumberofLanes',
]

X_train = train_feat[FEATURES].copy()
y_train = train_feat['demand'].copy()
X_test  = test_feat[FEATURES].copy()

# Fill remaining NaN with -999 (tree models handle sentinel values)
X_train = X_train.fillna(-999)
X_test  = X_test.fillna(-999)

print(f"Training features shape: {X_train.shape}")
print(f"Test features shape:     {X_test.shape}")
print(f"NaN in X_train: {X_train.isnull().sum().sum()}")
print(f"NaN in X_test:  {X_test.isnull().sum().sum()}")
print()

# ============================================================
# STEP 7: Train LightGBM with K-Fold CV
#         (same hyperparameters as previous version)
# ============================================================
print("=" * 65)
print("STEP 7: Training LightGBM...")
print("=" * 65)

lgb_params = {
    'objective':         'regression',
    'metric':            'rmse',
    'boosting_type':     'gbdt',
    'num_leaves':        127,
    'max_depth':         -1,
    'learning_rate':     0.05,
    'n_estimators':      2000,
    'subsample':         0.8,
    'colsample_bytree':  0.8,
    'min_child_samples': 20,
    'reg_alpha':         0.01,
    'reg_lambda':        0.05,
    'random_state':      SEED,
    'n_jobs':            -1,
    'verbose':           -1,
}

N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

lgb_oof_preds  = np.zeros(len(X_train))
lgb_test_preds = np.zeros(len(X_test))
lgb_r2_scores  = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
    print(f"  Fold {fold}/{N_FOLDS}...", end=' ', flush=True)

    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    val_pred = np.clip(model.predict(X_val, num_iteration=model.best_iteration_), 0, 1)
    fold_r2  = r2_score(y_val, val_pred)
    lgb_r2_scores.append(fold_r2)
    lgb_oof_preds[val_idx] = val_pred
    lgb_test_preds += model.predict(X_test, num_iteration=model.best_iteration_) / N_FOLDS

    print(f"R? = {fold_r2:.4f}  |  Score = {max(0, 100*fold_r2):.2f}")

lgb_test_preds = np.clip(lgb_test_preds, 0, 1)
lgb_oof_r2     = r2_score(y_train, lgb_oof_preds)
print(f"\nLightGBM OOF R? = {lgb_oof_r2:.4f}  |  Score = {max(0, 100*lgb_oof_r2):.2f}")
print()

# ============================================================
# STEP 8: Train XGBoost with K-Fold CV
#         (same hyperparameters as previous version)
# ============================================================
print("=" * 65)
print("STEP 8: Training XGBoost...")
print("=" * 65)

xgb_params = {
    'objective':        'reg:squarederror',
    'eval_metric':      'rmse',
    'max_depth':        7,
    'learning_rate':    0.05,
    'n_estimators':     2000,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'gamma':            0.01,
    'reg_alpha':        0.01,
    'reg_lambda':       1.0,
    'random_state':     SEED,
    'n_jobs':           -1,
    'verbosity':        0,
}

xgb_oof_preds  = np.zeros(len(X_train))
xgb_test_preds = np.zeros(len(X_test))
xgb_r2_scores  = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
    print(f"  Fold {fold}/{N_FOLDS}...", end=' ', flush=True)

    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=100)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    val_pred = np.clip(model.predict(X_val), 0, 1)
    fold_r2  = r2_score(y_val, val_pred)
    xgb_r2_scores.append(fold_r2)
    xgb_oof_preds[val_idx] = val_pred
    xgb_test_preds += model.predict(X_test) / N_FOLDS

    print(f"R? = {fold_r2:.4f}  |  Score = {max(0, 100*fold_r2):.2f}")

xgb_test_preds = np.clip(xgb_test_preds, 0, 1)
xgb_oof_r2     = r2_score(y_train, xgb_oof_preds)
print(f"\nXGBoost OOF R? = {xgb_oof_r2:.4f}  |  Score = {max(0, 100*xgb_oof_r2):.2f}")
print()

# ============================================================
# STEP 9: Train HistGradientBoosting with K-Fold CV
#         (same hyperparameters as previous version)
# ============================================================
print("=" * 65)
print("STEP 9: Training HistGradientBoostingRegressor...")
print("=" * 65)

hgb_params = {
    'max_iter':            2000,
    'max_leaf_nodes':      127,
    'max_depth':           None,
    'learning_rate':       0.05,
    'l2_regularization':   0.05,
    'min_samples_leaf':    20,
    'random_state':        SEED,
    'early_stopping':      True,
    'n_iter_no_change':    100,
    'validation_fraction': 0.1,
}

hgb_oof_preds  = np.zeros(len(X_train))
hgb_test_preds = np.zeros(len(X_test))
hgb_r2_scores  = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
    print(f"  Fold {fold}/{N_FOLDS}...", end=' ', flush=True)

    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # HistGBR handles NaN natively -> replace -999 sentinel
    X_tr_h   = X_tr.replace(-999, np.nan)
    X_val_h  = X_val.replace(-999, np.nan)
    X_test_h = X_test.replace(-999, np.nan)

    model = HistGradientBoostingRegressor(**hgb_params)
    model.fit(X_tr_h, y_tr)

    val_pred = np.clip(model.predict(X_val_h), 0, 1)
    fold_r2  = r2_score(y_val, val_pred)
    hgb_r2_scores.append(fold_r2)
    hgb_oof_preds[val_idx] = val_pred
    hgb_test_preds += model.predict(X_test_h) / N_FOLDS

    print(f"R? = {fold_r2:.4f}  |  Score = {max(0, 100*fold_r2):.2f}")

hgb_test_preds = np.clip(hgb_test_preds, 0, 1)
hgb_oof_r2     = r2_score(y_train, hgb_oof_preds)
print(f"\nHistGBR OOF R? = {hgb_oof_r2:.4f}  |  Score = {max(0, 100*hgb_oof_r2):.2f}")
print()

# ============================================================
# STEP 10: Optimize ensemble weights using OOF predictions
#          (same approach as previous version)
# ============================================================
print("=" * 65)
print("STEP 10: Optimizing ensemble weights...")
print("=" * 65)

def ensemble_objective(weights):
    w = np.abs(weights)
    w = w / w.sum()
    preds = (w[0] * lgb_oof_preds +
             w[1] * xgb_oof_preds +
             w[2] * hgb_oof_preds)
    return -r2_score(y_train, np.clip(preds, 0, 1))

result = minimize(
    ensemble_objective,
    [1.0, 1.0, 1.0],
    method='Nelder-Mead',
    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6},
)

opt_w = np.abs(result.x) / np.abs(result.x).sum()
print(f"  LightGBM weight: {opt_w[0]:.4f}")
print(f"  XGBoost weight:  {opt_w[1]:.4f}")
print(f"  HistGBR weight:  {opt_w[2]:.4f}")

ensemble_oof = np.clip(
    opt_w[0] * lgb_oof_preds +
    opt_w[1] * xgb_oof_preds +
    opt_w[2] * hgb_oof_preds,
    0, 1,
)
ensemble_r2 = r2_score(y_train, ensemble_oof)

final_preds = np.clip(
    opt_w[0] * lgb_test_preds +
    opt_w[1] * xgb_test_preds +
    opt_w[2] * hgb_test_preds,
    0, 1,
)

print()
print("Individual model OOF scores:")
print(f"  LightGBM  : {max(0, 100*lgb_oof_r2):.2f}")
print(f"  XGBoost   : {max(0, 100*xgb_oof_r2):.2f}")
print(f"  HistGBR   : {max(0, 100*hgb_oof_r2):.2f}")
print(f"  >> Ensemble: {max(0, 100*ensemble_r2):.2f}")
print()

# ============================================================
# STEP 11: Generate and save submission.csv
# ============================================================
print("=" * 65)
print("STEP 11: Generating submission.csv...")
print("=" * 65)

submission = pd.DataFrame({
    'Index':  test['Index'],
    'demand': final_preds,
})

# Validation checks
assert submission.shape == (41778, 2), (
    f"Shape mismatch: expected (41778, 2), got {submission.shape}"
)
assert list(submission.columns) == ['Index', 'demand'], (
    f"Column mismatch: {submission.columns.tolist()}"
)
assert submission['demand'].between(0, 1).all(), "Some demand values out of [0,1] range!"
assert submission['demand'].isna().sum() == 0, "NaN values in demand column!"

print(f"  Submission shape:    {submission.shape}  [OK]")
print(f"  Columns:             {submission.columns.tolist()}  [OK]")
print(f"  Values in [0,1]:     {submission['demand'].between(0, 1).all()}  [OK]")
print(f"  NaN values:          {submission['demand'].isna().sum()}  [OK]")
print()
print("  Demand statistics:")
print(f"    min  = {submission['demand'].min():.6f}")
print(f"    max  = {submission['demand'].max():.6f}")
print(f"    mean = {submission['demand'].mean():.6f}")
print()
print("  Preview (first 5 rows):")
print(submission.head().to_string(index=False))

submission.to_csv('./submission.csv', index=False)

print()
print("=" * 65)
print("DONE  submission.csv saved to ./submission.csv")
print(f"    Rows: {len(submission)} | Columns: {list(submission.columns)}")
print(f"    Final ensemble OOF score: {max(0, 100*ensemble_r2):.2f}")
print("=" * 65)
