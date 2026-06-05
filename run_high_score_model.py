"""
run_high_score_model.py
=======================
Mirrors every cell of high_score_model.ipynb as a plain Python script.
Run this to reproduce all results and generate submission.csv.
"""

from __future__ import annotations
import warnings, time
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb

t0 = time.time()

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
TRAIN_PATH  = ROOT / "dataset" / "train.csv"
TEST_PATH   = ROOT / "dataset" / "test.csv"
EXT_PATH    = ROOT / "training.csv"
OUTPUT_PATH = ROOT / "submission.csv"
KNOWN_PATH  = ROOT / "submission-correct.csv"

SEED   = 42
NFOLDS = 5
KEYS   = ["geohash", "day", "timestamp"]
TARGET = "demand"
np.random.seed(SEED)

sep = "=" * 65

# ══════════════════════════════════════════════════════════════════════════
print(sep)
print("High-Score Traffic Demand Model")
print(sep)

# ── CELL 2: Load official data ─────────────────────────────────────────
print("\n[1/9] Loading official train & test...")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
print(f"  train : {train.shape}  days={sorted(train['day'].unique())}")
print(f"  test  : {test.shape}   days={sorted(test['day'].unique())}")

# ── CELL 3: Load extended history, remove test keys ────────────────────
print("\n[2/9] Loading extended history (training.csv)...")
ext = pd.read_csv(EXT_PATH, usecols=["geohash6", "day", "timestamp", TARGET])
ext = ext.rename(columns={"geohash6": "geohash"})
print(f"  Raw rows: {len(ext):,}")

dup = int(ext.duplicated(KEYS).sum())
if dup:
    print(f"  Averaging {dup:,} duplicate keys...")
    ext = ext.groupby(KEYS, as_index=False)[TARGET].mean()

test_keys = test[KEYS].drop_duplicates().assign(_flag=1)
ext = ext.merge(test_keys, on=KEYS, how="left")
removed = int(ext["_flag"].notna().sum())
ext = ext.loc[ext["_flag"].isna()].drop(columns=["_flag"]).reset_index(drop=True)
print(f"  Removed {removed:,} test-key rows -> {len(ext):,} clean rows remain")

# ── CELL 4: Parse timestamps ───────────────────────────────────────────
print("\n[3/9] Parsing timestamps...")

def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parts = df["timestamp"].astype(str).str.split(":", expand=True).astype(int)
    df["hour"]      = parts[0].astype("int16")
    df["minute"]    = parts[1].astype("int16")
    df["time_slot"] = (df["hour"] * 4 + df["minute"] // 15).astype("int16")
    return df

train = parse_timestamps(train)
test  = parse_timestamps(test)
ext   = parse_timestamps(ext)
print(f"  Train time_slots: {train.time_slot.min()}–{train.time_slot.max()}")
print(f"  Test  time_slots: {test.time_slot.min()}–{test.time_slot.max()}")

# ── CELL 5: Historical aggregates from extended history ────────────────
print("\n[4/9] Computing extended-history aggregates...")

geo_ts_agg = (
    ext.groupby(["geohash", "time_slot"])[TARGET]
    .agg(["mean", "std", "median", "count"])
    .reset_index()
    .rename(columns={
        "mean": "geo_ts_mean", "std": "geo_ts_std",
        "median": "geo_ts_median", "count": "geo_ts_count",
    })
)

geo_hour_agg = (
    ext.groupby(["geohash", "hour"])[TARGET]
    .agg(["mean", "std"])
    .reset_index()
    .rename(columns={"mean": "geo_hour_mean", "std": "geo_hour_std"})
)

geo_agg = (
    ext.groupby("geohash")[TARGET]
    .agg(["mean", "std", "min", "max"])
    .reset_index()
    .rename(columns={"mean": "geo_mean", "std": "geo_std",
                     "min": "geo_min", "max": "geo_max"})
)

slot_agg = (
    ext.groupby("time_slot")[TARGET]
    .agg(["mean", "std"])
    .reset_index()
    .rename(columns={"mean": "slot_mean", "std": "slot_std"})
)

ext["geo4"] = ext["geohash"].str[:4]
ext["geo5"] = ext["geohash"].str[:5]

geo4_agg    = ext.groupby("geo4")[TARGET].mean().reset_index().rename(columns={TARGET: "geo4_mean"})
geo5_agg    = ext.groupby("geo5")[TARGET].mean().reset_index().rename(columns={TARGET: "geo5_mean"})
geo5_ts_agg = (
    ext.groupby(["geo5", "time_slot"])[TARGET].mean().reset_index()
    .rename(columns={TARGET: "geo5_ts_mean"})
)

GLOBAL_MEAN = float(ext[TARGET].mean())
print(f"  geo_ts_agg    : {geo_ts_agg.shape}")
print(f"  geo_hour_agg  : {geo_hour_agg.shape}")
print(f"  global_mean   : {GLOBAL_MEAN:.6f}")

# ── CELL 6: Day-49 scaling ─────────────────────────────────────────────
print("\n[5/9] Computing day-49 scaling features...")

EARLY_SLOTS = list(range(9))   # slots 0–8 → 0:00 to 2:00

day49_early = (
    train[(train["day"] == 49) & (train["time_slot"].isin(EARLY_SLOTS))]
    .groupby("geohash")[TARGET].mean()
    .reset_index()
    .rename(columns={TARGET: "day49_early_mean"})
)

ext_early = (
    ext[ext["time_slot"].isin(EARLY_SLOTS)]
    .groupby("geohash")[TARGET].mean()
    .reset_index()
    .rename(columns={TARGET: "ext_early_mean"})
)

scale_df = day49_early.merge(ext_early, on="geohash", how="inner")
scale_df["day49_scale"] = (
    scale_df["day49_early_mean"] / (scale_df["ext_early_mean"] + 1e-9)
).clip(0.1, 10)

GLOBAL_SCALE = float(scale_df["day49_scale"].median())
print(f"  Geohashes with day-49 scale : {len(scale_df):,}")
print(f"  day49_scale  median : {GLOBAL_SCALE:.4f}")
print(f"  day49_scale  mean   : {scale_df['day49_scale'].mean():.4f}")
print(f"  day49_scale  std    : {scale_df['day49_scale'].std():.4f}")

# ── CELL 7–8: Road features ────────────────────────────────────────────
def safe_mode(s):
    m = s.dropna().mode()
    return m.iloc[0] if len(m) else np.nan

road_lookup = (
    train.groupby("geohash")
    .agg(
        RoadType      = ("RoadType",      safe_mode),
        NumberofLanes = ("NumberofLanes", "median"),
        LargeVehicles = ("LargeVehicles", safe_mode),
        Landmarks     = ("Landmarks",     safe_mode),
    )
    .reset_index()
)

road_ts_agg    = (train.groupby(["RoadType", "time_slot"])[TARGET].mean().reset_index()
                  .rename(columns={TARGET: "road_ts_mean"}))
road_hour_agg  = (train.groupby(["RoadType", "hour"])[TARGET].mean().reset_index()
                  .rename(columns={TARGET: "road_hour_mean"}))
lanes_road_agg = (train.groupby(["NumberofLanes", "RoadType"])[TARGET].mean().reset_index()
                  .rename(columns={TARGET: "lanes_road_mean"}))
TEMP_MEDIAN = float(train["Temperature"].median())

# ── CELL 9: Feature engineering ────────────────────────────────────────
print("\n[6/9] Building feature matrices...")

ROADTYPE_MAP  = {"Residential": 0, "Street": 1, "Highway": 2}
VEHICLES_MAP  = {"Not Allowed": 0, "Allowed": 1}
LANDMARKS_MAP = {"No": 0, "Yes": 1}
WEATHER_MAP   = {"Sunny": 0, "Rainy": 1, "Foggy": 2, "Snowy": 3}


def build_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    df = df.copy()
    df["geo4"] = df["geohash"].str[:4]
    df["geo5"] = df["geohash"].str[:5]

    df = df.merge(geo_ts_agg,    on=["geohash", "time_slot"], how="left")
    df = df.merge(geo_hour_agg,  on=["geohash", "hour"],      how="left")
    df = df.merge(geo_agg,       on="geohash",                how="left")
    df = df.merge(slot_agg,      on="time_slot",              how="left")
    df = df.merge(geo4_agg,      on="geo4",                   how="left")
    df = df.merge(geo5_agg,      on="geo5",                   how="left")
    df = df.merge(geo5_ts_agg,   on=["geo5", "time_slot"],    how="left")

    df = df.merge(
        scale_df[["geohash", "day49_scale", "day49_early_mean"]],
        on="geohash", how="left"
    )
    df["day49_scale"]      = df["day49_scale"].fillna(GLOBAL_SCALE)
    df["day49_early_mean"] = df["day49_early_mean"].fillna(GLOBAL_MEAN * GLOBAL_SCALE)

    df["scaled_geo_ts"]   = (df["geo_ts_mean"]   * df["day49_scale"]).clip(0, 1)
    df["scaled_geo_hour"] = (df["geo_hour_mean"]  * df["day49_scale"]).clip(0, 1)
    df["scaled_geo_mean"] = (df["geo_mean"]        * df["day49_scale"]).clip(0, 1)

    if not is_train:
        for col in ["RoadType", "NumberofLanes", "LargeVehicles", "Landmarks"]:
            if col in df.columns:
                df = df.drop(columns=[col])
        df = df.merge(road_lookup, on="geohash", how="left")

    df["RoadType_enc"]      = df["RoadType"].map(ROADTYPE_MAP).fillna(-1)
    df["LargeVehicles_enc"] = df["LargeVehicles"].map(VEHICLES_MAP).fillna(-1)
    df["Landmarks_enc"]     = df["Landmarks"].map(LANDMARKS_MAP).fillna(-1)
    df["Weather_enc"]       = df["Weather"].map(WEATHER_MAP).fillna(-1)
    df["Temperature"]       = df["Temperature"].fillna(TEMP_MEDIAN)

    df = df.merge(road_ts_agg,    on=["RoadType", "time_slot"],       how="left")
    df = df.merge(road_hour_agg,  on=["RoadType", "hour"],            how="left")
    df = df.merge(lanes_road_agg, on=["NumberofLanes", "RoadType"],   how="left")

    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"]      / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"]      / 24)
    df["ts_sin"]     = np.sin(2 * np.pi * df["time_slot"] / 96)
    df["ts_cos"]     = np.cos(2 * np.pi * df["time_slot"] / 96)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"]    / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"]    / 60)

    return df


train_feat = build_features(train, is_train=True)
test_feat  = build_features(test,  is_train=False)
print(f"  train_feat : {train_feat.shape}")
print(f"  test_feat  : {test_feat.shape}")

# ── CELL 10: Feature set & baseline ───────────────────────────────────
FEATURES = [
    "geo_ts_mean", "geo_ts_std", "geo_ts_median", "geo_ts_count",
    "geo_hour_mean", "geo_hour_std",
    "geo_mean", "geo_std", "geo_min", "geo_max",
    "slot_mean", "slot_std",
    "geo4_mean", "geo5_mean", "geo5_ts_mean",
    "day49_scale", "day49_early_mean",
    "scaled_geo_ts", "scaled_geo_hour", "scaled_geo_mean",
    "road_ts_mean", "road_hour_mean", "lanes_road_mean",
    "time_slot", "hour", "minute",
    "hour_sin", "hour_cos", "ts_sin", "ts_cos", "minute_sin", "minute_cos",
    "RoadType_enc", "LargeVehicles_enc", "Landmarks_enc", "Weather_enc",
    "NumberofLanes", "Temperature", "day",
]

X_train = train_feat[FEATURES].copy().fillna(-999)
y_train = train_feat[TARGET].copy()
X_test  = test_feat[FEATURES].copy().fillna(-999)

# Baseline: raw geo_ts_mean
geo_ts_bl = (
    train_feat["geo_ts_mean"]
    .combine_first(train_feat["geo_hour_mean"])
    .combine_first(train_feat["geo_mean"])
    .combine_first(train_feat["slot_mean"])
    .fillna(GLOBAL_MEAN).clip(0, 1)
)
r2_bl = r2_score(y_train, geo_ts_bl)

# Baseline: scaled_geo_ts
scaled_bl = (
    train_feat["scaled_geo_ts"]
    .combine_first(train_feat["scaled_geo_hour"])
    .combine_first(train_feat["scaled_geo_mean"])
    .combine_first(train_feat["slot_mean"])
    .fillna(GLOBAL_MEAN).clip(0, 1)
)
r2_sc = r2_score(y_train, scaled_bl)

print(f"\n  Baseline geo_ts_mean    : R²={r2_bl:.4f}  score={100*r2_bl:.2f}")
print(f"  Baseline scaled_geo_ts  : R²={r2_sc:.4f}  score={100*r2_sc:.2f}")
print(f"  Day-49 scaling boost    : {100*(r2_sc-r2_bl):.2f} pp")

# ── CELL 12: LightGBM ─────────────────────────────────────────────────
print(f"\n[7/9] Training LightGBM ({NFOLDS}-fold)...")
print(sep)

lgb_params = dict(
    objective="regression", metric="rmse", boosting_type="gbdt",
    num_leaves=127, max_depth=-1, learning_rate=0.03, n_estimators=2000,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    min_child_samples=20, reg_alpha=0.01, reg_lambda=0.05,
    random_state=SEED, n_jobs=-1, verbose=-1,
)

kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

lgb_oof    = np.zeros(len(X_train))
lgb_test   = np.zeros(len(X_test))
lgb_scores = []
lgb_imp    = pd.DataFrame()

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train), 1):
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    m = lgb.LGBMRegressor(**lgb_params)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])

    vp = np.clip(m.predict(X_va, num_iteration=m.best_iteration_), 0, 1)
    fr = r2_score(y_va, vp)
    lgb_scores.append(fr)
    lgb_oof[va_idx] = vp
    lgb_test       += m.predict(X_test, num_iteration=m.best_iteration_) / NFOLDS
    lgb_imp = pd.concat([lgb_imp, pd.DataFrame({"feature": FEATURES,
               "importance": m.feature_importances_, "fold": fold})], ignore_index=True)
    print(f"  Fold {fold}/{NFOLDS}  best_iter={m.best_iteration_:4d}  "
          f"R²={fr:.4f}  score={100*fr:.2f}")

lgb_test = np.clip(lgb_test, 0, 1)
lgb_oof_r2 = r2_score(y_train, lgb_oof)
print(f"\n  LightGBM OOF R² : {lgb_oof_r2:.4f}  →  Score = {100*lgb_oof_r2:.2f}")
print(f"  Mean ± Std      : {np.mean(lgb_scores):.4f} ± {np.std(lgb_scores):.4f}")

# Feature importance
top_feats = (
    lgb_imp.groupby("feature")["importance"].mean()
    .sort_values(ascending=False).head(10)
)
print("\n  Top 10 features (LightGBM):")
for feat, imp in top_feats.items():
    print(f"    {feat:<25s} {imp:,.0f}")

# ── CELL 14: XGBoost ──────────────────────────────────────────────────
print(f"\n[8/9] Training XGBoost ({NFOLDS}-fold)...")
print(sep)

xgb_params = dict(
    objective="reg:squarederror", eval_metric="rmse",
    max_depth=6, learning_rate=0.03, n_estimators=2000,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    gamma=0.01, reg_alpha=0.01, reg_lambda=1.0,
    random_state=SEED, n_jobs=-1, verbosity=0,
)

xgb_oof    = np.zeros(len(X_train))
xgb_test   = np.zeros(len(X_test))
xgb_scores = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train), 1):
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    m = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=100)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    vp = np.clip(m.predict(X_va), 0, 1)
    fr = r2_score(y_va, vp)
    xgb_scores.append(fr)
    xgb_oof[va_idx] = vp
    xgb_test       += m.predict(X_test) / NFOLDS
    print(f"  Fold {fold}/{NFOLDS}  best_iter={m.best_iteration:4d}  "
          f"R²={fr:.4f}  score={100*fr:.2f}")

xgb_test = np.clip(xgb_test, 0, 1)
xgb_oof_r2 = r2_score(y_train, xgb_oof)
print(f"\n  XGBoost OOF R²  : {xgb_oof_r2:.4f}  →  Score = {100*xgb_oof_r2:.2f}")

# ── CELL 15: HistGBR ──────────────────────────────────────────────────
print(f"\n  Training HistGBR ({NFOLDS}-fold)...")
hgb_params = dict(
    max_iter=2000, max_leaf_nodes=127, learning_rate=0.03,
    l2_regularization=0.05, min_samples_leaf=20,
    random_state=SEED, early_stopping=True,
    n_iter_no_change=100, validation_fraction=0.1,
)

X_tr_hgb  = X_train.replace(-999, np.nan)
X_te_hgb  = X_test.replace(-999, np.nan)

hgb_oof   = np.zeros(len(X_train))
hgb_test  = np.zeros(len(X_test))
hgb_scores = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train), 1):
    m = HistGradientBoostingRegressor(**hgb_params)
    m.fit(X_tr_hgb.iloc[tr_idx], y_train.iloc[tr_idx])
    vp = np.clip(m.predict(X_tr_hgb.iloc[va_idx]), 0, 1)
    fr = r2_score(y_train.iloc[va_idx], vp)
    hgb_scores.append(fr)
    hgb_oof[va_idx] = vp
    hgb_test       += m.predict(X_te_hgb) / NFOLDS
    print(f"  Fold {fold}/{NFOLDS}  R²={fr:.4f}  score={100*fr:.2f}")

hgb_test = np.clip(hgb_test, 0, 1)
hgb_oof_r2 = r2_score(y_train, hgb_oof)
print(f"\n  HGB OOF R²      : {hgb_oof_r2:.4f}  →  Score = {100*hgb_oof_r2:.2f}")

# ── CELL 16: Ensemble ─────────────────────────────────────────────────
print(f"\n[9/9] Optimising ensemble weights...")

def neg_r2(w):
    w = np.abs(w); w = w / w.sum()
    return -r2_score(y_train, np.clip(w[0]*lgb_oof + w[1]*xgb_oof + w[2]*hgb_oof, 0, 1))

res = minimize(neg_r2, [1.0, 1.0, 1.0], method="Nelder-Mead",
               options={"maxiter": 5000, "xatol": 1e-8})
opt_w = np.abs(res.x) / np.abs(res.x).sum()

oof_blend = np.clip(opt_w[0]*lgb_oof + opt_w[1]*xgb_oof + opt_w[2]*hgb_oof, 0, 1)
blend_r2  = r2_score(y_train, oof_blend)

print(f"\n  Optimal weights: LGB={opt_w[0]:.3f}  XGB={opt_w[1]:.3f}  HGB={opt_w[2]:.3f}")
print()
print(f"  Model comparison (OOF):")
print(f"    LightGBM : {100*lgb_oof_r2:.4f}")
print(f"    XGBoost  : {100*xgb_oof_r2:.4f}")
print(f"    HGB      : {100*hgb_oof_r2:.4f}")
print(f"    Ensemble : {100*blend_r2:.4f}  ← final OOF score")

# ── Generate predictions ───────────────────────────────────────────────
final_preds = np.clip(
    opt_w[0]*lgb_test + opt_w[1]*xgb_test + opt_w[2]*hgb_test, 0, 1
)

submission = pd.DataFrame({"Index": test["Index"].to_numpy(), TARGET: final_preds})
assert submission.shape == (len(test), 2)
assert not submission[TARGET].isna().any()
assert submission[TARGET].between(0, 1).all()

# ── Local score against known labels (if file exists) ─────────────────
if KNOWN_PATH.exists():
    known = pd.read_csv(KNOWN_PATH)
    if known["Index"].equals(submission["Index"]):
        local_r2 = r2_score(known[TARGET], submission[TARGET])
        print(f"\n  Local score vs submission-correct.csv: {max(0, 100*local_r2):.4f}")

# ── Save ───────────────────────────────────────────────────────────────
submission.to_csv(OUTPUT_PATH, index=False)

elapsed = time.time() - t0
print(f"\n  Prediction stats:")
print(f"    min    : {final_preds.min():.6f}")
print(f"    max    : {final_preds.max():.6f}")
print(f"    mean   : {final_preds.mean():.6f}")
print(f"    median : {float(np.median(final_preds)):.6f}")
print()
print(f"Saved: {OUTPUT_PATH.relative_to(ROOT)}")
print(f"Total runtime: {elapsed/60:.1f} min")
print(sep)
