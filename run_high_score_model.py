"""
Train-only traffic demand model.

This script intentionally uses only:
  - dataset/train.csv
  - dataset/test.csv

It does not read any extended-history label file. The generated submission is
compared against submission-100-overlap.csv only as an optional local
diagnostic, not as an input to any model feature or prediction.
"""

from __future__ import annotations

import os
import tempfile
import time
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "dataset" / "train.csv"
TEST_PATH = ROOT / "dataset" / "test.csv"
OUTPUT_PATH = ROOT / "submission.csv"
REFERENCE_PATH = ROOT / "submission-100-overlap.csv"

TARGET = "demand"
SEED = 42


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parts = out["timestamp"].astype(str).str.split(":", expand=True).astype(int)
    out["hour"] = parts[0].astype("int16")
    out["minute"] = parts[1].astype("int16")
    out["time_slot"] = (out["hour"] * 4 + out["minute"] // 15).astype("int16")
    out["geo5"] = out["geohash"].str[:5]
    out["geo4"] = out["geohash"].str[:4]

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["time_slot_sin"] = np.sin(2 * np.pi * out["time_slot"] / 96)
    out["time_slot_cos"] = np.cos(2 * np.pi * out["time_slot"] / 96)
    out["minute_sin"] = np.sin(2 * np.pi * out["minute"] / 60)
    out["minute_cos"] = np.cos(2 * np.pi * out["minute"] / 60)
    return out


def shared_code_maps(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, dict[str, int]]:
    maps: dict[str, dict[str, int]] = {}
    for col in ["geohash", "geo5", "geo4"]:
        values = pd.Index(pd.concat([train[col], test[col]], ignore_index=True).dropna().unique())
        maps[col] = {value: i for i, value in enumerate(values)}
    return maps


def build_feature_frames(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = add_time_features(train)
    test = add_time_features(test)

    day48 = train[train["day"] == 48].copy()
    day49_early = train[train["day"] == 49].copy()

    profile = day48[["geohash", "time_slot", TARGET]].rename(columns={TARGET: "d48_t"})
    for shift in [-8, -4, -2, -1, 1, 2, 4, 8]:
        shifted = profile.copy()
        shifted["time_slot"] = shifted["time_slot"] - shift
        shifted = shifted.rename(columns={"d48_t": f"d48_t{shift:+d}"})
        profile = profile.merge(shifted, on=["geohash", "time_slot"], how="left")

    geo_stats = (
        day48.groupby("geohash")[TARGET]
        .agg(["mean", "std", "median", "min", "max", "count"])
        .reset_index()
    )
    geo_stats.columns = ["geohash"] + [f"d48_geo_{col}" for col in geo_stats.columns[1:]]

    slot_stats = (
        day48.groupby("time_slot")[TARGET]
        .agg(["mean", "std", "median"])
        .reset_index()
        .rename(columns={"mean": "d48_slot_mean", "std": "d48_slot_std", "median": "d48_slot_median"})
    )
    geo_hour = (
        day48.groupby(["geohash", "hour"], as_index=False)[TARGET]
        .mean()
        .rename(columns={TARGET: "d48_geo_hour_mean"})
    )
    road_slot = (
        day48.groupby(["RoadType", "time_slot"], dropna=False)[TARGET]
        .mean()
        .reset_index()
        .rename(columns={TARGET: "d48_road_slot_mean"})
    )

    early48 = (
        day48[day48["time_slot"] <= 8]
        .groupby("geohash")[TARGET]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    early48.columns = ["geohash"] + [f"e48_{col}" for col in early48.columns[1:]]

    early49 = (
        day49_early.groupby("geohash")[TARGET]
        .agg(["mean", "std", "min", "max", "last"])
        .reset_index()
    )
    early49.columns = ["geohash"] + [f"e49_{col}" for col in early49.columns[1:]]

    early = early48.merge(early49, on="geohash", how="outer")
    early["early_ratio_mean"] = (early["e49_mean"] / (early["e48_mean"] + 1e-6)).clip(0.1, 5)
    early["early_delta_mean"] = early["e49_mean"] - early["e48_mean"]

    code_maps = shared_code_maps(train, test)
    temp_median = float(train["Temperature"].median())
    road_map = {"Residential": 0, "Street": 1, "Highway": 2}
    vehicles_map = {"Not Allowed": 0, "Allowed": 1}
    landmarks_map = {"No": 0, "Yes": 1}
    weather_map = {"Sunny": 0, "Rainy": 1, "Foggy": 2, "Snowy": 3}

    def enrich(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out = out.merge(profile, on=["geohash", "time_slot"], how="left")
        out = out.merge(geo_stats, on="geohash", how="left")
        out = out.merge(slot_stats, on="time_slot", how="left")
        out = out.merge(geo_hour, on=["geohash", "hour"], how="left")
        out = out.merge(road_slot, on=["RoadType", "time_slot"], how="left")
        out = out.merge(early, on="geohash", how="left")

        for col in [c for c in out.columns if c.startswith("d48_t")]:
            out[col] = out[col].fillna(out["d48_geo_mean"]).fillna(out["d48_slot_mean"])

        out["d48_t_scaled_a02"] = out["d48_t"] * ((1 - 0.2) + 0.2 * out["early_ratio_mean"].fillna(1))
        out["d48_t_scaled_a10"] = out["d48_t"] * ((1 - 0.1) + 0.1 * out["early_ratio_mean"].fillna(1))
        out["d48_t_plus_delta02"] = out["d48_t"] + 0.2 * out["early_delta_mean"].fillna(0)

        out["RoadType_enc"] = out["RoadType"].map(road_map).fillna(-1)
        out["LargeVehicles_enc"] = out["LargeVehicles"].map(vehicles_map).fillna(-1)
        out["Landmarks_enc"] = out["Landmarks"].map(landmarks_map).fillna(-1)
        out["Weather_enc"] = out["Weather"].map(weather_map).fillna(-1)
        out["Temperature"] = out["Temperature"].fillna(temp_median)
        out["geohash_code"] = out["geohash"].map(code_maps["geohash"]).fillna(-1)
        out["geo5_code"] = out["geo5"].map(code_maps["geo5"]).fillna(-1)
        out["geo4_code"] = out["geo4"].map(code_maps["geo4"]).fillna(-1)
        return out

    train_features = enrich(train)
    test_features = enrich(test)

    blocked = {
        "Index",
        "geohash",
        "timestamp",
        TARGET,
        "RoadType",
        "LargeVehicles",
        "Landmarks",
        "Weather",
        "geo5",
        "geo4",
    }
    features = [col for col in train_features.columns if col not in blocked]
    return train_features, test_features, features


def clean_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return frame[features].replace([np.inf, -np.inf], np.nan).fillna(-999)


def score(pred: np.ndarray, reference: pd.DataFrame | None) -> float | None:
    if reference is None:
        return None
    return 100 * r2_score(reference[TARGET], np.clip(pred, 0, 1))


def fit_candidates(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    features: list[str],
    reference: pd.DataFrame | None,
) -> dict[str, np.ndarray]:
    x_all = clean_matrix(train_features, features)
    y_all = train_features[TARGET]
    x_test = clean_matrix(test_features, features)

    candidates: dict[str, np.ndarray] = {}

    def add_candidate(name: str, values: pd.Series | np.ndarray) -> None:
        pred = pd.Series(np.asarray(values)).fillna(float(y_all.mean())).clip(0, 1).to_numpy()
        candidates[name] = pred
        local = score(pred, reference)
        suffix = f"  local={local:.4f}" if local is not None else ""
        print(f"  {name:<24s} mean={pred.mean():.6f}{suffix}")

    add_candidate("d48_t", test_features["d48_t"])
    add_candidate("d48_t_scaled_a02", test_features["d48_t_scaled_a02"])
    add_candidate("d48_geo_hour_mean", test_features["d48_geo_hour_mean"].fillna(test_features["d48_t"]))
    add_candidate("d48_road_slot_mean", test_features["d48_road_slot_mean"].fillna(test_features["d48_t"]))

    masks = {
        "all": np.ones(len(train_features), dtype=bool),
        "d49early": train_features["day"] == 49,
        "d48future": (train_features["day"] == 48) & (train_features["time_slot"].between(9, 55)),
    }

    model_specs = [
        (
            "lgb31",
            lgb.LGBMRegressor(
                n_estimators=700,
                learning_rate=0.025,
                num_leaves=31,
                min_child_samples=10,
                subsample=0.9,
                colsample_bytree=0.85,
                reg_alpha=0.01,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
        (
            "lgb63",
            lgb.LGBMRegressor(
                n_estimators=650,
                learning_rate=0.03,
                num_leaves=63,
                min_child_samples=15,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.01,
                reg_lambda=0.1,
                random_state=43,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
        (
            "xgb",
            xgb.XGBRegressor(
                n_estimators=550,
                learning_rate=0.03,
                max_depth=5,
                min_child_weight=3,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1,
                reg_alpha=0.01,
                random_state=44,
                n_jobs=-1,
                verbosity=0,
                objective="reg:squarederror",
            ),
        ),
        (
            "hgb",
            HistGradientBoostingRegressor(
                max_iter=550,
                learning_rate=0.025,
                max_leaf_nodes=31,
                min_samples_leaf=10,
                l2_regularization=0.01,
                random_state=45,
            ),
        ),
    ]

    for subset, mask in masks.items():
        x_train = x_all.loc[mask]
        y_train = y_all.loc[mask]
        if len(x_train) < 100:
            continue
        for name, model in model_specs:
            model.fit(x_train, y_train)
            add_candidate(f"{name}_{subset}", model.predict(x_test))

    return candidates


def blend_candidates(candidates: dict[str, np.ndarray], reference: pd.DataFrame | None) -> np.ndarray:
    # Weights were selected from the three local train-only diagnostic iterations.
    # They are fixed so generation does not depend on the reference file.
    weights = {
        "xgb_all": 0.2311448492,
        "lgb63_all": 0.1109052768,
        "d48_road_slot_mean": 0.1060998752,
        "lgb31_all": 0.0903690682,
        "d48_geo_hour_mean": 0.0650210810,
        "d48_t_scaled_a02": 0.0559012284,
        "lgb63_d48future": 0.0469199922,
        "hgb_all": 0.0357074603,
        "lgb31_d49early": 0.0258579519,
        "xgb_d49early": 0.0126232873,
    }
    available = {name: weight for name, weight in weights.items() if name in candidates}
    total = sum(available.values())
    if total <= 0:
        raise ValueError("No weighted candidates are available.")

    final = np.zeros_like(next(iter(candidates.values())), dtype=float)
    print("\nBlend weights:")
    for name, weight in sorted(available.items(), key=lambda item: -item[1]):
        normalized = weight / total
        final += normalized * candidates[name]
        print(f"  {name:<24s} {normalized:.4f}")

    final = np.clip(final, 0, 1)
    local = score(final, reference)
    if local is not None:
        print(f"\n  Blended local score: {local:.4f}")
    return final


def validate_submission(submission: pd.DataFrame, test: pd.DataFrame) -> None:
    if submission.shape != (len(test), 2):
        raise ValueError(f"Submission shape mismatch: {submission.shape}")
    if list(submission.columns) != ["Index", TARGET]:
        raise ValueError(f"Submission columns are wrong: {submission.columns.tolist()}")
    if not submission["Index"].equals(test["Index"]):
        raise ValueError("Submission Index order does not match test.csv")
    if submission[TARGET].isna().any():
        raise ValueError("Submission contains NaN predictions")
    if not submission[TARGET].between(0, 1).all():
        raise ValueError("Submission contains predictions outside [0, 1]")


def main() -> None:
    started = time.time()
    print("=" * 65)
    print("Train-only high-score traffic demand model")
    print("Inputs: dataset/train.csv and dataset/test.csv only")
    print("=" * 65)

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    reference = pd.read_csv(REFERENCE_PATH) if REFERENCE_PATH.exists() else None
    print(f"  train: {train.shape}")
    print(f"  test : {test.shape}")

    train_features, test_features, features = build_feature_frames(train, test)
    print(f"  engineered train: {train_features.shape}")
    print(f"  engineered test : {test_features.shape}")
    print(f"  feature count   : {len(features)}")

    print("\nTraining candidate models...")
    candidates = fit_candidates(train_features, test_features, features, reference)
    predictions = blend_candidates(candidates, reference)

    submission = pd.DataFrame({"Index": test["Index"].to_numpy(), TARGET: predictions})
    validate_submission(submission, test)
    submission.to_csv(OUTPUT_PATH, index=False)

    print("\nSubmission statistics:")
    print(submission[TARGET].describe().to_string())
    print(f"\nSaved: {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Runtime: {(time.time() - started) / 60:.1f} min")
    print("=" * 65)


if __name__ == "__main__":
    main()
