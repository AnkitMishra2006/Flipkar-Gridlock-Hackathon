"""
Traffic Demand Prediction - Full History Model
==============================================

Python equivalent of traffic_demand_extended_history_model.ipynb.

This script trains on dataset/train.csv (the official labeled training
data) using an exact (geohash, day, timestamp) demand lookup table.
When an exact key is unavailable for a test row it falls back through
progressively broader historical averages:

    (geohash, time_slot)  →  (geohash, hour)  →  geohash
    →  time_slot  →  global mean

The evaluation metric is:  score = max(0, 100 × R²(actual, predicted))

Usage
-----
    python traffic_demand_extended_history_model.py

Output
------
    submission.csv  — two columns: Index, demand
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT               = Path(__file__).resolve().parent
OFFICIAL_TRAIN_PATH = ROOT / "dataset" / "train.csv"
TEST_PATH           = ROOT / "dataset" / "test.csv"
OUTPUT_PATH         = ROOT / "submission.csv"

# Column constants
KEYS   = ["geohash", "day", "timestamp"]
TARGET = "demand"

# ---------------------------------------------------------------------------
# 1. Feature Utility
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse 'H:MM' timestamp strings into numeric time features.

    Adds:
        hour      – integer hour of day (0–23)
        minute    – integer minute (0, 15, 30, or 45)
        time_slot – 15-minute slot index (0–95)
    """
    out = df.copy()
    parts = out["timestamp"].astype(str).str.split(":", expand=True).astype(int)
    out["hour"]      = parts[0].astype("int16")
    out["minute"]    = parts[1].astype("int16")
    out["time_slot"] = (out["hour"] * 4 + out["minute"] // 15).astype("int16")
    return out


# ---------------------------------------------------------------------------
# 2. Full-History Demand Model
# ---------------------------------------------------------------------------

class FullHistoryDemandModel:
    """
    Exact-key demand model with a 5-level aggregate fallback chain.

    fit(history)
        Memorises exact (geohash, day, timestamp) → demand values and
        pre-computes fallback aggregate tables from the training history.

    predict(rows) -> (predictions: np.ndarray, exact_count: int)
        Returns clipped [0, 1] predictions for every row in *rows*.
        *exact_count* reports how many rows were served by an exact match.

    Fallback priority (applied per-row when exact key is absent):
        1. Mean demand for (geohash, time_slot) across all training days
        2. Mean demand for (geohash, hour)
        3. Mean demand for geohash
        4. Mean demand for time_slot (global time-of-day signal)
        5. Global training mean
    """

    def fit(self, history: pd.DataFrame) -> "FullHistoryDemandModel":
        """
        Build lookup tables from *history* (must already contain time features).
        """
        # Scalar fallback of last resort
        self.global_mean_: float = float(history[TARGET].mean())

        # Exact-match table
        self.exact_table_: pd.DataFrame = history[KEYS + [TARGET]].copy()

        # Fallback table 1: (geohash, time_slot) mean
        self.geo_ts_table_: pd.DataFrame = (
            history.groupby(["geohash", "time_slot"], as_index=False)[TARGET]
            .mean()
            .rename(columns={TARGET: "geo_ts_mean"})
        )

        # Fallback table 2: (geohash, hour) mean
        self.geo_hour_table_: pd.DataFrame = (
            history.groupby(["geohash", "hour"], as_index=False)[TARGET]
            .mean()
            .rename(columns={TARGET: "geo_hour_mean"})
        )

        # Fallback table 3: geohash mean
        self.geo_table_: pd.DataFrame = (
            history.groupby("geohash", as_index=False)[TARGET]
            .mean()
            .rename(columns={TARGET: "geo_mean"})
        )

        # Fallback table 4: time_slot mean
        self.slot_table_: pd.DataFrame = (
            history.groupby("time_slot", as_index=False)[TARGET]
            .mean()
            .rename(columns={TARGET: "slot_mean"})
        )

        return self

    def predict(self, rows: pd.DataFrame) -> tuple[np.ndarray, int]:
        """
        Generate demand predictions for every row in *rows*.

        Returns
        -------
        predictions : np.ndarray, shape (n_rows,), dtype float64
            Values clipped to [0, 1].
        exact_count : int
            Number of rows matched exactly in the training history.
        """
        frame = add_time_features(rows)

        # Attempt exact-key lookup first
        frame = frame.merge(
            self.exact_table_, on=KEYS, how="left", validate="many_to_one"
        )
        exact_count = int(frame[TARGET].notna().sum())

        # Fast-path: all rows matched exactly — no fallback needed
        if exact_count == len(frame):
            return frame[TARGET].clip(0, 1).to_numpy(), exact_count

        # Attach all fallback tables (only used where exact match is NaN)
        frame = frame.merge(
            self.geo_ts_table_, on=["geohash", "time_slot"], how="left"
        )
        frame = frame.merge(
            self.geo_hour_table_, on=["geohash", "hour"], how="left"
        )
        frame = frame.merge(self.geo_table_,  on="geohash",  how="left")
        frame = frame.merge(self.slot_table_, on="time_slot", how="left")

        # Chain fallbacks using combine_first (takes first non-null value)
        fallback = (
            frame["geo_ts_mean"]
            .combine_first(frame["geo_hour_mean"])
            .combine_first(frame["geo_mean"])
            .combine_first(frame["slot_mean"])
            .fillna(self.global_mean_)
        )

        # Where the exact match exists, use it; otherwise use the fallback
        predictions = frame[TARGET].combine_first(fallback).clip(0, 1)
        return predictions.to_numpy(), exact_count


# ---------------------------------------------------------------------------
# 3. Submission Validation
# ---------------------------------------------------------------------------

def validate_submission(submission: pd.DataFrame, test: pd.DataFrame) -> None:
    """
    Assert that the submission satisfies contest-format requirements.

    Checks:
        - shape: (n_test_rows, 2)
        - columns: exactly ['Index', 'demand']
        - Index matches test order exactly
        - no NaN demand values
        - all demand values within [0, 1]
    """
    if submission.shape != (len(test), 2):
        raise ValueError(
            f"Submission shape mismatch: got {submission.shape}, "
            f"expected ({len(test)}, 2)."
        )
    if list(submission.columns) != ["Index", TARGET]:
        raise ValueError(
            f"Submission columns are wrong: {submission.columns.tolist()!r}"
        )
    if not submission["Index"].equals(test["Index"]):
        raise ValueError("Submission Index column does not match test row order.")
    if submission[TARGET].isna().any():
        raise ValueError("Submission contains NaN demand values.")
    if not submission[TARGET].between(0, 1).all():
        raise ValueError("Submission contains demand values outside [0, 1].")


# ---------------------------------------------------------------------------
# 4. Optional: Self-evaluation on train split
# ---------------------------------------------------------------------------

def evaluate_on_train(model: FullHistoryDemandModel,
                      train: pd.DataFrame) -> None:
    """
    Report in-sample coverage and R² on the official training set.

    Note: this is NOT a held-out evaluation — it only checks how many
    train rows the model can recall exactly (should be 100% since we
    trained on the same data).

    We pass a copy of train with the target renamed to '_truth' so the
    predict() merge does not produce ambiguous demand_x / demand_y columns.
    """
    truth = train[TARGET].to_numpy()

    # predict() expects rows without a 'demand' column, since it adds
    # 'demand' via merge with exact_table_.  Drop it from the input.
    rows_no_target = train.drop(columns=[TARGET])
    preds, exact = model.predict(rows_no_target)

    r2 = r2_score(truth, preds)
    print(f"  Train exact-key coverage: {exact:,} / {len(train):,} "
          f"({100 * exact / len(train):.1f}%)")
    print(f"  Train R²: {r2:.6f}  |  Score: {max(0, 100 * r2):.4f}")


# ---------------------------------------------------------------------------
# 5. Main Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    sep = "=" * 65
    print(sep)
    print("Traffic Demand Prediction — Full History Model")
    print("Training source: dataset/train.csv")
    print(sep)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading official train and test files...")
    train = pd.read_csv(OFFICIAL_TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    print(f"  dataset/train.csv shape : {train.shape}")
    print(f"  dataset/test.csv shape  : {test.shape}")

    # Sanity-check that required columns exist
    missing_train_cols = set(KEYS + [TARGET]) - set(train.columns)
    if missing_train_cols:
        raise ValueError(
            f"dataset/train.csv is missing required columns: {missing_train_cols}"
        )
    missing_test_cols = set(KEYS) - set(test.columns)
    if missing_test_cols:
        raise ValueError(
            f"dataset/test.csv is missing required columns: {missing_test_cols}"
        )

    # ------------------------------------------------------------------
    # Prepare training history
    # ------------------------------------------------------------------
    print("\n[2/5] Preparing training history...")

    # Check for duplicate keys in training data and average if present
    dup_count = int(train.duplicated(KEYS).sum())
    if dup_count:
        print(f"  Found {dup_count:,} duplicate (geohash, day, timestamp) keys "
              f"— averaging their demand values.")
        train = train.groupby(KEYS, as_index=False)[TARGET].mean()
        # Re-add any other columns if needed (not required for this model)

    # Add time features to training data
    history = add_time_features(train)
    print(f"  Unique (geohash, day, timestamp) keys : {len(history):,}")
    print(f"  Unique geohashes in train             : "
          f"{history['geohash'].nunique():,}")
    print(f"  Days in training data                 : "
          f"{sorted(history['day'].unique())}")
    print(f"  Time slots in training data           : "
          f"{history['time_slot'].nunique():,} / 96")

    # ------------------------------------------------------------------
    # Fit the model
    # ------------------------------------------------------------------
    print("\n[3/5] Fitting Full-History Demand Model...")
    model = FullHistoryDemandModel().fit(history)
    print(f"  global_mean              : {model.global_mean_:.6f}")
    print(f"  exact_table_ rows        : {len(model.exact_table_):,}")
    print(f"  geo_ts_table_ rows       : {len(model.geo_ts_table_):,}")
    print(f"  geo_hour_table_ rows     : {len(model.geo_hour_table_):,}")
    print(f"  geo_table_ rows          : {len(model.geo_table_):,}")
    print(f"  slot_table_ rows         : {len(model.slot_table_):,}")

    # Self-evaluation (in-sample, for sanity-check only)
    print("\n  In-sample sanity check (train):")
    evaluate_on_train(model, train)

    # ------------------------------------------------------------------
    # Predict test demand
    # ------------------------------------------------------------------
    print("\n[4/5] Predicting demand for dataset/test.csv...")
    predictions, exact_count = model.predict(test)
    fallback_count = len(test) - exact_count
    print(f"  Exact-key matches used : {exact_count:,} / {len(test):,} "
          f"({100 * exact_count / len(test):.1f}%)")
    if fallback_count:
        print(f"  Fallback predictions   : {fallback_count:,} rows")
        print(f"  NOTE: {fallback_count:,} test rows had no exact match in "
              f"dataset/train.csv — they were served by aggregate fallbacks.")
    else:
        print("  All test rows matched exactly — no fallback used.")

    # ------------------------------------------------------------------
    # Build and validate submission
    # ------------------------------------------------------------------
    print("\n[5/5] Building and validating submission...")
    submission = pd.DataFrame({
        "Index":  test["Index"].to_numpy(),
        TARGET:   predictions,
    })
    validate_submission(submission, test)
    print("  Format validation: PASSED")

    # Statistics
    s = submission[TARGET]
    print(f"\n  Submission statistics:")
    print(f"    rows  : {len(submission):,}")
    print(f"    min   : {s.min():.9f}")
    print(f"    max   : {s.max():.9f}")
    print(f"    mean  : {s.mean():.9f}")
    print(f"    std   : {s.std():.9f}")
    print(f"    median: {s.median():.9f}")

    # Preview
    print("\n  Preview (first 5 rows):")
    print(submission.head().to_string(index=False))

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH.relative_to(ROOT)}")
    print(sep)


if __name__ == "__main__":
    main()
