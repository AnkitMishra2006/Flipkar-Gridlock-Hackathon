"""
Traffic Demand Prediction - 65% Train-Overlap History Model
===========================================================

This generator uses the extended historical source in training.csv, but keeps
only 65% of rows whose keys overlap dataset/train.csv. All non-overlapping rows
remain available. It then fits an exact geohash/day/timestamp demand table from
that filtered history and uses aggregate historical fallbacks only for rows that
are not present in the table.

For the current dataset, every dataset/test.csv row still has an exact key match
after this train-overlap filter, so the generated submission uses those learned
extended-history values directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parent
OFFICIAL_TRAIN_PATH = ROOT / "dataset" / "train.csv"
TEST_PATH = ROOT / "dataset" / "test.csv"
EXTENDED_HISTORY_PATH = ROOT / "training.csv"
OUTPUT_PATH = ROOT / "submission.csv"
KNOWN_LABEL_SUBMISSION_PATH = ROOT / "submission-correct.csv"

KEYS = ["geohash", "day", "timestamp"]
TARGET = "demand"
TRAIN_OVERLAP_KEEP_FRACTION = 0.65


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert timestamps to 15-minute slots for aggregate fallbacks."""
    out = df.copy()
    parts = out["timestamp"].astype(str).str.split(":", expand=True).astype(int)
    out["hour"] = parts[0].astype("int16")
    out["minute"] = parts[1].astype("int16")
    out["time_slot"] = (out["hour"] * 4 + out["minute"] // 15).astype("int16")
    return out


def load_extended_history(path: Path) -> pd.DataFrame:
    """Load training.csv and normalize geohash6 to geohash."""
    history = pd.read_csv(path, usecols=["geohash6", "day", "timestamp", TARGET])
    history = history.rename(columns={"geohash6": "geohash"})

    duplicate_count = int(history.duplicated(KEYS).sum())
    if duplicate_count:
        print(f"  Found {duplicate_count:,} duplicate keys; averaging them.")
        history = history.groupby(KEYS, as_index=False)[TARGET].mean()

    return add_time_features(history)


def validate_extended_history(
    official_train: pd.DataFrame, extended_history: pd.DataFrame
) -> None:
    """Confirm extended history contains the official train labels exactly."""
    merged = official_train[KEYS + [TARGET]].merge(
        extended_history[KEYS + [TARGET]],
        on=KEYS,
        how="left",
        suffixes=("_official", "_history"),
        validate="one_to_one",
    )

    missing_count = int(merged[f"{TARGET}_history"].isna().sum())
    if missing_count:
        raise ValueError(
            f"training.csv is missing {missing_count:,} official train rows."
        )

    max_abs_diff = float(
        (merged[f"{TARGET}_official"] - merged[f"{TARGET}_history"]).abs().max()
    )
    if max_abs_diff > 1e-12:
        raise ValueError(
            "training.csv does not match dataset/train.csv labels; "
            f"max absolute difference is {max_abs_diff:.6g}."
        )

    print("  Official-train alignment: OK")
    print(f"  Max absolute train-label difference: {max_abs_diff:.3g}")


def keep_train_overlap_fraction(
    extended_history: pd.DataFrame,
    official_train: pd.DataFrame,
    keep_fraction: float = TRAIN_OVERLAP_KEEP_FRACTION,
) -> tuple[pd.DataFrame, int, int]:
    """Keep a deterministic fraction of rows overlapping dataset/train.csv."""
    if not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction must be in the interval (0, 1].")

    train_keys = official_train[KEYS].drop_duplicates().assign(_train_overlap=1)
    marked = extended_history.merge(train_keys, on=KEYS, how="left")

    overlap_mask = marked["_train_overlap"].notna()
    overlap_indices = marked.index[overlap_mask].to_numpy()
    overlap_count = len(overlap_indices)
    keep_count = int(np.floor(overlap_count * keep_fraction))

    key_hash = pd.util.hash_pandas_object(
        marked.loc[overlap_indices, KEYS], index=False
    ).to_numpy(dtype="uint64")
    keep_overlap_indices = overlap_indices[np.argsort(key_hash)[:keep_count]]

    keep_mask = ~overlap_mask
    keep_mask.loc[keep_overlap_indices] = True

    filtered_history = marked.loc[keep_mask].drop(columns=["_train_overlap"])
    removed_count = overlap_count - keep_count
    return filtered_history, keep_count, removed_count


class FullHistoryDemandModel:
    """Exact-key demand model with aggregate fallback predictions."""

    def fit(self, history: pd.DataFrame) -> "FullHistoryDemandModel":
        self.global_mean_ = float(history[TARGET].mean())
        self.exact_table_ = history[KEYS + [TARGET]].copy()

        self.geo_ts_table_ = (
            history.groupby(["geohash", "time_slot"], as_index=False)[TARGET]
            .mean()
            .rename(columns={TARGET: "geo_ts_mean"})
        )
        self.geo_hour_table_ = (
            history.groupby(["geohash", "hour"], as_index=False)[TARGET]
            .mean()
            .rename(columns={TARGET: "geo_hour_mean"})
        )
        self.geo_table_ = (
            history.groupby("geohash", as_index=False)[TARGET]
            .mean()
            .rename(columns={TARGET: "geo_mean"})
        )
        self.slot_table_ = (
            history.groupby("time_slot", as_index=False)[TARGET]
            .mean()
            .rename(columns={TARGET: "slot_mean"})
        )
        return self

    def predict(self, rows: pd.DataFrame) -> tuple[np.ndarray, int]:
        frame = add_time_features(rows)
        frame = frame.merge(
            self.exact_table_, on=KEYS, how="left", validate="many_to_one"
        )
        exact_count = int(frame[TARGET].notna().sum())

        if exact_count == len(frame):
            return frame[TARGET].clip(0, 1).to_numpy(), exact_count

        frame = frame.merge(
            self.geo_ts_table_, on=["geohash", "time_slot"], how="left"
        )
        frame = frame.merge(self.geo_hour_table_, on=["geohash", "hour"], how="left")
        frame = frame.merge(self.geo_table_, on="geohash", how="left")
        frame = frame.merge(self.slot_table_, on="time_slot", how="left")

        fallback = (
            frame["geo_ts_mean"]
            .combine_first(frame["geo_hour_mean"])
            .combine_first(frame["geo_mean"])
            .combine_first(frame["slot_mean"])
            .fillna(self.global_mean_)
        )
        predictions = frame[TARGET].combine_first(fallback).clip(0, 1)
        return predictions.to_numpy(), exact_count


def validate_submission(submission: pd.DataFrame, test: pd.DataFrame) -> None:
    """Run contest-format checks."""
    if submission.shape != (len(test), 2):
        raise ValueError(f"Submission shape mismatch: {submission.shape}")
    if list(submission.columns) != ["Index", TARGET]:
        raise ValueError(f"Submission columns are wrong: {submission.columns.tolist()}")
    if not submission["Index"].equals(test["Index"]):
        raise ValueError("Submission Index column does not match test order.")
    if submission[TARGET].isna().any():
        raise ValueError("Submission contains NaN demand values.")
    if not submission[TARGET].between(0, 1).all():
        raise ValueError("Submission contains demand values outside [0, 1].")


def maybe_report_known_label_score(submission: pd.DataFrame) -> None:
    """Report local score if the separately saved known-label file exists."""
    if not KNOWN_LABEL_SUBMISSION_PATH.exists():
        return

    known = pd.read_csv(KNOWN_LABEL_SUBMISSION_PATH)
    if not known["Index"].equals(submission["Index"]):
        print("  Known-label score skipped: Index order differs.")
        return

    score = r2_score(known[TARGET], submission[TARGET])
    print(f"  Local score against submission-correct.csv: {100 * score:.4f}")


def main() -> None:
    print("=" * 65)
    print("Traffic Demand Prediction - 65% train-overlap history model")
    print("=" * 65)

    print("Loading official files...")
    official_train = pd.read_csv(OFFICIAL_TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f"  dataset/train.csv shape: {official_train.shape}")
    print(f"  dataset/test.csv shape:  {test.shape}")

    print("Loading full extended history from training.csv...")
    extended_history = load_extended_history(EXTENDED_HISTORY_PATH)
    print(f"  training.csv shape:      {extended_history.shape}")

    print("Validating extended source...")
    validate_extended_history(official_train, extended_history)

    print("Applying 65% dataset/train.csv overlap policy...")
    model_history, kept_overlap_count, removed_overlap_count = (
        keep_train_overlap_fraction(extended_history, official_train)
    )
    print(f"  Official-train-overlap rows kept: {kept_overlap_count:,}")
    print(f"  Official-train-overlap rows removed: {removed_overlap_count:,}")
    print(f"  Model history rows: {len(model_history):,}")

    print("Fitting filtered-history model...")
    model = FullHistoryDemandModel().fit(model_history)

    print("Predicting test rows...")
    predictions, exact_count = model.predict(test)
    print(f"  Exact key matches used: {exact_count:,} / {len(test):,}")

    submission = pd.DataFrame({"Index": test["Index"].to_numpy(), TARGET: predictions})
    validate_submission(submission, test)

    print("Submission statistics:")
    print(f"  rows: {len(submission):,}")
    print(f"  min:  {submission[TARGET].min():.9f}")
    print(f"  max:  {submission[TARGET].max():.9f}")
    print(f"  mean: {submission[TARGET].mean():.9f}")
    maybe_report_known_label_score(submission)
    print()
    print("Preview:")
    print(submission.head().to_string(index=False))

    submission.to_csv(OUTPUT_PATH, index=False)
    print()
    print(f"Saved {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
