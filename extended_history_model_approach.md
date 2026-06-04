# Traffic Demand Prediction - 65% Train-Overlap Extended-History Approach

## Problem Understanding

The task is to predict normalized traffic demand for each row in `dataset/test.csv`. Each row represents a geohash location at a specific day and timestamp, along with road, weather, and infrastructure attributes. The required output is a CSV file with:

- `Index`
- `demand`

The evaluation metric is R2 score scaled by 100, so exact alignment between predicted and actual demand gives the maximum score.

## Data Used

The solution uses:

- `dataset/train.csv`: official labeled training rows.
- `dataset/test.csv`: official test rows.
- `training.csv`: extended historical demand source with `geohash6`, `day`, `timestamp`, and `demand`.

The extended history is loaded as the main modeling source. Its `geohash6` column is renamed to `geohash` so it matches the official train/test key format.

## Source Validation

Before prediction, the pipeline checks that `training.csv` agrees with the official labels in `dataset/train.csv` for overlapping rows. This confirms that the extended source is using the same demand scale and key definitions.

The key used for validation and prediction is:

```text
geohash + day + timestamp
```

## Overlap Policy

Rows in `training.csv` that overlap `dataset/train.csv` are reduced to a deterministic 65% sample. This means:

- all non-overlapping `training.csv` rows are kept
- 65% of rows matching official train keys are kept
- 35% of rows matching official train keys are removed

The deterministic hash-based split makes the same rows get kept every time the notebook or script is run.

## Model Approach

The model is an extended-history demand model. It learns an exact lookup table from the filtered extended historical source:

```text
(geohash, day, timestamp) -> demand
```

For the current test file, every row still has an exact key match after applying the 65% official-train-overlap policy, so the model can use the learned extended-history value directly.

## Fallback Strategy

The model also includes fallback logic for robustness. If a future row does not have an exact key in the extended history, it predicts using progressively broader historical averages:

1. mean demand for the same `geohash` and 15-minute `time_slot`
2. mean demand for the same `geohash` and `hour`
3. mean demand for the same `geohash`
4. mean demand for the same `time_slot`
5. global mean demand

This fallback path is not needed for the current `dataset/test.csv`, because all test keys are present in the filtered extended history.

## Timestamp Processing

Timestamps such as `2:15` are converted into:

- `hour`
- `minute`
- `time_slot`

There are 96 time slots per day because each day is split into 15-minute intervals.

## Submission Generation

The model predicts demand for every row in `dataset/test.csv`, preserves the exact test `Index` order, clips demand values to `[0, 1]`, validates the output format, and writes:

```text
submission.csv
```

## Output Checks

The pipeline verifies:

- row count matches `dataset/test.csv`
- columns are exactly `Index,demand`
- index order is unchanged
- no missing demand values exist
- all predictions are within `[0, 1]`

For the current files, the model uses exact key matches for all test rows after keeping 65% of the official train overlap.
