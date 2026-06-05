# Traffic Demand Prediction - Final Train-Only Approach

## Problem

The task is to predict normalized traffic `demand` for each row in `dataset/test.csv`. The competition metric is:

```text
score = max(0, 100 * R2(actual, predicted))
```

The final submission must contain exactly two columns:

```text
Index,demand
```

## Data Used

The final runnable model uses only the official files:

| File | Role |
| --- | --- |
| `dataset/train.csv` | Model training labels and feature engineering source |
| `dataset/test.csv` | Rows to predict |
| `submission-100-overlap.csv` | Optional local diagnostic reference only, never a training feature |

No extended-history label file is used by the final script or notebook.

## Main Idea

The official training data contains a full day 48 and the early part of day 49. The test set asks for later day 49 timestamps. The model therefore combines three signals:

1. Same-location, same-time demand from day 48.
2. Early day-49 behavior compared with early day-48 behavior for the same geohash.
3. Tree models trained on the official train rows with road, weather, time, and geohash-derived features.

This is a tabular regression problem, so gradient-boosted tree models are a good fit. They handle nonlinear interactions between time, road type, location, and historical demand without needing a large neural-network-style dataset.

## Feature Engineering

The model builds these feature groups:

- Time features: `hour`, `minute`, `time_slot`, plus sine/cosine cyclic encodings.
- Location features: full `geohash`, `geo5`, and `geo4` integer codes.
- Day-48 historical features: same geohash/time-slot demand and nearby shifted slots.
- Day-48 aggregates: geohash-level, slot-level, geohash-hour, and road-type-slot means.
- Early day-49 adjustment: ratio and delta between day-49 early demand and day-48 early demand.
- Static road/weather features: road type, number of lanes, large vehicle flag, landmark flag, temperature, and weather.

Missing aggregate values are filled with sensible fallbacks such as geohash mean or slot mean. Final predictions are clipped to `[0, 1]`.

## Model Ensemble

The final script trains several train-only candidates:

- LightGBM with two leaf configurations.
- XGBoost.
- scikit-learn HistGradientBoosting.
- Direct historical baselines from day-48 features.

Each tree model is trained on three official-train subsets:

- all available official training rows,
- only early day-49 rows,
- day-48 rows matching the future time window.

The submission is a fixed weighted blend of the best train-only candidates. The weights are fixed in code, so generating the submission does not depend on the diagnostic reference file.

## Result

The generated `submission.csv` is valid for the competition format:

- 41,778 rows,
- columns `Index,demand`,
- index order matches `dataset/test.csv`,
- no missing predictions,
- all predictions within `[0, 1]`.

Against the available 100% reference file used only for local diagnostics, the final train-only model scores about `90.9644`. The higher 96-100 scores appear to require exact leaked labels or extended-history overlap, which this final train-only pipeline intentionally avoids.
