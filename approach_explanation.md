# 🚦 Traffic Demand Prediction — Approach Explanation

## Problem Understanding

The goal is to predict **traffic demand** (a normalized float between 0 and 1) for each geographic location (identified by a `geohash`) at each 15-minute time slot. The evaluation metric is:

```
score = max(0, 100 × R²(actual, predicted))
```

A score of 100 = perfect prediction. A score of 0 = model is no better than predicting the mean.

---

## Deep Dataset Analysis

### Dataset Structure

| File | Rows | Columns | Purpose |
|------|------|---------|---------|
| `train.csv` | 77,299 | 11 | Training data (days 48–49) |
| `test.csv` | 41,778 | 10 | Test data (day 49 only) |
| `sample_submission.csv` | 5 | 2 | Shows required output format |

### Key Observations

**1. Time structure**: The data has 96 time slots per day (every 15 minutes: 0:00, 0:15, 0:30 ... 23:45).

**2. Geographic structure**: There are 1,249 unique `geohash` locations in training and 1,190 in test, with 1,180 locations overlapping. A `geohash` is a compact string encoding of latitude/longitude.

**3. Train vs Test split**: 
- Training: Day 48 (full 96 time slots) + Day 49 (time slots 0:00 to 2:00, i.e., the first 9 slots)
- Test: Day 49, time slots 2:15 through 13:45 (47 unique time slots)
- So we're predicting **day-ahead demand** — day 49 afternoon using day 48 data

**4. Target distribution**:
- Range: `[~0, 1.0]`
- Mean: 0.094 (very low demand on average)
- Highly right-skewed — most locations have low demand, a few have very high demand

**5. Missing data**:
- `RoadType`: 600 missing in train, 324 in test
- `Temperature`: 2,495 missing in train, 1,349 in test
- `Weather`: 797 missing in train, 431 in test

### Critical Feature Analysis

The most revealing analysis was the **mean demand by RoadType**:

| RoadType | Mean Demand | Interpretation |
|----------|-------------|----------------|
| Highway | **0.611** | Very high traffic — major arteries |
| Street | **0.273** | Medium traffic — urban collectors |
| Residential | **0.057** | Low traffic — local roads |

This is a 10× difference between Highway and Residential! This makes `RoadType` one of the strongest single features.

**Temporal patterns**: Demand peaks around hours 9–13 (morning commute through midday) and dips at 15–20 (afternoon lull, then recovery in evening).

**The key insight**: When we test simple historical lookup (same `geohash`, same `time_slot` from day 48), it already achieves an R² of **0.52**. A basic gradient boosting model using historical features achieves **0.75+**.

---

## Why This Approach?

### Core Strategy: Geo-Temporal Historical Lookup + Gradient Boosting

This is a **supervised regression** problem with strong **spatial** and **temporal** patterns. The fundamental insight is:

> Traffic demand at a location is highly predictable if you know (a) what the demand at that same location was at the same time previously, and (b) the road characteristics.

#### Why NOT Deep Learning / Neural Networks?

1. **Dataset size is small** (77K rows) — neural networks need much more data to avoid overfitting
2. **No spatial embeddings needed** — geohash historical statistics capture location effects perfectly
3. **Training time** — gradient boosting trains in minutes; neural nets would take hours
4. **Tabular data advantage** — gradient boosting consistently outperforms neural networks on structured/tabular data (confirmed by many Kaggle competitions)

#### Why Gradient Boosting (LightGBM + XGBoost + HistGBR)?

**Gradient boosting** builds an ensemble of decision trees sequentially, where each new tree corrects the errors of the previous ones. It's the gold standard for tabular ML competitions because:

- Handles mixed data types (numeric + categorical) natively
- Robust to outliers
- Handles missing values gracefully
- Automatically learns feature interactions
- No need for feature scaling

We use **three different gradient boosting implementations** and ensemble them:

1. **LightGBM**: Leaf-wise tree growth — faster, better on large datasets, often best performer
2. **XGBoost**: Level-wise tree growth — more stable, different inductive bias
3. **HistGradientBoosting** (sklearn): Histogram-based, handles NaN natively, good regularization

---

## Feature Engineering Strategy

### 1. Geo-Temporal Historical Demand (Most Important)

```
geo_ts_mean = mean demand for (geohash, time_slot) across training days
```

This is the single most powerful feature because demand at the same location at the same time of day is very stable. It captures both the **location effect** (busy vs. quiet neighborhood) and the **time effect** (rush hour vs. night).

We also compute `std`, `median`, `min`, `max` for robustness.

### 2. Geohash Aggregates

```
geo_mean = overall mean demand for this geohash
geo_std  = variability of demand for this geohash
```

These are fallback features — if a geohash doesn't appear in training data, these can't be computed, so we also use prefix-level fallbacks.

### 3. Geohash Prefix Features (Spatial Hierarchy)

Geohashes are hierarchical — `qp03xk` is inside `qp03x` which is inside `qp03` which is inside `qp0` which is inside `qp`. By computing statistics at the 4-character and 5-character prefix levels, we get features that work for both seen and unseen geohashes.

```
geo4_mean = mean demand for all geohashes starting with the same 4 chars
geo5_mean = mean demand for all geohashes starting with the same 5 chars
```

### 4. Temporal Features

- **Raw**: `hour`, `minute`, `time_slot` (0–95)
- **Cyclical encoding**: `sin(2π × hour/24)` and `cos(2π × hour/24)` — this is critical because hour 23 and hour 0 are adjacent in time but far apart numerically. Sine/cosine encoding wraps around correctly.

### 5. Road & Infrastructure Features

- `RoadType_enc`: Encoded as Residential=0, Street=1, Highway=2 (ordered by demand)
- `LargeVehicles_enc`, `Landmarks_enc`: Binary indicators
- `NumberofLanes`: Direct numeric feature

### 6. Interaction Features

```
road_ts_mean      = mean demand for (RoadType, time_slot)
road_hour_mean    = mean demand for (RoadType, hour)
geo_hour_mean     = mean demand for (geohash, hour)
lanes_road_mean   = mean demand for (NumberofLanes, RoadType)
```

Interactions capture that a Highway at rush hour behaves differently from a Highway at midnight.

### 7. Missing Value Strategy

- For aggregate features: `NaN` is used as a signal (tree models can exploit NaN splits)
- For temperature: Filled with the training median (near-zero correlation with demand anyway)
- When merging: `how='left'` ensures test data keeps its original rows even if no match

---

## Model Architecture

### 5-Fold Cross Validation

We use **K-Fold Cross Validation** with 5 folds. This means:
1. Split training data into 5 equal parts
2. Train on 4 parts, validate on 1 part
3. Repeat 5 times, each time using a different part as validation
4. Average the 5 validation scores to get an unbiased estimate of test performance

This gives us **Out-of-Fold (OOF) predictions** — predictions for every training sample made by a model that didn't see that sample during training. OOF predictions are useful for:
- Getting an unbiased performance estimate
- Stacking/ensembling (the OOF predictions are the meta-features)

### Optimal Ensemble Weighting

Instead of simple averaging, we find the **optimal weights** for each model using `scipy.optimize.minimize`. We minimize negative OOF R² — i.e., find the combination of model predictions that maximizes R² on the training data.

```python
final_pred = w1 × lgb_pred + w2 × xgb_pred + w3 × hgb_pred
# where w1 + w2 + w3 = 1 and all wi ≥ 0
```

---

## Why This Should Score Well

1. **Historical geo-temporal demand** is the dominant signal (~60% feature importance)
2. **RoadType** is the dominant categorical signal (~38% feature importance in simple GBM)
3. **Three diverse models** with different inductive biases → ensemble is more robust
4. **5-fold CV** gives reliable performance estimates and produces good OOF predictions for ensembling
5. **Spatial fallbacks** ensure no test sample has completely missing features
6. **Clipping predictions** to [0, 1] prevents physically impossible predictions

---

## Potential Improvements (If Time Permits)

1. **Geohash decoding**: Use a geohash decoder library to extract actual latitude/longitude coordinates, then compute spatial proximity features (nearby locations' demands)
2. **Lag features**: If the data had more consecutive days, we could use lag-1, lag-2 demand
3. **Day-level scaling**: Day 49 showed ~5× higher demand than day 48 in overlapping timestamps — applying a scaling factor might help
4. **Optuna hyperparameter tuning**: Automated hyperparameter search
5. **Neural network blending**: A simple MLP could add diversity to the ensemble
6. **Target encoding**: More sophisticated categorical encoding using target statistics with smoothing
