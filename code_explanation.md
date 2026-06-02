# 📖 Code Explanation — Traffic Demand Prediction
## (Complete Guide for Python Beginners in Machine Learning)

---

## 🎯 What This Code Does

This notebook predicts **traffic demand** (how busy a road location will be) for the next day using historical traffic data. It reads data, engineers features, trains three machine learning models, combines their predictions, and saves the result as a CSV file for submission.

---

## 📦 CELL 1: Libraries — What They Are and Why We Need Them

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
```

### 🐼 `pandas` (`pd`)
**What it is**: The most important data manipulation library in Python for ML.  
**What it does**: Reads CSV files, creates DataFrames (think Excel spreadsheets in code), joins tables, groups data, computes statistics.  
**Why we need it**: All our data lives in DataFrames. Almost every line of data processing uses pandas.

```python
# Example: Reading a CSV file
train = pd.read_csv('train.csv')
# Example: Getting only rows where day == 48
day48_data = train[train['day'] == 48]
# Example: Computing mean demand per geohash
geo_means = train.groupby('geohash')['demand'].mean()
```

### 🔢 `numpy` (`np`)
**What it is**: The fundamental numerical computation library.  
**What it does**: Fast math on arrays — sine, cosine, clip values, generate random numbers, matrix operations.  
**Why we need it**: Pandas uses numpy under the hood. We use it for cyclical encoding (sin/cos), clipping predictions to [0,1], and creating arrays of zeros.

```python
# Example: Clipping predictions to valid range
np.clip(predictions, 0, 1)   # Ensures nothing goes below 0 or above 1

# Example: Sine encoding for time
np.sin(2 * np.pi * hour / 24)   # Wraps hour values cyclically
```

### 📊 `matplotlib` + `seaborn`
**What they are**: Visualization libraries for creating charts and graphs.  
**matplotlib** is the base library; **seaborn** builds on it with prettier statistical plots.  
**Why we need them**: To understand the data (EDA) and visualize model performance.

### 🤖 `sklearn` (scikit-learn)
**What it is**: The most widely-used machine learning library in Python.  
**What we use from it**:
- `r2_score`: Computes R² (how well predictions match actual values)
- `KFold`: Splits data into folds for cross-validation
- `HistGradientBoostingRegressor`: A gradient boosting model
- `minimize` (scipy): Finds optimal ensemble weights

### ⚡ `lightgbm` (`lgb`)
**What it is**: Microsoft's gradient boosting library — the fastest and often most accurate.  
**Why we use it**: Handles large datasets efficiently, learns complex patterns, handles missing values.

### 🚀 `xgboost` (`xgb`)
**What it is**: Another gradient boosting library — the first one to become popular for Kaggle competitions.  
**Why we use it**: Provides a different "learning style" from LightGBM, so their combined predictions are more accurate than either alone.

---

## 📂 CELL 2: Loading the Dataset

```python
train = pd.read_csv(DATA_PATH + 'train.csv')
test  = pd.read_csv(DATA_PATH + 'test.csv')
```

**What happens**: `pd.read_csv()` reads a CSV file from disk and creates a DataFrame. The result is a table where each row is one observation (one road location at one time) and each column is one variable.

**`train`**: 77,299 rows × 11 columns — has the `demand` column (the answer we're learning from)  
**`test`**: 41,778 rows × 10 columns — same structure but NO `demand` column (this is what we need to predict)

---

## 🔍 CELL 3: Exploratory Data Analysis (EDA)

EDA = looking at the data before modeling. This is crucial — you can't build a good model without understanding your data.

```python
train.head()           # Show first 5 rows
train.isnull().sum()   # Count missing values per column
train['demand'].describe()  # Statistics: min, max, mean, std, quartiles
```

### The Histogram Plot Explained

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
```
**What it does**: Creates a figure with 1 row and 3 columns of subplots, each 18×5 inches total.

```python
axes[0].hist(train['demand'], bins=50, ...)
```
**What it does**: Plots a histogram — divides the demand values into 50 equal-width "buckets" and shows how many values fall in each bucket. This tells us the **distribution** of demand.

```python
axes[2].hist(np.log1p(train['demand']), bins=50, ...)
```
**Why log?**: Demand is right-skewed (most values are near 0, few are near 1). `log1p(x) = log(1 + x)` compresses the scale, making skewed data look more like a bell curve and revealing patterns at low values.

---

## ⚙️ CELL 4: Feature Engineering — The Core of ML

This is where we create the **features** (inputs) for our machine learning models. Raw data rarely has the right form — we need to transform and combine columns to extract useful signal.

### Step 1: Parsing Timestamps

```python
def parse_timestamp(df):
    df['hour']      = df['timestamp'].apply(lambda x: int(x.split(':')[0]))
    df['minute']    = df['timestamp'].apply(lambda x: int(x.split(':')[1]))
    df['time_slot'] = df['hour'] * 4 + df['minute'] // 15
    return df
```

**`df['timestamp'].apply(lambda x: ...)`**: Applies a function to every value in the `timestamp` column.

**`lambda x: int(x.split(':')[0])`**: This is a tiny anonymous function. For input `'14:30'`:
1. `x.split(':')` → `['14', '30']` (splits string at the colon)
2. `[0]` → `'14'` (takes the first element)
3. `int('14')` → `14` (converts string to integer)

**`time_slot = hour * 4 + minute // 15`**: Converts time to a slot number 0–95.
- Hour 0, Minute 0 → slot 0 (midnight)
- Hour 0, Minute 15 → slot 1
- Hour 23, Minute 45 → slot 95
- There are 96 slots total (24 hours × 4 per hour)

**`//`**: Integer division (floor division) — `45 // 15 = 3`, `30 // 15 = 2`

### Step 2: Geohash Prefix Features

```python
df['geo_prefix_4'] = df['geohash'].str[:4]   # First 4 characters
df['geo_prefix_5'] = df['geohash'].str[:5]   # First 5 characters
```

**What's a geohash?**: A geohash is a compact encoding of geographic coordinates. `qp03xk` represents a specific ~100m × 150m area. The first few characters represent a broader area: `qp03` is a ~20km × 20km region.

By computing statistics at multiple prefix levels, we create a **spatial hierarchy**: exact location → neighborhood → district → city.

### Step 3: Historical Aggregate Features

```python
geo_ts_agg = (
    train.groupby(['geohash', 'time_slot'])['demand']
    .agg(['mean', 'std', 'median', 'min', 'max'])
    .reset_index()
)
```

**`groupby(['geohash', 'time_slot'])`**: Groups the data by unique combinations of geohash and time_slot. For example, all rows where `geohash='qp03xk'` AND `time_slot=36` (9:00 AM) are grouped together.

**`.agg(['mean', 'std', 'median', 'min', 'max'])`**: For each group, computes 5 statistics:
- `mean`: Average demand (the typical demand)
- `std`: Standard deviation (how variable the demand is)
- `median`: Middle value (robust to outliers)
- `min`/`max`: Extreme values

**`.reset_index()`**: After groupby, the group keys become the index. `reset_index()` turns them back into regular columns.

**Why these features?**: If location `qp03xk` had mean demand of 0.35 at 9:00 AM on day 48, it's very likely to have similar demand at 9:00 AM on day 49. This is the core prediction signal.

### Step 4: Road Type Statistics

```python
road_ts_agg = (
    train.groupby(['RoadType', 'time_slot'])['demand']
    .mean()
    .reset_index()
)
```

**What it captures**: "On average, what is the demand on Highways at 9:00 AM?" This interaction feature is powerful because Highways might peak at a different time than Residential roads.

---

## 🔗 CELL 5: Merging Features and Encoding Categoricals

### DataFrame Merging (Like SQL JOIN)

```python
df = df.merge(geo_ts_agg, on=['geohash', 'time_slot'], how='left')
```

**What `.merge()` does**: Joins two DataFrames on matching key columns — exactly like an SQL JOIN or Excel VLOOKUP.

- `on=['geohash', 'time_slot']`: Match rows where both geohash AND time_slot match
- `how='left'`: Keep ALL rows from the left DataFrame (`df`), and attach matching rows from right (`geo_ts_agg`). If no match is found, the columns from the right get `NaN` (not a number = missing).

**Example**:
```
Left DataFrame:          Right DataFrame (geo_ts_agg):
geohash  time_slot       geohash  time_slot  geo_ts_mean
qp03xk   36              qp03xk   36         0.350
qp08by   10              qp08by   10         0.021

After merge:
geohash  time_slot  geo_ts_mean
qp03xk   36         0.350
qp08by   10         0.021
```

### Categorical Encoding

Machine learning models can only work with **numbers**. Categorical text columns like `RoadType` must be converted to numbers.

```python
ROADTYPE_MAP = {'Residential': 0, 'Street': 1, 'Highway': 2}
df['RoadType_enc'] = df['RoadType'].map(ROADTYPE_MAP).fillna(-1)
```

**`.map(ROADTYPE_MAP)`**: Replaces each string value with its corresponding number using the dictionary.

**Why 0, 1, 2 (not random)?**: We ordered them by mean demand (Residential has lowest demand, Highway has highest). This creates a natural ordinal encoding that tree models can exploit.

**`.fillna(-1)`**: Missing values (NaN) become -1. For tree models, -1 acts as a signal: "this value was missing" — the model can learn to treat missing RoadType differently.

### Cyclical Time Encoding

```python
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
```

**Why sin/cos?** Consider hours 23 and 0. They're only 15 minutes apart in real life, but numerically they're 23 units apart. If we just feed `hour` as a number, the model thinks 23 and 0 are far away.

Sin/cos encoding wraps the 24-hour cycle into a circle:
- Hour 0: sin=0, cos=1 (right side of circle)
- Hour 6: sin=1, cos=0 (top of circle)
- Hour 12: sin=0, cos=-1 (left side of circle)
- Hour 18: sin=-1, cos=0 (bottom of circle)
- Hour 23: sin≈-0.26, cos≈0.97 (just before hour 0 — correctly close!)

You need BOTH sin AND cos together — one alone is ambiguous.

---

## 📊 CELL 6: Defining Features and Preparing Data

```python
FEATURES = [
    'geo_ts_mean', 'geo_ts_std', ...
    'RoadType_enc', 'Temperature', ...
]

X_train = train_feat[FEATURES].copy()
y_train = train_feat['demand'].copy()
X_test  = test_feat[FEATURES].copy()
```

**Convention**: In ML, `X` is the **features matrix** (inputs), `y` is the **target vector** (what we're predicting).

```python
X_train = X_train.fillna(-999)
```

**Why -999?**: Tree-based models (LightGBM, XGBoost) can use the value -999 as a proxy for "missing". When a tree splits on `geo_ts_mean`, it can create a branch specifically for values of -999 (i.e., "this geohash was never seen in training"). This is much better than filling with 0 or mean, which would give a misleading signal.

---

## 🤖 CELL 7: LightGBM — How It Works

### What is Gradient Boosting?

Imagine you're trying to predict traffic demand:

1. **Start**: Make a simple prediction (just the average demand = 0.094 for everyone)
2. **Look at errors**: Where were you wrong? Highway locations are way too low.
3. **Build Tree 1**: A small decision tree that corrects the worst mistakes
4. **Update prediction**: `new_pred = old_pred + 0.05 × Tree1_correction`
5. **Repeat**: Build Tree 2 to correct remaining errors, then Tree 3, etc.
6. **After 1000 trees**: The cumulative corrections give a very accurate prediction

Each tree "boosts" the performance of the previous ensemble. This is **gradient boosting** — "gradient" refers to how the corrections are computed mathematically (following the gradient of the loss function).

### Key Hyperparameters Explained

```python
lgb_params = {
    'num_leaves':       127,    # How complex each tree is (more leaves = more complex)
    'learning_rate':    0.05,   # How much each tree contributes (smaller = more trees needed but better)
    'n_estimators':     1000,   # Maximum number of trees to build
    'subsample':        0.8,    # Use only 80% of data rows per tree (prevents overfitting)
    'colsample_bytree': 0.8,    # Use only 80% of features per tree (prevents overfitting)
    'min_child_samples': 20,   # Minimum samples in a leaf (prevents overfitting to noise)
    'reg_alpha':        0.01,  # L1 regularization (pushes weights toward 0)
    'reg_lambda':       0.01,  # L2 regularization (keeps weights small)
}
```

**Overfitting**: When a model memorizes training data too well and doesn't generalize. Like a student who memorizes answers but can't solve new problems. `subsample`, `colsample_bytree`, `min_child_samples`, and regularization all fight overfitting.

**Learning rate**: A small learning rate (0.05) means each tree makes tiny corrections. Combined with many trees (1000), this gives smooth, accurate predictions. Large learning rate (0.5) makes big jumps — faster but less accurate.

### K-Fold Cross Validation

```python
N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
```

**What's happening**: The 77,299 training rows are split into 5 groups:

```
Fold 1: [VALIDATE] [TRAIN ] [TRAIN ] [TRAIN ] [TRAIN ]
Fold 2: [TRAIN ] [VALIDATE] [TRAIN ] [TRAIN ] [TRAIN ]
Fold 3: [TRAIN ] [TRAIN ] [VALIDATE] [TRAIN ] [TRAIN ]
Fold 4: [TRAIN ] [TRAIN ] [TRAIN ] [VALIDATE] [TRAIN ]
Fold 5: [TRAIN ] [TRAIN ] [TRAIN ] [TRAIN ] [VALIDATE]
```

- `train_idx`: Indices of rows to use for training in this fold
- `val_idx`: Indices of rows to use for validation in this fold
- `enumerate(..., 1)`: Like enumerate but starts counting from 1 instead of 0

**`.iloc[train_idx]`**: Selects rows by their integer positions. `iloc` = integer location.

**Why cross-validation?** We can't just train on all training data and check accuracy — the model would "cheat" by looking at the answers. CV gives us a reliable estimate of how well the model will perform on truly new data.

### Early Stopping

```python
callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
```

We set `n_estimators=1000` but we don't always need all 1000 trees. Early stopping watches the validation score and stops when it stops improving for 50 consecutive trees. This prevents overfitting and saves computation time.

### Out-of-Fold (OOF) Predictions

```python
lgb_oof_preds  = np.zeros(len(X_train))
lgb_test_preds = np.zeros(len(X_test))

# Inside the loop:
lgb_oof_preds[val_idx] = val_pred                    # Save validation predictions
lgb_test_preds += model.predict(X_test) / N_FOLDS    # Accumulate test predictions
```

**OOF predictions**: After all 5 folds, every training row has been in the validation set exactly once. So `lgb_oof_preds` contains predictions for all 77,299 training rows — each predicted by a model that NEVER saw that row. This is unbiased.

**Test predictions**: For the test set, we average predictions from all 5 trained models. Since each model saw 80% of the training data (4 out of 5 folds), this is essentially bagging — reducing variance through averaging.

```python
np.clip(val_pred, 0, 1)   # Demand must be in [0, 1]
```

---

## 🔬 CELL 8: Feature Importance

```python
lgb_importances = pd.DataFrame({
    'feature':    FEATURES,
    'importance': model.feature_importances_,
})
```

LightGBM tracks how often and by how much each feature was used for splitting in the trees. Features that are used frequently at high-level splits (near the root of trees) get high importance scores.

From our analysis, the top features were:
1. `geo_ts_mean` (historical demand at same location+time) — ~60% importance
2. `RoadType_enc` — ~38% importance

This confirms our intuition: knowing the historical demand at that exact spot, and what kind of road it is, tells you almost everything.

---

## 🚀 CELL 9: XGBoost — Why a Second Model?

XGBoost has different hyperparameters and a slightly different algorithm than LightGBM:

- **LightGBM**: Grows trees leaf-by-leaf (faster, sometimes too complex)
- **XGBoost**: Grows trees level-by-level (more stable, different error patterns)

```python
model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
```

**`**xgb_params`**: Python's "dictionary unpacking" — passes all key-value pairs as keyword arguments. `**{'a': 1, 'b': 2}` is equivalent to `a=1, b=2`.

---

## 🌲 CELL 10: HistGradientBoostingRegressor

This is sklearn's built-in gradient booster. It uses a histogram-based approach:
- Instead of computing exact splits, it bins continuous values into bins (e.g., 256 bins)
- Makes it much faster for large datasets
- Handles NaN values natively (doesn't need -999 trick)

```python
X_tr_hgb = X_tr.replace(-999, np.nan)   # Restore actual NaN for HistGBR
```

---

## 🎯 CELL 11: Ensemble — Combining Models

### Why Ensemble?

Each model makes different errors. By combining them, errors can cancel out:
- Model A is wrong on row 5 (predicts 0.3, actual is 0.5)
- Model B is correct on row 5 (predicts 0.52)
- Average: 0.41 — closer to actual than Model A alone

### Optimal Weight Search

```python
from scipy.optimize import minimize

def ensemble_objective(weights):
    w = np.array(weights) / np.array(weights).sum()  # Normalize to sum=1
    preds = w[0]*lgb_oof_preds + w[1]*xgb_oof_preds + w[2]*hgb_oof_preds
    return -r2_score(y_train, np.clip(preds, 0, 1))   # Negative because minimize() minimizes

result = minimize(ensemble_objective, [1.0, 1.0, 1.0], method='Nelder-Mead')
```

**`scipy.optimize.minimize`**: Finds the input values that minimize a function. We want to **maximize** R², so we minimize **negative** R².

**`Nelder-Mead`**: An optimization algorithm that doesn't need gradients — it works by exploring the "landscape" of the objective function with a "simplex" (like a multi-dimensional triangle), contracting around the minimum.

### R² Score Explained

**R² (R-squared)** is the main evaluation metric:

```
R² = 1 - (Sum of Squared Errors) / (Total Variance)
```

- **R² = 1.0**: Perfect — model predictions match actual exactly
- **R² = 0.0**: Model is no better than just predicting the mean
- **R² < 0**: Model is worse than predicting the mean
- **Score = max(0, 100 × R²)**: If R²=0.85, score=85. If R²<0, score=0.

---

## 📉 CELL 12: Residual Analysis

**Residuals** = actual - predicted. Analyzing them tells us where the model is wrong.

```python
residuals = y_train.values - ensemble_oof
```

Good properties of residuals:
- Centered around 0 (no systematic over- or under-prediction)
- Roughly normally distributed
- No clear pattern when plotted against predictions (random scatter)

If residuals are systematically high for certain predictions, it means the model is missing some pattern.

---

## 💾 CELL 13: Generating Submission

```python
submission = pd.DataFrame({
    'Index': test['Index'],
    'demand': final_preds
})

submission.to_csv('./submission.csv', index=False)
```

**`pd.DataFrame({'col1': values1, 'col2': values2})`**: Creates a DataFrame from a dictionary. Each key becomes a column name, each value becomes the column's data.

**`index=False`**: When saving to CSV, don't write the DataFrame's row numbers. The submission format only wants `Index` and `demand` columns.

---

## ✅ CELL 14: Verification

```python
test_indices = set(test['Index'].values)
our_indices  = set(our_sub['Index'].values)
print(f'All match: {test_indices == our_indices}')
```

**`set()`**: A set is an unordered collection of unique values. Comparing two sets checks if they contain exactly the same elements (regardless of order). This verifies every test row has a prediction.

---

## 🏃 How to Run the Code

### Step 1: Set Up Your Environment

Make sure you have Python and the required packages:

```bash
# Install packages
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost jupyter
```

### Step 2: Organize Your Files

Your folder structure should look like:
```
Flipkar-Gridlock/
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── traffic_demand_prediction.ipynb   ← The notebook
├── approach_explanation.md
└── code_explanation.md
```

### Step 3: Launch Jupyter Notebook

```bash
# Navigate to your project folder
cd /path/to/Flipkar-Gridlock

# Launch Jupyter
jupyter notebook
```

This opens a browser at `http://localhost:8888`. Click on `traffic_demand_prediction.ipynb`.

### Step 4: Run the Notebook

**Option A — Run all cells automatically**:
- Click `Kernel` in the menu
- Click `Restart & Run All`
- Wait ~10-15 minutes (the model training takes time)

**Option B — Run cell by cell**:
- Click on a cell
- Press `Shift + Enter` to run it and move to the next
- Repeat for each cell

### Step 5: Check the Output

After all cells run:
1. A file called `submission.csv` will appear in your project folder
2. The last cells will show:
   - Model R² scores
   - Verification that the submission is correctly formatted
   - A preview of the predictions

### Step 6: Understand the Progress

While running, look for these outputs:
```
Fold 1/5... R² = 0.8234, Score = 82.34
Fold 2/5... R² = 0.8156, Score = 81.56
...
Ensemble OOF Score: 84.12
```

Higher scores = better model.

---

## 📤 How to Submit

### Step 1: Find your submission file

After running the notebook, you'll have `submission.csv` in your project folder.

### Step 2: Verify the file

Open it in Excel or any text editor. It should look like:
```
Index,demand
0,0.0812
1,0.0523
2,0.3145
...
```
- Should have exactly **41,778 data rows** (+ 1 header row)
- Should have exactly **2 columns**: `Index` and `demand`
- `demand` values should be between 0 and 1

### Step 3: Go to the competition page

1. Open the Flipkart Gridlock hackathon page
2. Scroll down to the **Upload File** section

### Step 4: Upload predictions

1. Click **Upload File** under the **"Upload File"** section
2. Select your `submission.csv` file
3. Wait for confirmation

### Step 5: Upload source code

1. Click **Upload File** under the **"Upload Source Code"** section
2. Upload your `traffic_demand_prediction.ipynb` file
3. Optionally upload this markdown documentation

### Step 6: Add comments

In the **"Your Answer"** text box, you can describe your approach:
> "Used LightGBM + XGBoost + HistGBR ensemble with geo-temporal historical demand features, 5-fold cross-validation, and optimal weight search."

### Step 7: Submit

Click the **Submit** button and wait for your score!

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'lightgbm'"
```bash
pip install lightgbm
```

### "FileNotFoundError: train.csv not found"
Update the `DATA_PATH` variable in Cell 2:
```python
DATA_PATH = '/full/path/to/your/dataset/'  # e.g., '/Users/yourname/Downloads/dataset/'
```

### Notebook runs very slowly
The model training takes 5-15 minutes. If it's much slower:
- Reduce `n_estimators` from 1000 to 300 in the model parameters
- Reduce `N_FOLDS` from 5 to 3

### R² score is very low (< 0.5)
Check that `DATA_PATH` points to the correct folder with the actual dataset files.

---

## 📚 Key ML Concepts Summary

| Concept | Simple Explanation |
|---------|-------------------|
| **Feature** | An input variable the model uses to make predictions |
| **Target** | What we're predicting (demand) |
| **R²** | How good the predictions are (1 = perfect, 0 = useless) |
| **Gradient Boosting** | Build many small decision trees, each correcting previous errors |
| **Cross-Validation** | Split data multiple ways to get reliable performance estimates |
| **Overfitting** | Model memorizes training data but fails on new data |
| **Ensemble** | Combining multiple models for better predictions than any single model |
| **Feature Engineering** | Transforming raw data into informative inputs for the model |
| **Hyperparameter** | Settings that control model behavior (learning rate, tree depth, etc.) |
| **OOF Predictions** | Predictions on training data made by a model that never saw those rows |


41775,0.003361347942345856
41776,0.08616682105481974
41777,0.0035596619983268058