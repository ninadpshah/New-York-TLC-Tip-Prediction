# NYC Taxi Tip Prediction

A deep learning regression model that predicts taxi tip amounts using 5.6M+ NYC Yellow Taxi trip records from the [TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). The model uses temporal and trip features to estimate tips, achieving an RMSE of ~$2.62 on held-out test data.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Approach](#approach)
- [Results](#results)
- [Getting Started](#getting-started)
- [Technologies](#technologies)

## Problem Statement

Predicting taxi tip amounts is valuable for driver income estimation, fare transparency, and urban transportation analytics. This project frames the problem as a **regression task**: given trip metadata (distance, fare, time of day, rate code, payment type), predict the tip a passenger will leave.

Key challenges addressed:
- **Scale**: Processing 5.6M+ records efficiently with GPU-accelerated training
- **Noise**: Handling outliers and skewed distributions via Z-score filtering
- **Feature Engineering**: Extracting cyclical temporal patterns from raw timestamps

## Dataset

| Property | Value |
|----------|-------|
| **Source** | [NYC TLC Yellow Taxi Trip Records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) |
| **Period** | August - September 2023 |
| **Raw Records** | ~6M+ trips |
| **After Cleaning** | 5,670,178 trips |
| **Features** | 13 input features (5 original + 8 engineered) |
| **Target** | `tip_amount` (continuous, USD) |

### Feature Descriptions

| Feature | Type | Description |
|---------|------|-------------|
| `trip_distance` | Float | Trip distance in miles |
| `fare_amount` | Float | Base fare amount in USD |
| `RatecodeID` | Categorical | Rate code (standard, JFK, Newark, etc.) |
| `payment_type` | Categorical | Payment method (credit card, cash, etc.) |
| `Airport_fee` | Categorical | Airport surcharge indicator |
| `pickup_weekday` | Integer | Day of week for pickup (0=Mon, 6=Sun) |
| `pickup_hour` | Integer | Hour of pickup (0-23) |
| `pickup_minute` | Integer | Minute of pickup (0-59) |
| `pickup_week_hour` | Integer | Composite: weekday * 24 + hour |
| `dropoff_weekday` | Integer | Day of week for dropoff |
| `dropoff_hour` | Integer | Hour of dropoff |
| `dropoff_minute` | Integer | Minute of dropoff |
| `dropoff_week_hour` | Integer | Composite: weekday * 24 + hour |

## Project Structure

```
New-York-TLC-Tip-Prediction/
├── README.md
├── requirements.txt
├── TLC_Trip_Data.ipynb              # Main notebook: EDA, training, evaluation
├── TLC_NYC_Trip_Dataset_DSE598_Final_Report.pdf
└── src/
    ├── __init__.py
    ├── data_loader.py               # Data loading and downloading utilities
    ├── preprocessing.py             # Feature engineering, outlier removal, encoding
    ├── model.py                     # Neural network architecture and training
    └── visualization.py             # All plotting and EDA visualizations
```

## Approach

### 1. Data Preprocessing

- **Feature Selection**: Selected 6 raw features most relevant to tipping behavior based on domain knowledge
- **Temporal Feature Engineering**: Extracted weekday, hour, minute, and composite week-hour features from pickup/dropoff timestamps to capture cyclical commuter patterns
- **Outlier Removal**: Applied Z-score filtering (|z| < 3) on `trip_distance` and `fare_amount` to remove anomalous records
- **Encoding**: Label-encoded categorical variables (`RatecodeID`, `payment_type`, `Airport_fee`)
- **Scaling**: Standardized all features using `StandardScaler` (fit on training set only to prevent data leakage)

### 2. Model Architecture

```
Input (13 features)
  -> Dense(128, ReLU)
  -> Dropout(0.5)
  -> Dense(64, ReLU)
  -> Dense(1)            # Linear output for regression
```

- **Optimizer**: Adam (lr=0.001)
- **Loss**: Mean Squared Error
- **Regularization**: 50% dropout to reduce overfitting on 5.6M samples
- **Training**: 50 epochs, batch size 8,196, 20% validation split, GPU-accelerated

### 3. Evaluation

The model is evaluated on a 30% held-out test set (~1.7M records) with the following metrics:

| Metric | Value |
|--------|-------|
| **Test MSE** | 6.86 |
| **Test RMSE** | ~$2.62 |
| **Training MSE** (final epoch) | 6.06 |

## Results

### Training Convergence

The model converges within ~20 epochs, with training loss decreasing from 8.71 to 6.06. Validation loss stabilizes around 6.30, showing the model generalizes beyond the training data.

### Key Findings

- **Fare amount** is the strongest predictor of tip amount, which aligns with the common practice of percentage-based tipping
- **Payment type** is highly correlated with tipping: credit card payments show significantly higher tips than cash transactions
- **Temporal features** (pickup hour, weekday) capture commuter vs. leisure trip patterns that influence tipping behavior
- **Trip distance** provides moderate predictive power, partially collinear with fare amount

### Visualizations

The notebook includes:
- Distribution analysis of the target variable (tip amount)
- Correlation heatmaps across all features
- Pairwise scatter plots of key feature relationships
- True vs. predicted tip comparison plots
- Prediction density (KDE) distributions

## Getting Started

### Prerequisites

- Python 3.8+
- GPU recommended for training (CUDA-compatible)

### Installation

```bash
git clone https://github.com/ninadpshah/New-York-TLC-Tip-Prediction.git
cd New-York-TLC-Tip-Prediction
pip install -r requirements.txt
```

### Download Data

Download the Yellow Taxi trip data from the [NYC TLC website](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page):
- `yellow_tripdata_2023-08.parquet`
- `yellow_tripdata_2023-09.parquet`

Place the files in a `data/` directory or update the paths in the notebook/source code.

### Run the Notebook

```bash
jupyter notebook TLC_Trip_Data.ipynb
```

Or open directly in Google Colab using the badge at the top of the notebook.

### Run as Python Modules

```python
from src.data_loader import load_tlc_data
from src.preprocessing import preprocess_pipeline
from src.model import build_model, train_model

# Load and preprocess
df = load_tlc_data("data/yellow_tripdata_2023-08.parquet",
                    "data/yellow_tripdata_2023-09.parquet")
X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(df)

# Train
model = build_model(input_dim=X_train.shape[1])
history = train_model(model, X_train, y_train)

# Evaluate
loss = model.evaluate(X_test, y_test)
print(f"Test MSE: {loss:.4f}, RMSE: {loss**0.5:.4f}")
```

## Technologies

| Category | Tools |
|----------|-------|
| **Language** | Python 3 |
| **Deep Learning** | TensorFlow / Keras |
| **Data Processing** | pandas, NumPy, scikit-learn |
| **Visualization** | matplotlib, seaborn |
| **Statistics** | SciPy |
| **Environment** | Google Colab (GPU), Jupyter |
