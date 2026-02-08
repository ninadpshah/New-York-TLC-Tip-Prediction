"""Feature engineering, outlier removal, and preprocessing pipeline."""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


SELECTED_FEATURES = [
    "trip_distance",
    "RatecodeID",
    "fare_amount",
    "payment_type",
    "Airport_fee",
    "tip_amount",
]

CATEGORICAL_COLUMNS = ["RatecodeID", "payment_type", "Airport_fee"]

OUTLIER_COLUMNS = ["trip_distance", "fare_amount"]


def engineer_temporal_features(df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from pickup and dropoff timestamps.

    Creates weekday, hour, minute, and composite week-hour features
    that capture cyclical commuter and time-of-day patterns.

    Args:
        df: Working DataFrame to add features to.
        data: Original DataFrame containing datetime columns.

    Returns:
        DataFrame with added temporal features.
    """
    df = df.copy()

    for prefix, col in [("pickup", "tpep_pickup_datetime"),
                         ("dropoff", "tpep_dropoff_datetime")]:
        dt = pd.to_datetime(data[col])
        df[f"{prefix}_weekday"] = dt.dt.weekday
        df[f"{prefix}_hour"] = dt.dt.hour
        df[f"{prefix}_minute"] = dt.dt.minute
        df[f"{prefix}_week_hour"] = df[f"{prefix}_weekday"] * 24 + df[f"{prefix}_hour"]

    return df


def remove_outliers_zscore(df: pd.DataFrame, columns: list[str],
                           threshold: float = 3.0) -> pd.DataFrame:
    """Remove rows with Z-score outliers in specified columns.

    Args:
        df: Input DataFrame.
        columns: Columns to check for outliers.
        threshold: Z-score threshold (default 3.0 = 3 sigma).

    Returns:
        Filtered DataFrame with outliers removed.
    """
    z_scores = np.abs(stats.zscore(df[columns]))
    mask = (z_scores < threshold).all(axis=1)
    return df[mask]


def encode_categoricals(df: pd.DataFrame,
                        columns: list[str]) -> tuple[pd.DataFrame, dict]:
    """Label-encode categorical columns.

    Args:
        df: Input DataFrame.
        columns: Columns to encode.

    Returns:
        Tuple of (encoded DataFrame, dict of fitted LabelEncoders).
    """
    df = df.copy()
    encoders = {}
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders


def preprocess_pipeline(
    df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, StandardScaler]:
    """Full preprocessing pipeline: feature engineering -> cleaning -> split -> scale.

    Args:
        df: Raw combined trip DataFrame.
        test_size: Fraction of data reserved for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, y_train, y_test, scaler).
    """
    # Select features and engineer temporal columns
    filtered = df[SELECTED_FEATURES].copy()
    filtered = engineer_temporal_features(filtered, df)

    # Remove outliers
    filtered = remove_outliers_zscore(filtered, OUTLIER_COLUMNS)

    # Encode categoricals
    filtered, _ = encode_categoricals(filtered, CATEGORICAL_COLUMNS)

    # Split features and target
    X = filtered.drop("tip_amount", axis=1)
    y = filtered["tip_amount"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Standardize (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
