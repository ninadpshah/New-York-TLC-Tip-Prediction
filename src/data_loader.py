"""Data loading utilities for NYC TLC Yellow Taxi trip data."""

import pandas as pd


def load_tlc_data(*file_paths: str, parse_dates: bool = True) -> pd.DataFrame:
    """Load and concatenate one or more TLC trip data files.

    Supports both CSV and Parquet formats. Files are auto-detected by extension.

    Args:
        *file_paths: Paths to trip data files (CSV or Parquet).
        parse_dates: Whether to parse pickup/dropoff datetime columns.

    Returns:
        Combined DataFrame with all trip records.
    """
    date_cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
    frames = []

    for path in file_paths:
        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(
                path,
                parse_dates=date_cols if parse_dates else False,
                low_memory=False,
            )
        frames.append(df)

    return pd.concat(frames, ignore_index=True)
