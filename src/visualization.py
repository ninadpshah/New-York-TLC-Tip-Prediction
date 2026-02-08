"""Visualization utilities for EDA and model evaluation."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_target_distribution(df: pd.DataFrame, column: str = "tip_amount") -> None:
    """Plot histogram with KDE overlay for the target variable."""
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], bins=30, kde=True)
    plt.xlabel("Tip Amount ($)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Tip Amount")
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Heatmap",
                              figsize: tuple = (10, 8)) -> None:
    """Plot annotated correlation heatmap."""
    plt.figure(figsize=figsize)
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_pairplot(df: pd.DataFrame,
                  features: list[str] | None = None,
                  sample_size: int = 5000) -> None:
    """Plot pairwise scatter plots for selected features."""
    if features is None:
        features = ["trip_distance", "fare_amount", "pickup_weekday",
                     "pickup_hour", "tip_amount"]
    sample = df.sample(min(sample_size, len(df)))
    sns.pairplot(sample, vars=features, diag_kind="kde")
    plt.suptitle("Pairplot of Selected Features", y=1.02)
    plt.show()


def plot_prediction_kde(predictions_df: pd.DataFrame, sample_size: int = 9999) -> None:
    """Plot KDE and cumulative KDE of predictions vs true values."""
    sample = predictions_df.sample(min(sample_size, len(predictions_df)))
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    for ax in axs:
        ax.set(xlim=[-3, 20])

    sns.kdeplot(data=sample, ax=axs[0], bw_adjust=3)
    axs[0].set_title("Prediction Density")

    sns.kdeplot(data=sample, ax=axs[1], bw_adjust=3, cumulative=True)
    axs[1].set_title("Cumulative Prediction Density")

    fig.tight_layout()
    plt.show()


def plot_true_vs_predicted(X_test: pd.DataFrame, y_test, predictions: np.ndarray) -> None:
    """Scatter plots comparing true and predicted tip amounts."""
    viz = pd.DataFrame({
        "Trip Distance": X_test["trip_distance"].values,
        "Fare Amount": X_test["fare_amount"].values,
        "True Tip Amount": y_test.values if hasattr(y_test, "values") else y_test,
        "Predicted Tip Amount": predictions.flatten(),
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, feature in zip(axes, ["Trip Distance", "Fare Amount"]):
        ax.scatter(viz[feature], viz["True Tip Amount"],
                   alpha=0.3, s=5, label="True")
        ax.scatter(viz[feature], viz["Predicted Tip Amount"],
                   alpha=0.3, s=5, label="Predicted")
        ax.set_xlabel(feature)
        ax.set_ylabel("Tip Amount ($)")
        ax.set_title(f"{feature} vs Tip Amount")
        ax.legend()

    fig.tight_layout()
    plt.show()
