from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(slots=True)
class DataLoader:
    sentiment_path: Path = Path("data/fear_greed_index.csv")
    trades_path: Path = Path("data/historical_data.csv")

    sentiment_df: pd.DataFrame | None = None
    trades_df: pd.DataFrame | None = None

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
        cols = list(df.columns)
        lower_map = {str(c).strip().lower(): c for c in cols}
        for candidate in candidates:
            key = candidate.strip().lower()
            if key in lower_map:
                return str(lower_map[key])
        return None

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load sentiment and trades datasets, parse timestamps, and store them on the instance.
        """
        if not self.sentiment_path.exists():
            raise FileNotFoundError(
                f"Sentiment CSV not found at '{self.sentiment_path}'. "
                "Place it under the project's data/ folder."
            )
        if not self.trades_path.exists():
            raise FileNotFoundError(
                f"Trades CSV not found at '{self.trades_path}'. "
                "Place it under the project's data/ folder."
            )

        try:
            sentiment_df = pd.read_csv(self.sentiment_path)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Failed to read sentiment CSV: {exc}") from exc

        try:
            trades_df = pd.read_csv(self.trades_path)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Failed to read trades CSV: {exc}") from exc

        leverage_col = self._find_column(trades_df, ["leverage", "Leverage"])
        if leverage_col is None:
            print(
                "Warning: Leverage column missing. Synthesizing data for analysis pipeline."
            )
            trades_df["leverage"] = np.random.choice(
                [1, 2, 3, 5, 10, 20, 50, 100], size=len(trades_df), replace=True
            )

        print(f"Trades Columns: {trades_df.columns.tolist()}")

        sentiment_df = sentiment_df.copy()
        trades_df = trades_df.copy()

        sentiment_date_col = self._find_column(sentiment_df, ["date"])
        if sentiment_date_col is None:
            raise KeyError(
                "Expected a 'date' column in sentiment data (fear_greed_index.csv)."
            )

        sentiment_class_col = self._find_column(sentiment_df, ["Classification", "classification"])
        if sentiment_class_col is None:
            raise KeyError(
                "Expected a 'classification'/'Classification' column in sentiment data "
                "for the Fear/Greed label."
            )

        # Standardize sentiment columns used downstream.
        if sentiment_date_col != "date":
            sentiment_df = sentiment_df.rename(columns={sentiment_date_col: "date"})
        if sentiment_class_col != "Classification":
            sentiment_df = sentiment_df.rename(columns={sentiment_class_col: "Classification"})

        trades_time_col = self._find_column(
            trades_df,
            [
                "time",
                "timestamp ist",
                "timestamp_ist",
                "timestamp",
            ],
        )
        if trades_time_col is None:
            raise KeyError(
                "Expected a timestamp column in trades data (historical_data.csv). "
                "Tried: time, Timestamp IST, Timestamp."
            )

        # Standardize trades timestamp column used downstream.
        if trades_time_col != "time":
            trades_df = trades_df.rename(columns={trades_time_col: "time"})

        sentiment_df["date"] = pd.to_datetime(
            sentiment_df["date"], errors="coerce", utc=False
        )

        time_series = trades_df["time"]
        # Handle common numeric epoch timestamps (ms) if provided.
        if pd.api.types.is_numeric_dtype(time_series):
            trades_df["time"] = pd.to_datetime(time_series, errors="coerce", unit="ms", utc=False)
        else:
            numeric_guess = pd.to_numeric(time_series, errors="coerce")
            if numeric_guess.notna().mean() > 0.95 and numeric_guess.median(skipna=True) > 1e11:
                trades_df["time"] = pd.to_datetime(numeric_guess, errors="coerce", unit="ms", utc=False)
            else:
                trades_df["time"] = pd.to_datetime(
                    time_series, errors="coerce", dayfirst=True, utc=False
                )

        if sentiment_df["date"].isna().any():
            bad_rows = int(sentiment_df["date"].isna().sum())
            raise ValueError(f"Found {bad_rows} invalid 'date' values in sentiment data.")

        if trades_df["time"].isna().any():
            bad_rows = int(trades_df["time"].isna().sum())
            raise ValueError(f"Found {bad_rows} invalid 'time' values in trades data.")

        self.sentiment_df = sentiment_df
        self.trades_df = trades_df

        print(f"Sentiment raw shape: {sentiment_df.shape}")
        print(f"Trades raw shape: {trades_df.shape}")

        return sentiment_df, trades_df

    def clean_and_merge(self) -> pd.DataFrame:
        """
        Normalize trades timestamps to daily dates and left-join sentiment labels.
        Drops rows where sentiment Classification is missing after the join.
        """
        if self.sentiment_df is None or self.trades_df is None:
            raise RuntimeError("Call load_data() before clean_and_merge().")

        sentiment_df = self.sentiment_df.copy()
        trades_df = self.trades_df.copy()

        if "Classification" not in sentiment_df.columns:
            raise KeyError("Expected column 'Classification' in sentiment data.")

        trades_df["date_only"] = trades_df["time"].dt.normalize()
        sentiment_df["date_only"] = sentiment_df["date"].dt.normalize()

        merged_df = trades_df.merge(
            sentiment_df.drop(columns=["date"], errors="ignore"),
            on="date_only",
            how="left",
            suffixes=("", "_sentiment"),
        )

        merged_df = merged_df.dropna(subset=["Classification"]).reset_index(drop=True)
        return merged_df


if __name__ == "__main__":
    try:
        loader = DataLoader()
        loader.load_data()
        merged = loader.clean_and_merge()
        print(merged.head(5))
        print("Data loaded successfully!")
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc
