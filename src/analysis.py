from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class Analyzer:
    df: pd.DataFrame

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
        cols = list(df.columns)
        lower_map = {str(c).strip().lower(): c for c in cols}
        for candidate in candidates:
            key = candidate.strip().lower()
            if key in lower_map:
                return str(lower_map[key])
        return None

    def _resolve_columns(self) -> dict[str, str]:
        date_col = self._find_column(self.df, ["date", "date_only"])
        if date_col is None:
            time_col = self._find_column(self.df, ["time", "timestamp", "timestamp ist"])
            if time_col is None:
                raise KeyError("Expected a date column ('date'/'date_only') or a timestamp column.")
            date_col = "__derived_date__"

        class_col = self._find_column(self.df, ["Classification", "classification"])
        if class_col is None:
            raise KeyError("Expected sentiment label column 'Classification'.")

        pnl_col = self._find_column(self.df, ["closedPnL", "closed_pnl", "Closed PnL", "closed pnl"])
        if pnl_col is None:
            raise KeyError("Expected PnL column 'closedPnL' (or 'Closed PnL').")

        leverage_col = self._find_column(self.df, ["leverage", "Leverage"])
        account_col = self._find_column(self.df, ["account", "Account", "user_id", "User ID"])
        if account_col is None:
            raise KeyError("Expected trader id column 'account' (or 'Account').")

        cols: dict[str, str] = {
            "date": date_col,
            "classification": class_col,
            "pnl": pnl_col,
            "account": account_col,
        }
        if leverage_col is not None:
            cols["leverage"] = leverage_col
        return cols

    def calculate_daily_metrics(self) -> pd.DataFrame:
        """
        Group by date and sentiment Classification and compute daily trade performance metrics.

        Returns a DataFrame with:
        - avg_pnl: mean of closed PnL
        - win_rate: fraction of trades with positive PnL
        - trade_count: number of trades
        - avg_leverage: mean of leverage (NaN if leverage not available)
        """
        if self.df.empty:
            raise ValueError("Analyzer input DataFrame is empty.")

        cols = self._resolve_columns()
        df = self.df.copy()

        if cols["date"] == "__derived_date__":
            time_col = self._find_column(df, ["time", "timestamp", "timestamp ist"])
            df["__derived_date__"] = pd.to_datetime(df[time_col], errors="coerce").dt.normalize()

        df[cols["date"]] = pd.to_datetime(df[cols["date"]], errors="coerce").dt.normalize()
        if df[cols["date"]].isna().any():
            bad = int(df[cols["date"]].isna().sum())
            raise ValueError(f"Found {bad} invalid date values; cannot compute daily metrics.")

        df[cols["pnl"]] = pd.to_numeric(df[cols["pnl"]], errors="coerce")

        group_keys = [cols["date"], cols["classification"]]

        def _win_rate(pnl: pd.Series) -> float:
            total = int(pnl.shape[0])
            if total == 0:
                return float("nan")
            wins = int((pnl > 0).sum())
            return wins / total

        aggregated = (
            df.groupby(group_keys, dropna=False)
            .agg(
                avg_pnl=(cols["pnl"], "mean"),
                win_rate=(cols["pnl"], _win_rate),
                trade_count=(cols["pnl"], "size"),
                avg_leverage=(cols.get("leverage", cols["pnl"]), "mean"),
            )
            .reset_index()
        )

        if "leverage" not in cols:
            aggregated["avg_leverage"] = np.nan

        aggregated = aggregated.rename(
            columns={cols["date"]: "date", cols["classification"]: "Classification"}
        )
        return aggregated

    def segment_traders(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Segment traders by average leverage:
        - High Leverage: top 25% by average leverage
        - Low Leverage: bottom 25% by average leverage

        Returns (high_lev_traders, low_lev_traders) with columns:
        - account
        - avg_leverage
        - total_pnl
        """
        if self.df.empty:
            raise ValueError("Analyzer input DataFrame is empty.")

        cols = self._resolve_columns()
        if "leverage" not in cols:
            raise KeyError(
                "Cannot segment traders: leverage column not found in merged data."
            )

        df = self.df.copy()
        df[cols["pnl"]] = pd.to_numeric(df[cols["pnl"]], errors="coerce")
        df[cols["leverage"]] = pd.to_numeric(df[cols["leverage"]], errors="coerce")

        trader_stats = (
            df.groupby(cols["account"], dropna=False)
            .agg(
                avg_leverage=(cols["leverage"], "mean"),
                total_pnl=(cols["pnl"], "sum"),
            )
            .reset_index()
            .rename(columns={cols["account"]: "account"})
        )

        lev = trader_stats["avg_leverage"].dropna()
        if lev.empty:
            raise ValueError("Leverage column exists but contains no numeric values.")

        low_thresh = float(lev.quantile(0.25))
        high_thresh = float(lev.quantile(0.75))

        low_lev_traders = trader_stats[trader_stats["avg_leverage"] <= low_thresh].copy()
        high_lev_traders = trader_stats[trader_stats["avg_leverage"] >= high_thresh].copy()

        return high_lev_traders.reset_index(drop=True), low_lev_traders.reset_index(drop=True)


if __name__ == "__main__":
    try:
        # Works for `python src/analysis.py`
        from data_loader import DataLoader  # type: ignore
    except Exception:  # pragma: no cover
        # Works for `python -m src.analysis`
        from src.data_loader import DataLoader  # type: ignore

    try:
        loader = DataLoader()
        loader.load_data()
        merged_df = loader.clean_and_merge()

        analyzer = Analyzer(merged_df)

        daily_metrics = analyzer.calculate_daily_metrics()
        print("\nFear vs Greed (daily metrics):")
        print(
            daily_metrics.groupby("Classification")[["avg_pnl", "win_rate", "trade_count"]]
            .mean(numeric_only=True)
            .sort_index()
        )

        try:
            high_lev, low_lev = analyzer.segment_traders()
            print(f"\nHigh Leverage traders: {high_lev['account'].nunique()}")
            print(f"Low Leverage traders: {low_lev['account'].nunique()}")
        except KeyError as exc:
            print(f"\nSkipping leverage segmentation: {exc}")

    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc

