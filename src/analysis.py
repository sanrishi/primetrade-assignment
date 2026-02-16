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
        size_usd_col = self._find_column(
            self.df,
            [
                "Size USD",
                "size usd",
                "size_usd",
                "sizeusd",
            ],
        )
        side_col = self._find_column(
            self.df,
            [
                "Side",
                "side",
                "Direction",
                "direction",
            ],
        )
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
        if size_usd_col is not None:
            cols["size_usd"] = size_usd_col
        if side_col is not None:
            cols["side"] = side_col
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
        if "size_usd" in cols:
            df[cols["size_usd"]] = pd.to_numeric(df[cols["size_usd"]], errors="coerce")

        if "side" in cols:
            side = df[cols["side"]].astype(str).str.lower()
            df["__is_long__"] = side.str.contains("buy|long", na=False).astype("int64")
            df["__is_short__"] = side.str.contains("sell|short", na=False).astype("int64")
        else:
            df["__is_long__"] = 0
            df["__is_short__"] = 0

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
                avg_trade_size_usd=(cols.get("size_usd", cols["pnl"]), "mean"),
                long_count=("__is_long__", "sum"),
                short_count=("__is_short__", "sum"),
            )
            .reset_index()
        )

        if "leverage" not in cols:
            aggregated["avg_leverage"] = np.nan
        if "size_usd" not in cols:
            aggregated["avg_trade_size_usd"] = np.nan

        denom = aggregated["short_count"].to_numpy(dtype="float64")
        num = aggregated["long_count"].to_numpy(dtype="float64")
        aggregated["long_short_ratio"] = np.divide(
            num, denom, out=np.full_like(num, np.nan, dtype="float64"), where=denom != 0
        )
        total_dir = (aggregated["long_count"] + aggregated["short_count"]).to_numpy(dtype="float64")
        aggregated["long_share"] = np.divide(
            num,
            total_dir,
            out=np.full_like(num, np.nan, dtype="float64"),
            where=total_dir != 0,
        )

        aggregated = aggregated.rename(
            columns={cols["date"]: "date", cols["classification"]: "Classification"}
        )
        return aggregated

    def segment_profitability(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Profitability segment:
        - Consistent Winners: positive lifetime PnL
        - Net Losers: non-positive lifetime PnL
        """
        if self.df.empty:
            raise ValueError("Analyzer input DataFrame is empty.")

        cols = self._resolve_columns()
        df = self.df.copy()
        df[cols["pnl"]] = pd.to_numeric(df[cols["pnl"]], errors="coerce")

        trader_pnl = (
            df.groupby(cols["account"], dropna=False)
            .agg(total_pnl=(cols["pnl"], "sum"))
            .reset_index()
            .rename(columns={cols["account"]: "account"})
        )

        winners = trader_pnl[trader_pnl["total_pnl"] > 0].copy().reset_index(drop=True)
        losers = trader_pnl[trader_pnl["total_pnl"] <= 0].copy().reset_index(drop=True)
        return winners, losers

    def segment_activity(
        self, trades_per_day_threshold: float = 5.0
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Activity segment based on average trades per active day per account:
        - High Frequency: avg_trades_per_day > trades_per_day_threshold
        - Low Frequency: avg_trades_per_day <= trades_per_day_threshold
        """
        if self.df.empty:
            raise ValueError("Analyzer input DataFrame is empty.")

        cols = self._resolve_columns()
        df = self.df.copy()

        date_col = cols["date"]
        if date_col == "__derived_date__":
            time_col = self._find_column(df, ["time", "timestamp", "timestamp ist"])
            df["__derived_date__"] = pd.to_datetime(df[time_col], errors="coerce").dt.normalize()
            date_col = "__derived_date__"
        else:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

        per_day = (
            df.groupby([cols["account"], date_col], dropna=False)
            .size()
            .reset_index(name="trades")
        )

        activity = (
            per_day.groupby(cols["account"], dropna=False)
            .agg(
                avg_trades_per_day=("trades", "mean"),
                total_trades=("trades", "sum"),
                active_days=("trades", "size"),
            )
            .reset_index()
            .rename(columns={cols["account"]: "account"})
        )

        high = activity[activity["avg_trades_per_day"] > trades_per_day_threshold].copy()
        low = activity[activity["avg_trades_per_day"] <= trades_per_day_threshold].copy()
        return high.reset_index(drop=True), low.reset_index(drop=True)

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

        winners, losers = analyzer.segment_profitability()
        print(f"\nConsistent Winners: {winners['account'].nunique()}")
        print(f"Net Losers: {losers['account'].nunique()}")

        high_freq, low_freq = analyzer.segment_activity(trades_per_day_threshold=5.0)
        print(f"\nHigh Frequency traders (>5 trades/day): {high_freq['account'].nunique()}")
        print(f"Low Frequency traders (<=5 trades/day): {low_freq['account'].nunique()}")

        # Generate reproducible artifacts (HTML charts + CSV tables) for the submission.
        try:
            from pathlib import Path

            import plotly.express as px

            out_charts = Path("output/charts")
            out_tables = Path("output/tables")
            out_charts.mkdir(parents=True, exist_ok=True)
            out_tables.mkdir(parents=True, exist_ok=True)

            dm = daily_metrics.copy()
            dm["sentiment_bucket"] = "Other"
            dm.loc[
                dm["Classification"].astype(str).str.contains("fear", case=False, na=False),
                "sentiment_bucket",
            ] = "Fear"
            dm.loc[
                dm["Classification"].astype(str).str.contains("greed", case=False, na=False),
                "sentiment_bucket",
            ] = "Greed"

            fg = (
                dm[dm["sentiment_bucket"].isin(["Fear", "Greed"])]
                .groupby("sentiment_bucket", as_index=False)
                .agg(
                    win_rate=("win_rate", "mean"),
                    avg_pnl=("avg_pnl", "mean"),
                    avg_trade_size_usd=("avg_trade_size_usd", "mean"),
                    avg_trades_per_day=("trade_count", "mean"),
                    long_count=("long_count", "sum"),
                    short_count=("short_count", "sum"),
                    long_share=("long_share", "mean"),
                )
                .sort_values("sentiment_bucket")
            )
            fg["long_short_ratio_total"] = np.divide(
                fg["long_count"].to_numpy(dtype="float64"),
                fg["short_count"].to_numpy(dtype="float64"),
                out=np.full(len(fg), np.nan, dtype="float64"),
                where=fg["short_count"].to_numpy(dtype="float64") != 0,
            )
            fg["long_share_total"] = np.divide(
                fg["long_count"].to_numpy(dtype="float64"),
                (fg["long_count"] + fg["short_count"]).to_numpy(dtype="float64"),
                out=np.full(len(fg), np.nan, dtype="float64"),
                where=(fg["long_count"] + fg["short_count"]).to_numpy(dtype="float64") != 0,
            )
            fg.to_csv(out_tables / "fear_vs_greed_summary.csv", index=False)
            (out_tables / "fear_vs_greed_summary.json").write_text(
                fg.to_json(orient="records", indent=2),
                encoding="utf-8",
            )

            px.bar(fg, x="sentiment_bucket", y="win_rate", title="Win Rate: Fear vs Greed").write_html(
                out_charts / "win_rate_fear_vs_greed.html"
            )
            px.bar(fg, x="sentiment_bucket", y="avg_pnl", title="Avg PnL: Fear vs Greed").write_html(
                out_charts / "avg_pnl_fear_vs_greed.html"
            )
            px.bar(
                fg,
                x="sentiment_bucket",
                y="avg_trade_size_usd",
                title="Average Trade Size (USD): Fear vs Greed",
            ).write_html(out_charts / "avg_trade_size_fear_vs_greed.html")
            px.bar(
                fg,
                x="sentiment_bucket",
                y="long_short_ratio_total",
                title="Long/Short Ratio: Fear vs Greed",
            ).write_html(out_charts / "long_short_ratio_fear_vs_greed.html")

            if "leverage" in merged_df.columns:
                px.histogram(
                    merged_df,
                    x="leverage",
                    title="Leverage Distribution",
                ).write_html(out_charts / "leverage_distribution.html")

            seg_counts = pd.DataFrame(
                {
                    "segment": [
                        "Consistent Winners",
                        "Net Losers",
                        "High Frequency (>5/day)",
                        "Low Frequency (<=5/day)",
                    ],
                    "traders": [
                        winners["account"].nunique(),
                        losers["account"].nunique(),
                        high_freq["account"].nunique(),
                        low_freq["account"].nunique(),
                    ],
                }
            )
            seg_counts.to_csv(out_tables / "segment_counts.csv", index=False)
            (out_tables / "segment_counts.json").write_text(
                seg_counts.to_json(orient="records", indent=2),
                encoding="utf-8",
            )
            px.bar(seg_counts, x="segment", y="traders", title="Trader Segments (counts)").write_html(
                out_charts / "segment_counts.html"
            )
        except Exception as exc:
            print(f"\nSkipping output generation: {exc}")

    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc
