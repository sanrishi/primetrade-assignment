from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis import Analyzer
from src.data_loader import DataLoader


@st.cache_data(show_spinner="Loading and preparing data...")
def load_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    loader = DataLoader()
    loader.load_data()
    merged_df = loader.clean_and_merge()

    analyzer = Analyzer(merged_df)
    daily_metrics = analyzer.calculate_daily_metrics()
    return merged_df, daily_metrics


def bucket_sentiment(classification: pd.Series) -> pd.Series:
    # Normalize to lowercase strings
    c = classification.astype(str).str.lower()

    # Initialize a Series full of "Other"
    result = pd.Series("Other", index=classification.index, dtype="object")

    # Update indices where "fear" or "greed" is found
    result.loc[c.str.contains("fear", na=False)] = "Fear"
    result.loc[c.str.contains("greed", na=False)] = "Greed"

    return result


def main() -> None:
    st.set_page_config(
        page_title="Bitcoin Sentiment vs. Trader Behavior",
        layout="wide",
    )

    st.sidebar.header("Controls")
    if st.sidebar.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    merged_df, daily_metrics = load_pipeline()

    st.title("Bitcoin Sentiment vs. Trader Behavior")
    st.subheader("Executive Summary")
    st.markdown(
        "- Compares trader outcomes across Fear vs. Greed sentiment regimes.\n"
        "- Uses daily Fear/Greed classification merged onto trade-level records.\n"
        "- Metrics shown: win rate, average PnL, and leverage distribution."
    )

    if "Classification" not in daily_metrics.columns:
        st.error("Missing expected column 'Classification' in daily metrics output.")
        return

    daily_metrics = daily_metrics.copy()
    daily_metrics["Sentiment"] = bucket_sentiment(daily_metrics["Classification"])
    fg = (
        daily_metrics[daily_metrics["Sentiment"].isin(["Fear", "Greed"])]
        .groupby("Sentiment", as_index=False)
        .agg(
            win_rate=("win_rate", "mean"),
            avg_pnl=("avg_pnl", "mean"),
            trade_count=("trade_count", "sum"),
        )
        .sort_values("Sentiment")
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Win Rate (Fear vs. Greed)")
        fig_win = px.bar(
            fg,
            x="Sentiment",
            y="win_rate",
            text=fg["win_rate"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "NA"),
            labels={"win_rate": "Win Rate", "Sentiment": "Sentiment"},
        )
        fig_win.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig_win, use_container_width=True)

    with col2:
        st.markdown("### Avg PnL (Fear vs. Greed)")
        fig_pnl = px.bar(
            fg,
            x="Sentiment",
            y="avg_pnl",
            text=fg["avg_pnl"].map(lambda v: f"{v:,.2f}" if pd.notna(v) else "NA"),
            labels={"avg_pnl": "Avg PnL", "Sentiment": "Sentiment"},
        )
        st.plotly_chart(fig_pnl, use_container_width=True)

    st.markdown("### Leverage Distribution")
    if "leverage" in merged_df.columns:
        lev = pd.to_numeric(merged_df["leverage"], errors="coerce").dropna()
        fig_lev = px.histogram(
            lev.to_frame(name="leverage"),
            x="leverage",
            nbins=min(20, max(1, int(lev.nunique()))),
            labels={"leverage": "Leverage"},
        )
        st.plotly_chart(fig_lev, use_container_width=True)
    else:
        st.info("Leverage column not found in merged data.")

    st.markdown("### Insights")
    if {"Fear", "Greed"}.issubset(set(fg["Sentiment"])) and fg.shape[0] >= 2:
        fear = fg.loc[fg["Sentiment"] == "Fear"].iloc[0]
        greed = fg.loc[fg["Sentiment"] == "Greed"].iloc[0]

        higher_win = "Fear" if fear["win_rate"] >= greed["win_rate"] else "Greed"
        lower_pnl = "Fear" if fear["avg_pnl"] <= greed["avg_pnl"] else "Greed"

        if higher_win != lower_pnl:
            insight = (
                f"Key Insight: Traders tend to have higher win rates during {higher_win} days, "
                f"but lower average PnL."
            )
        else:
            insight = (
                f"Key Insight: Traders tend to have higher win rates during {higher_win} days, "
                f"and also higher average PnL."
            )
    else:
        insight = (
            "Key Insight: Traders tend to have higher win rates during [Fear/Greed] days, "
            "but lower average PnL."
        )

    st.write(insight)


if __name__ == "__main__":
    main()
