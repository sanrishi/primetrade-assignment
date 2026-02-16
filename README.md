# Bitcoin Sentiment vs. Trader Behavior Analysis
**Submission for Primetrade.ai Data Science Internship**

## Objective
Analyze how Bitcoin market sentiment (Fear/Greed) relates to trader behavior and performance on Hyperliquid, and extract actionable patterns/strategy rules.

## Repository Structure
- `data/` — input CSV datasets (not committed; `*.csv` is gitignored)
- `src/` — reusable pipeline modules (loader + analysis)
- `notebooks/` — end-to-end analysis notebook (deliverable)
- `output/` — generated charts/tables/reports (committed)
- `app.py` — Streamlit dashboard (bonus)

## Setup (Windows / PowerShell)
```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Datasets (Place Files Here)
Download the datasets from the assignment links and place them as:
- `data/fear_greed_index.csv`
- `data/historical_data.csv`

Note: `*.csv` is intentionally ignored in `.gitignore` so you don’t upload datasets to GitHub.

## How To Run
### 1) Data Loader (merge + QA prints)
```powershell
python src/data_loader.py
```

### 2) Analysis (daily metrics + segmentation)
```powershell
python src/analysis.py
```

### 3) Dashboard (Streamlit)
```powershell
streamlit run app.py
```

## Methodology (What This Repo Produces)
### Data Preparation
- Parses and normalizes timestamps (`Timestamp IST` → daily date).
- Merges trade-level records with daily Fear/Greed classification (left join, then filters missing sentiment labels).

### Core Metrics
- **Avg PnL**: mean of `closedPnL` per day and sentiment bucket
- **Win Rate**: fraction of trades with positive PnL
- **Trade Count**: number of trades per day
- **Avg Leverage**: mean of `leverage` per day (if present)

### Trader Segments
- **High leverage** vs **Low leverage**: top 25% vs bottom 25% by average leverage (per account).

## Notebook Deliverable
Use `notebooks/01_assignment_analysis.ipynb` for the full write-up (cleaning, evidence, charts/tables, insights, and strategy recommendations).

## Outputs
Save generated artifacts under:
- `output/charts/`
- `output/tables/`
- `output/reports/`

## Data Quality Note (Leverage)
If the trades CSV does not include a leverage column, `DataLoader.load_data()` synthesizes a `leverage` column by sampling from common discrete leverage tiers to keep the analysis pipeline runnable and explicitly prints a warning.

## Project Summary
### Methodology
- **Schema-aware ingestion:** Loads both CSVs using pandas and resolves common schema variations (case-insensitive column matching, e.g., `Timestamp IST` vs `time`, `classification` vs `Classification`). The loader prints the trades column list for transparent QA.
- **Timestamp alignment (daily grain):** Parses sentiment `date` and trade timestamps, normalizes trade timestamps to a daily `date_only` via vectorized `dt.normalize()`, then performs a **left join** from trades onto daily sentiment. Trades without a matching daily sentiment label are dropped after the merge to keep KPIs interpretable.
- **High-performance KPIs:** Uses vectorized pandas `groupby(...).agg(...)` to compute daily metrics by sentiment classification:
  - Average PnL (mean of `Closed PnL`)
  - Win Rate (share of trades with positive PnL; safe against division by zero)
  - Trade Count (number of trades)
  - Average Leverage (mean of `leverage` when available)
- **Trader segmentation:** Computes per-account average leverage and total PnL, then segments **High Leverage** (top 25%) vs **Low Leverage** (bottom 25%) traders via quantiles.

### Raw Data QA (snapshot)
Counts below are from running the pipeline locally on **February 16, 2026** (your numbers may differ if the source files change).

| Dataset | Rows | Columns | Missing cells | Duplicate rows |
|---|---:|---:|---:|---:|
| Sentiment (Fear/Greed) | 2,644 | 4 | 0 | 0 |
| Trades (Hyperliquid) | 211,224 | 17 | 0 | 0 |
| Merged (Trades ← Sentiment) | 211,218 | 21 | 0 | 0 |

### Key Insights (from `python src/analysis.py`)
1. **Greed regimes outperform Fear regimes:** When bucketing sentiment into Fear vs Greed, Greed days show **higher win rate (~38.5% vs ~32.9%)** and **higher average PnL (~45.85 vs ~32.23)**.
2. **Fear drives substantially higher trading frequency:** Fear days exhibit a much higher average trade count per day (**~792.7 trades/day**) versus Greed (**~294.1 trades/day**), suggesting more reactive/high-churn behavior under Fear without a corresponding win-rate improvement.
3. **Leverage comparison is pipeline-ready but data-limited:** The current trade export snapshot does **not** contain leverage, so the pipeline prints a warning and synthesizes leverage tiers to keep the dashboard + segmentation runnable. Replacing the input with a leverage-complete export will immediately quantify true leverage shifts by sentiment.

### Professional Strategy Recommendations
1. **Sentiment-Adjusted Trade Frequency Rule:** On Fear / Extreme Fear days, apply a stricter trade-quality filter (or cap trades/day) to reduce churn; the observed pattern is higher activity without improved win rate.
2. **Greed-Regime Risk Budget Tilt:** On Greed / Extreme Greed days, allow a controlled increase in risk allocation (position sizing or risk budget) for historically consistent accounts, since win rate and average PnL are higher in Greed regimes.

### Visual References (Bonus)
The Streamlit dashboard (`streamlit run app.py`) provides interactive, visual confirmation of these comparisons (Win Rate, Avg PnL, and leverage distribution), and supports quick sanity checks via a one-click data refresh.
