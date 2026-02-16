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
