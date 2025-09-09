# Netflix Genre Trend Forecasting — MVP

**Outcome:** 5-year genre forecasts with confidence bands and a Hype Index (Google Trends) to show how audience interest shifts the outlook. One-page dashboard, accuracy table, and executive brief.

**Decisions answered:** Top-3 rising genres; confidence (High/Med/Low); impact of hype vs. history-only.

**Folders**
/app — dashboard (later)
/assets — screenshots
/data/raw — input (place `all_df.csv` here)
/data/processed — outputs (agg & forecasts)
/docs — project docs (framing, metrics, hype)
/reports — executive brief

## Status

- ✅ Hour 1/7 complete — framing & scaffolding done
- ✅ IMDb base: `genre_year_agg.csv` created (1980–2020)
- ✅ Hype Index (Google Trends): 12 monthly CSVs exported and consolidated
  - Monthly → `data/processed/hype_index_monthly.csv`
  - Annual  → `data/processed/hype_index_annual.csv`
- ▶ With-sentiment subset selected: Sci-Fi, Thriller, Action, Drama, Horror, Romance

**Next visible milestone:** Baseline 5-year forecasts from IMDb history (`forecast_baseline.csv`) with uncertainty bands.
