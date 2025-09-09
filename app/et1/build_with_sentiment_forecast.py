# /app/et1/build_with_sentiment_forecast.py
from pathlib import Path
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---------------- Paths ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGG_FILE     = PROJECT_ROOT / "data" / "processed" / "genre_year_agg.csv"
HYPE_FILE    = PROJECT_ROOT / "data" / "processed" / "hype_index_annual.csv"
OUT_DIR      = PROJECT_ROOT / "data" / "processed"
OUT_FILE     = OUT_DIR / "forecast_with_sentiment.csv"

# ---------------- Settings ----------------
MODEL_VERSION    = "v0.2-with-sentiment"
CONF_ALPHA       = 0.05
FORECAST_H       = 5
MIN_YEARS_EXOG   = 5
SELECTED_GENRES  = ["Action","Adventure","Animation","Comedy","Drama","Horror","Romance","Sci-Fi","Thriller","Documentary","Family","Fantasy"]

# Set to a list like [2021,2022,2023,2024,2025] to force a uniform horizon.
LOCK_OUTPUT_YEARS = None   # or e.g., list(range(2021, 2026))

CANDIDATE_ORDERS = [(1,1,1), (0,1,1), (1,1,0), (0,1,0)]
warnings.filterwarnings("ignore")


# ------------- Helpers -------------
def fill_internal_gaps(y: pd.Series) -> pd.Series:
    """
    Reindex only from min->max observed year and fill INTERNAL gaps with 0.
    Does NOT extend beyond the last observed year (avoids padding tail with zeros).
    """
    y = y.sort_index()
    idx = np.arange(int(y.index.min()), int(y.index.max()) + 1)
    return y.reindex(idx, fill_value=0.0).astype("float64")

def trim_silent_edges(y: pd.Series) -> pd.Series:
    """Trim leading/trailing runs of zeros to the 'active' window."""
    arr = y.to_numpy()
    nz = np.flatnonzero(arr > 0)
    if nz.size == 0:
        return y.iloc[0:0]
    return y.iloc[nz[0]: nz[-1] + 1]

def last_observed_nonzero(gsub: pd.DataFrame):
    """
    From ACTUAL rows (no padding), return (value, year) of the last non-zero count.
    If none, return (0.0, last_year_in_data).
    """
    g = gsub[gsub["title_count"] > 0]
    if g.empty:
        return 0.0, int(gsub["year"].max())
    last = g.iloc[-1]
    return float(last["title_count"]), int(last["year"])

def fit_best_arimax(y: pd.Series, x: pd.Series):
    """Try small ARIMAX specs; pick lowest AIC."""
    best = None; best_order = None; best_aic = np.inf
    for order in CANDIDATE_ORDERS:
        try:
            mod = sm.tsa.statespace.SARIMAX(
                y, exog=x, order=order, trend="c",
                enforce_stationarity=False, enforce_invertibility=False
            )
            res = mod.fit(disp=False)
            if np.isfinite(res.aic) and res.aic < best_aic:
                best, best_order, best_aic = res, order, res.aic
        except Exception:
            continue
    return best, best_order, best_aic

def future_exog(x_hist: pd.Series, h: int) -> np.ndarray:
    """Future hype = last 3-year average (or last value)."""
    if len(x_hist) >= 3:
        base = float(x_hist.iloc[-3:].mean())
    elif len(x_hist) >= 1:
        base = float(x_hist.iloc[-1])
    else:
        base = 0.0
    return np.repeat(base, h)

def naive_forecast(last_val: float, h: int, scale: float = 0.2):
    yhat = np.repeat(last_val, h)
    lower = np.clip(yhat * (1 - scale), 0, None)
    upper = yhat * (1 + scale)
    return yhat, lower, upper


# ------------- Main -------------
def main():
    if not AGG_FILE.exists(): raise SystemExit(f"[ERROR] Missing: {AGG_FILE}")
    if not HYPE_FILE.exists(): raise SystemExit(f"[ERROR] Missing: {HYPE_FILE}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gdf = pd.read_csv(AGG_FILE)
    hdf = pd.read_csv(HYPE_FILE)

    # Basic checks
    need_g = {"genre","year","title_count"}
    need_h = {"genre","year","hype_index"}
    if not need_g.issubset(gdf.columns): raise SystemExit(f"[ERROR] genre_year_agg.csv needs: {need_g}")
    if not need_h.issubset(hdf.columns): raise SystemExit(f"[ERROR] hype_index_annual.csv needs: {need_h}")

    gdf["year"] = gdf["year"].astype(int)
    hdf["year"] = hdf["year"].astype(int)

    rows = []
    created_at = datetime.utcnow().isoformat()

    for genre in SELECTED_GENRES:
        gsub = gdf[gdf["genre"] == genre].sort_values("year").copy()
        if gsub.empty:
            print(f"[WARN] {genre}: no history — skipping.")
            continue

        # Build y (counts) without padding beyond last observed year
        y_raw = pd.Series(gsub["title_count"].values, index=gsub["year"].values, dtype="float64")
        y_full = fill_internal_gaps(y_raw)
        y_act  = trim_silent_edges(y_full)

        last_val_obs, last_year_obs = last_observed_nonzero(gsub)

        # Align hype to active window
        x_full = hdf[hdf["genre"] == genre].set_index("year")["hype_index"].astype("float64")
        years_overlap = sorted(set(y_act.index) & set(x_full.index))
        y_aligned = y_act.loc[years_overlap] if years_overlap else pd.Series(dtype="float64")
        x_aligned = x_full.loc[years_overlap] if years_overlap else pd.Series(dtype="float64")

        # Debug summary per genre
        print(f"[INFO] {genre}: last_nonzero={last_val_obs} in {last_year_obs}, "
              f"overlap_years={len(years_overlap)} ({years_overlap[:1]}..{years_overlap[-1:]})")

        # Decide base year (where forecast starts from)
        base_year = (years_overlap[-1] if years_overlap else last_year_obs)

        # Try ARIMAX if enough overlap, else naive using last observed non-zero
        if len(y_aligned) >= MIN_YEARS_EXOG:
            res, order, aic = fit_best_arimax(y_aligned, x_aligned)
            if res is not None:
                x_future = future_exog(x_aligned, FORECAST_H)
                fc = res.get_forecast(steps=FORECAST_H, exog=x_future.reshape(-1,1))
                ci = fc.conf_int(alpha=CONF_ALPHA)
                yhat  = np.clip(fc.predicted_mean.values, 0, None)
                lower = np.clip(ci.iloc[:,0].values,       0, None)
                upper = np.clip(ci.iloc[:,1].values,       0, None)
                model_name, order_str = "SARIMAX+HYPE", f"{order}"
            else:
                print(f"[WARN] {genre}: ARIMAX fit failed — using naive(last observed).")
                yhat, lower, upper = naive_forecast(last_val_obs, FORECAST_H)
                model_name, order_str, aic = "naive", "n/a", np.nan
        else:
            print(f"[WARN] {genre}: only {len(y_aligned)} overlapping years — using naive(last observed).")
            yhat, lower, upper = naive_forecast(last_val_obs, FORECAST_H)
            model_name, order_str, aic = "naive", "n/a", np.nan

        # Optional uniform horizon (e.g., 2021–2025)
        if LOCK_OUTPUT_YEARS is not None:
            years_out = LOCK_OUTPUT_YEARS
        else:
            years_out = [base_year + i + 1 for i in range(FORECAST_H)]

        for i, yr in enumerate(years_out):
            rows.append({
                "genre": genre,
                "year": int(yr),
                "yhat": float(yhat[i]),
                "yhat_lower": float(lower[i]),
                "yhat_upper": float(upper[i]),
                "model": model_name,
                "order": order_str,
                "aic": float(aic) if isinstance(aic, (int,float)) and np.isfinite(aic) else np.nan,
                "n_obs": int(len(y_aligned)),
                "model_version": MODEL_VERSION,
                "created_at": created_at,
            })

    out = pd.DataFrame(rows).sort_values(["genre","year"])
    tmp = OUT_FILE.with_suffix(".tmp.csv")
    out.to_csv(tmp, index=False)
    tmp.replace(OUT_FILE)

    print(f"[OK] Wrote with-sentiment forecasts → {OUT_FILE}")
    try:
        print(out.groupby("genre")["yhat"].mean().round(1).sort_values(ascending=False).head(6))
    except Exception:
        pass


if __name__ == "__main__":
    main()
