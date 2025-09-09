# /app/etl/build_baseline_forecast.py
from pathlib import Path
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_FILE      = PROJECT_ROOT / "data" / "processed" / "genre_year_agg.csv"
OUT_DIR      = PROJECT_ROOT / "data" / "processed"
OUT_FILE     = OUT_DIR / "forecast_baseline.csv"

# ---------- Settings ----------
MODEL_VERSION = "v0.1-baseline"
CONF_ALPHA = 0.05      # 95% CI
MIN_YEARS  = 8         # minimum observations to fit ARIMA after trimming
FORECAST_H = 5         # years ahead

CANON_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Drama", "Horror",
    "Romance", "Sci-Fi", "Thriller", "Documentary", "Family", "Fantasy"
]

CANDIDATE_ORDERS = [
    (1,1,1), (0,1,1), (1,1,0), (0,1,0)
]

warnings.filterwarnings("ignore")

def reindex_years(s: pd.Series) -> pd.Series:
    """Ensure continuous annual index; fill missing years with 0 (we trim later)."""
    s = s.sort_index()
    idx = np.arange(int(s.index.min()), int(s.index.max()) + 1)
    return s.reindex(idx, fill_value=0.0).astype("float64")

def trim_leading_trailing_zeros(y: pd.Series) -> pd.Series:
    """Remove leading and trailing runs of zeros; keep the ‘active’ window."""
    arr = y.to_numpy()
    nz = np.flatnonzero(arr > 0)
    if nz.size == 0:
        return y.iloc[0:0]  # empty series → means 'all zeros'
    start = nz[0]
    end   = nz[-1]
    return y.iloc[start:end+1]

def fit_best_arima(y: pd.Series):
    """Try a few small ARIMA orders and pick best by AIC."""
    best = None
    best_order = None
    best_aic = np.inf
    for order in CANDIDATE_ORDERS:
        try:
            mod = sm.tsa.statespace.SARIMAX(
                y, order=order, trend="c",
                enforce_stationarity=False, enforce_invertibility=False
            )
            res = mod.fit(disp=False)
            if np.isfinite(res.aic) and res.aic < best_aic:
                best, best_order, best_aic = res, order, res.aic
        except Exception:
            continue
    return best, best_order, best_aic

def naive_forecast(last_val: float, h: int, scale: float = 0.2):
    """Fallback: flat line with ±20% band."""
    yhat = np.repeat(last_val, h)
    lower = np.clip(yhat * (1 - scale), 0, None)
    upper = yhat * (1 + scale)
    return yhat, lower, upper

def main():
    if not IN_FILE.exists():
        raise SystemExit(f"[ERROR] Missing input: {IN_FILE}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_FILE)
    if not {"genre", "year", "title_count"}.issubset(df.columns):
        raise SystemExit("[ERROR] genre_year_agg.csv must have columns: genre, year, title_count")

    df = df[df["genre"].isin(CANON_GENRES)].copy()

    rows = []
    created_at = datetime.utcnow().isoformat()

    for genre in sorted(df["genre"].unique()):
        gdf = df[df["genre"] == genre].copy()
        gdf["year"] = gdf["year"].astype(int)
        gdf = gdf.sort_values("year")

        # Original observed last year (before any padding)
        last_obs_year = int(gdf["year"].max())

        # Build a continuous series then trim silent edges
        y = pd.Series(gdf["title_count"].values, index=gdf["year"].values, dtype="float64")
        y = reindex_years(y)
        y_trim = trim_leading_trailing_zeros(y)

        if len(y_trim) == 0:
            # Truly all zeros -> flat zero forecast
            print(f"[WARN] {genre}: all zeros after trimming — using naive(0).")
            last_val = 0.0
            yhat, lower, upper = naive_forecast(last_val, FORECAST_H)
            model_name, order_str, aic = "naive", "n/a", np.nan
            n_obs = int(len(y))
            base_year = last_obs_year
        else:
            n_obs = int(len(y_trim))
            base_year = int(y_trim.index.max())

            if n_obs < MIN_YEARS:
                print(f"[WARN] {genre}: only {n_obs} years after trimming — using naive.")
                last_val = float(y_trim.iloc[-1])
                yhat, lower, upper = naive_forecast(last_val, FORECAST_H)
                model_name, order_str, aic = "naive", "n/a", np.nan
            else:
                res, order, aic = fit_best_arima(y_trim)
                if res is None:
                    print(f"[WARN] {genre}: ARIMA fit failed — using naive.")
                    last_val = float(y_trim.iloc[-1])
                    yhat, lower, upper = naive_forecast(last_val, FORECAST_H)
                    model_name, order_str, aic = "naive", "n/a", np.nan
                else:
                    fc = res.get_forecast(steps=FORECAST_H)
                    ci = fc.conf_int(alpha=CONF_ALPHA)
                    yhat  = np.clip(fc.predicted_mean.values, 0, None)
                    lower = np.clip(ci.iloc[:, 0].values, 0, None)
                    upper = np.clip(ci.iloc[:, 1].values, 0, None)
                    model_name, order_str = "SARIMAX", f"{order}"

        # Write rows
        for i in range(FORECAST_H):
            rows.append({
                "genre": genre,
                "year": base_year + i + 1,  # continue from last active year
                "yhat": float(yhat[i]),
                "yhat_lower": float(lower[i]),
                "yhat_upper": float(upper[i]),
                "model": model_name,
                "order": order_str,
                "aic": float(aic) if np.isfinite(aic) else np.nan,
                "n_obs": n_obs,
                "model_version": MODEL_VERSION,
                "created_at": created_at,
            })

    out = pd.DataFrame(rows).sort_values(["genre", "year"])

    # Write atomically to avoid empty files
    tmp = OUT_FILE.with_suffix(".tmp.csv")
    out.to_csv(tmp, index=False)
    tmp.replace(OUT_FILE)

    print(f"[OK] Wrote baseline forecasts → {OUT_FILE}")
    try:
        print(out.groupby("genre")["yhat"].mean().round(1).sort_values(ascending=False).head(5))
    except Exception:
        pass

if __name__ == "__main__":
    main()
