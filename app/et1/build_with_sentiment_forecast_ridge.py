# /app/et1/build_with_sentiment_forecast_ridge.py
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

# -------- Paths --------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGG_FILE     = PROJECT_ROOT / "data" / "processed" / "genre_year_agg.csv"
HYPE_FILE    = PROJECT_ROOT / "data" / "processed" / "hype_index_annual.csv"
OUT_DIR      = PROJECT_ROOT / "data" / "processed"
OUT_FILE     = OUT_DIR / "forecast_with_sentiment.csv"   # overwrite on purpose

# -------- Settings --------
MODEL_VERSION   = "v0.3.1-ridge-ar1-hype"
H               = 5                          # horizon (years)
MIN_OBS         = 6                          # min rows AFTER creating lag to fit
L2              = 1.0                        # ridge strength
UNIFORM_YEARS   = list(range(2021, 2026))    # force a common 5-year horizon

# Canonical genres (use only those present in the data)
CANON_12 = [
    "Action","Adventure","Animation","Comedy","Drama","Horror",
    "Romance","Sci-Fi","Thriller","Documentary","Family","Fantasy"
]

# ---------------- Helpers ----------------
def zscore(x: np.ndarray):
    m = np.nanmean(x)
    s = np.nanstd(x)
    if s == 0 or np.isnan(s):
        return np.zeros_like(x), m, 1.0
    return (x - m) / s, m, s

def ridge_fit(X: np.ndarray, y: np.ndarray, l2: float):
    """
    Closed-form ridge: beta = (X^T X + l2*I)^{-1} X^T y
    Intercept is NOT penalized.
    """
    XtX = X.T @ X
    I = np.eye(X.shape[1])
    I[0, 0] = 0.0  # don't penalize intercept
    beta = np.linalg.pinv(XtX + l2 * I) @ (X.T @ y)
    return beta

def build_features(df_merged: pd.DataFrame, min_obs: int):
    """
    df_merged columns: year, y (title_count), hype
    Creates one-step lag of y, z-scores features.
    Returns (X, y, scalers, df_used) or None if not enough rows.
    """
    df = df_merged.sort_values("year").copy()
    df["lag1"] = df["y"].shift(1)
    df = df.dropna(subset=["lag1"]).reset_index(drop=True)

    # If too short after making lag1 -> signal caller to fallback
    if len(df) < min_obs:
        return None

    base_year = int(df["year"].iloc[0])
    df["trend"] = df["year"] - base_year

    lag1_z, lag1_m, lag1_s = zscore(df["lag1"].to_numpy(dtype=float))
    hype_z, hype_m, hype_s = zscore(df["hype"].to_numpy(dtype=float))
    trend_z, tr_m, tr_s    = zscore(df["trend"].to_numpy(dtype=float))

    X = np.column_stack([
        np.ones(len(df)),  # intercept
        lag1_z,
        hype_z,
        trend_z,
    ])
    y = df["y"].to_numpy(dtype=float)

    scalers = {
        "lag1": (lag1_m, lag1_s),
        "hype": (hype_m, hype_s),
        "trend": (tr_m, tr_s),
        "base_year": base_year,
    }
    return X, y, scalers, df

def future_hype_assumption(hype_hist: pd.Series) -> float:
    """Future hype value = last 3-year average (or last value, else 0)."""
    if len(hype_hist) >= 3:
        return float(hype_hist.iloc[-3:].mean())
    if len(hype_hist) >= 1:
        return float(hype_hist.iloc[-1])
    return 0.0

def last_nonzero_year_val(y_df: pd.DataFrame):
    """From raw counts df (cols: year,y), return (year,val) of last non-zero; else (last_year, 0)."""
    nz = y_df[y_df["y"] > 0]
    if nz.empty:
        return int(y_df["year"].max()), 0.0
    row = nz.iloc[-1]
    return int(row["year"]), float(row["y"])

def predict_roll_forward(beta: np.ndarray, scalers: dict, last_year_obs: int, last_y: float,
                         hype_future_val: float, h: int):
    """Step-by-step forecast so the lag updates with each prediction."""
    (lag1_m, lag1_s) = scalers["lag1"]
    (hype_m, hype_s) = scalers["hype"]
    (tr_m, tr_s)     = scalers["trend"]
    base_year        = scalers["base_year"]

    years = []
    preds = []
    prev_y = float(last_y)

    for i in range(1, h + 1):
        yr = last_year_obs + i
        trend_val = yr - base_year

        lag1_z = (prev_y - lag1_m) / (lag1_s if lag1_s != 0 else 1.0)
        hype_z = (hype_future_val - hype_m) / (hype_s if hype_s != 0 else 1.0)
        trend_z = (trend_val - tr_m) / (tr_s if tr_s != 0 else 1.0)

        x = np.array([1.0, lag1_z, hype_z, trend_z], dtype=float)
        pred = float(x @ beta)
        pred = max(pred, 0.0)  # counts can't be negative

        years.append(yr)
        preds.append(pred)
        prev_y = pred

    years = np.array(years, dtype=int)
    yhat = np.array(preds, dtype=float)

    # Simple bands: ±20% with a minimum band width of 2
    band = np.maximum(0.2 * yhat, 2.0)
    lower = np.clip(yhat - band, 0.0, None)
    upper = yhat + band
    return years, yhat, lower, upper

# ---------------- Main ----------------
def main():
    if not AGG_FILE.exists():
        raise SystemExit(f"[ERROR] Missing: {AGG_FILE}")
    if not HYPE_FILE.exists():
        raise SystemExit(f"[ERROR] Missing: {HYPE_FILE}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gdf = pd.read_csv(AGG_FILE)
    hdf = pd.read_csv(HYPE_FILE)

    for col in ("genre","year","title_count"):
        if col not in gdf.columns:
            raise SystemExit("[ERROR] genre_year_agg.csv needs columns: genre, year, title_count")
    for col in ("genre","year","hype_index"):
        if col not in hdf.columns:
            raise SystemExit("[ERROR] hype_index_annual.csv needs columns: genre, year, hype_index")

    gdf["year"] = gdf["year"].astype(int)
    hdf["year"] = hdf["year"].astype(int)

    genres = [g for g in CANON_12 if g in set(gdf["genre"].unique())]
    rows = []
    created_at = datetime.now(timezone.utc).isoformat()

    for genre in genres:
        # Raw series
        gsub = gdf[gdf["genre"] == genre][["year","title_count"]].rename(columns={"title_count":"y"}).copy()
        hsub = hdf[hdf["genre"] == genre][["year","hype_index"]].rename(columns={"hype_index":"hype"}).copy()

        # Merge on counts years (left join), fill hype sensibly
        merged = pd.merge(gsub, hsub, on="year", how="left").sort_values("year")
        if merged["hype"].isna().any():
            merged["hype"] = (
                merged["hype"].ffill()
                               .bfill()
                               .fillna(merged["hype"].mean())
                               .fillna(0.0)
            )

        # Build features; if too few rows AFTER lag1, fallback
        feats = build_features(merged, MIN_OBS)
        if feats is None:
            last_year, last_val = last_nonzero_year_val(gsub)
            years_out = UNIFORM_YEARS or [last_year + i for i in range(1, H + 1)]
            for yr in years_out:
                yhat = max(last_val, 0.0)
                band = max(2.0, 0.2 * yhat)
                rows.append({
                    "genre": genre, "year": int(yr),
                    "yhat": yhat,
                    "yhat_lower": max(yhat - band, 0.0),
                    "yhat_upper": yhat + band,
                    "model": "Ridge-AR1+Hype (naive fallback)",
                    "order": "ridge_ar1",
                    "aic": np.nan,
                    "n_obs": 0,
                    "model_version": MODEL_VERSION,
                    "created_at": created_at,
                })
            continue

        X, y, scalers, df_train = feats

        # Fit ridge
        beta = ridge_fit(X, y, L2)

        # Future hype value (constant for forward steps)
        hype_hist = merged.set_index("year")["hype"]
        hyp_future_val = future_hype_assumption(hype_hist)

        # Last observed training point
        last_year_obs = int(df_train["year"].iloc[-1])
        last_y        = float(df_train["y"].iloc[-1])

        # Predict forward
        years_f, yhat, lower, upper = predict_roll_forward(
            beta, scalers, last_year_obs, last_y, hyp_future_val, H
        )

        # Degenerate safety: if all ~0, fall back to last non-zero observed level
        if np.allclose(yhat, 0.0):
            last_year, last_val = last_nonzero_year_val(gsub)
            yhat = np.repeat(max(last_val, 0.0), H).astype(float)
            band = np.maximum(0.2 * yhat, 2.0)
            lower = np.clip(yhat - band, 0.0, None)
            upper = yhat + band
            model_name = "Ridge-AR1+Hype (degenerate→naive)"
            n_obs_out = int(len(df_train))
        else:
            model_name = "Ridge-AR1+Hype"
            n_obs_out = int(len(df_train))

        # Optionally force a uniform horizon
        years_out = UNIFORM_YEARS if UNIFORM_YEARS else years_f.tolist()

        for i in range(H):
            rows.append({
                "genre": genre,
                "year": int(years_out[i]),
                "yhat": float(yhat[i]),
                "yhat_lower": float(lower[i]),
                "yhat_upper": float(upper[i]),
                "model": model_name,
                "order": "ridge_ar1",
                "aic": np.nan,
                "n_obs": n_obs_out,
                "model_version": MODEL_VERSION,
                "created_at": created_at,
            })

    out = pd.DataFrame(rows).sort_values(["genre","year"])
    tmp = OUT_FILE.with_suffix(".tmp.csv")
    out.to_csv(tmp, index=False)
    tmp.replace(OUT_FILE)

    print(f"[OK] Wrote with-sentiment forecasts → {OUT_FILE}")
    try:
        print(out.groupby("genre")["yhat"].mean().round(1).sort_values(ascending=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()
