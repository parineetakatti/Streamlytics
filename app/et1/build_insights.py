# app/et1/build_insights.py
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_FILE = PROJECT_ROOT / "data" / "processed" / "forecast_baseline.csv"
WITH_FILE = PROJECT_ROOT / "data" / "processed" / "forecast_with_sentiment.csv"
AGG_FILE  = PROJECT_ROOT / "data" / "processed" / "genre_year_agg.csv"
HYPE_FILE = PROJECT_ROOT / "data" / "processed" / "hype_index_annual.csv"
OUT_DIR   = PROJECT_ROOT / "data" / "processed"
OUT_FILE  = OUT_DIR / "insights_genre.csv"
JOINED_OUT = OUT_DIR / "forecast_joined.csv"   # optional, helpful for debugging

MODEL_VERSION = "v0.6.1-insights"
EPS = 1e-6
SPARSE_BASELINE_THRESHOLD = 5.0   # if avg_baseline < 5 titles/yr, treat as “sparse baseline”

BACKTEST_GENRES = ["Drama", "Comedy", "Action", "Thriller"]
BACKTEST_CUTOFF = 2015  # train ≤ 2015, test 2016–2020

def safe_mean(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    return float(x.mean()) if len(x) else np.nan

def confidence_label(n_obs, rel_bw, non_naive_share):
    if (n_obs is not None and n_obs >= 10) and (rel_bw is not None and rel_bw <= 0.40):
        return "High"
    if (n_obs is not None and n_obs >= 7) and (rel_bw is not None and rel_bw <= 0.80):
        return "Medium"
    return "Low"

def confidence_score(n_obs, rel_bw, non_naive_share):
    score = 0.0
    if n_obs is not None:
        score += min(n_obs, 12) / 12 * 40  # up to 40 points
    if rel_bw is not None and np.isfinite(rel_bw):
        score += max(0.0, (0.8 - rel_bw) / 0.8) * 40  # tighter bands => more points
    if non_naive_share is not None and np.isfinite(non_naive_share):
        score += float(non_naive_share) * 20  # ARX/ridge share up to 20
    return round(float(np.clip(score, 0, 100)), 1)

def cagr_pct(first_val, last_val, years=4):
    a = max(first_val, EPS)
    b = max(last_val, EPS)
    return (b / a) ** (1/years) - 1.0

def summarize_joined(join):
    rows = []
    for g, gdf in join.groupby("genre"):
        gdf = gdf.sort_values("year")

        # Averages
        avg_base = safe_mean(gdf["yhat_base"])
        avg_with = safe_mean(gdf["yhat_with"])
        abs_delta = (avg_with - avg_base) if (np.isfinite(avg_with) and np.isfinite(avg_base)) else np.nan

        # Symmetric uplift % (bounded, robust when baseline ~0)
        denom = max(avg_with + avg_base, EPS)
        uplift_sym = 200.0 * (avg_with - avg_base) / denom  # −200%..+200%

        # Bands (with-sentiment)
        bw_half = 0.5 * (
            (gdf["yhat_upper_with"] - gdf["yhat_with"]).abs()
          + (gdf["yhat_with"] - gdf["yhat_lower_with"]).abs()
        )
        avg_bw = safe_mean(bw_half)
        rel_bw = (avg_bw / max(avg_with, EPS)) if np.isfinite(avg_with) else np.nan

        # CAGR using with-sentiment first/last horizon years
        first_row = gdf.iloc[0]
        last_row  = gdf.iloc[-1]
        cagr = cagr_pct(first_row["yhat_with"], last_row["yhat_with"], years=(int(last_row["year"]) - int(first_row["year"])))

        # n_obs (from with-sentiment, should be constant per genre)
        n_obs = safe_mean(gdf["n_obs_with"])
        # Share of non-naive models in with-sentiment horizon
        non_naive_share = (gdf["model_with"].str.contains("naive", case=False, na=False) == False).mean()

        # Baseline sparsity flag
        baseline_sparse = bool(avg_base < SPARSE_BASELINE_THRESHOLD if np.isfinite(avg_base) else True)

        label = confidence_label(n_obs, rel_bw, non_naive_share)
        score = confidence_score(n_obs, rel_bw, non_naive_share)

        rows.append({
            "genre": g,
            "avg_baseline": round(avg_base, 3) if np.isfinite(avg_base) else np.nan,
            "avg_with": round(avg_with, 3) if np.isfinite(avg_with) else np.nan,
            "abs_delta": round(abs_delta, 3) if np.isfinite(abs_delta) else np.nan,
            "uplift_pct": round(uplift_sym, 2) if np.isfinite(uplift_sym) else np.nan,  # symmetric uplift
            "cagr_pct": round(100 * cagr, 2) if np.isfinite(cagr) else np.nan,
            "avg_band_width": round(avg_bw, 3) if np.isfinite(avg_bw) else np.nan,
            "rel_band_width": round(rel_bw, 3) if np.isfinite(rel_bw) else np.nan,
            "n_obs": int(round(n_obs)) if np.isfinite(n_obs) else np.nan,
            "model_share_non_naive": round(float(non_naive_share), 2),
            "baseline_sparse": baseline_sparse,
            "confidence_label": label,
            "confidence_score": score,
        })
    # sort by confidence then growth
    return pd.DataFrame(rows).sort_values(["confidence_label","cagr_pct"], ascending=[True, False])

# --- Backtest helpers (light ridge AR1 + hype) ---
def zscore(x):
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x); s = np.nanstd(x)
    if not np.isfinite(s) or s == 0: return np.zeros_like(x), m, 1.0
    return (x - m)/s, m, s

def ridge_fit(X, y, l2=1.0):
    XtX = X.T @ X
    I = np.eye(X.shape[1]); I[0,0] = 0.0
    beta = np.linalg.pinv(XtX + l2 * I) @ (X.T @ y)
    return beta

def backtest_for_genre(genre, agg_df, hype_df, cutoff=2015, min_obs=6, l2=1.0):
    g = agg_df[agg_df["genre"] == genre][["year","title_count"]].rename(columns={"title_count":"y"}).copy()
    h = hype_df[hype_df["genre"] == genre][["year","hype_index"]].rename(columns={"hype_index":"h"}).copy()
    if g.empty: return None

    df = pd.merge(g, h, on="year", how="left").sort_values("year")
    if df["h"].isna().any():
        df["h"] = df["h"].ffill().bfill().fillna(df["h"].mean()).fillna(0.0)

    train = df[df["year"] <= cutoff].copy()
    test  = df[(df["year"] > cutoff) & (df["year"] <= 2020)].copy()
    if len(train) < min_obs + 1 or len(test) < 2:
        return None

    train["lag1"] = train["y"].shift(1)
    train = train.dropna(subset=["lag1"]).reset_index(drop=True)
    if len(train) < min_obs: return None

    base_year = int(train["year"].iloc[0])
    train["trend"] = train["year"] - base_year

    lag1_z, l_m, l_s = zscore(train["lag1"].to_numpy())
    hype_z, h_m, h_s = zscore(train["h"].to_numpy())
    tr_z,   t_m, t_s = zscore(train["trend"].to_numpy())
    X = np.column_stack([np.ones(len(train)), lag1_z, hype_z, tr_z])
    y = train["y"].to_numpy(dtype=float)
    beta = ridge_fit(X, y, l2=l2)

    preds = []
    prev_y = float(train["y"].iloc[-1])
    for yr in test["year"].tolist():
        trend = yr - base_year
        lag1 = (prev_y - l_m) / (l_s if l_s != 0 else 1.0)
        hype = (float(df.loc[df["year"]==yr, "h"].values[0]) - h_m) / (h_s if h_s != 0 else 1.0)
        tr    = (trend - t_m) / (t_s if t_s != 0 else 1.0)
        x = np.array([1.0, lag1, hype, tr])
        pred = float(x @ beta)
        pred = max(pred, 0.0)
        preds.append((yr, pred))
        prev_y = pred

    test = test.merge(pd.DataFrame(preds, columns=["year","yhat"]), on="year", how="left")
    naive_level = float(train["y"].iloc[-1])
    test["yhat_naive"] = naive_level

    def mape(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
        denom = np.maximum(np.abs(y_true), EPS)
        return float(np.mean(np.abs((y_true - y_pred) / denom)))*100.0

    mape_with = mape(test["y"], test["yhat"])
    mape_base = mape(test["y"], test["yhat_naive"])
    winner = "with_sentiment" if mape_with < mape_base else "baseline"
    return {"genre": genre, "MAPE_base": round(mape_base,2), "MAPE_with": round(mape_with,2), "winner": winner}

def main():
    # Load forecasts
    base = pd.read_csv(BASE_FILE)
    wth  = pd.read_csv(WITH_FILE)

    need_base = {"genre","year","yhat","yhat_lower","yhat_upper"}
    need_with = {"genre","year","yhat","yhat_lower","yhat_upper","model","n_obs"}
    if not need_base.issubset(base.columns):
        raise SystemExit(f"[ERROR] baseline missing columns: {need_base}")
    if not need_with.issubset(wth.columns):
        raise SystemExit(f"[ERROR] with-sentiment missing columns: {need_with}")

    base["year"] = base["year"].astype(int)
    wth["year"]  = wth["year"].astype(int)

    # Align horizons by genre+year (inner join)
    join = pd.merge(
        base.rename(columns={
            "yhat":"yhat_base","yhat_lower":"yhat_lower_base","yhat_upper":"yhat_upper_base",
            "model":"model_base","n_obs":"n_obs_base"
        }),
        wth.rename(columns={
            "yhat":"yhat_with","yhat_lower":"yhat_lower_with","yhat_upper":"yhat_upper_with",
            "model":"model_with","n_obs":"n_obs_with"
        }),
        on=["genre","year"], how="inner"
    ).sort_values(["genre","year"]).reset_index(drop=True)

    if len(join) == 0:
        raise SystemExit("[ERROR] No overlapping years between baseline and with-sentiment. Check horizons.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    join.to_csv(JOINED_OUT, index=False)

    # Summaries per genre
    insights = summarize_joined(join)

    # Backtests (optional light pass)
    if AGG_FILE.exists() and HYPE_FILE.exists():
        agg = pd.read_csv(AGG_FILE); agg["year"] = agg["year"].astype(int)
        hyp = pd.read_csv(HYPE_FILE); hyp["year"]  = hyp["year"].astype(int)
        bt_rows = []
        for g in BACKTEST_GENRES:
            res = backtest_for_genre(g, agg, hyp, cutoff=BACKTEST_CUTOFF)
            if res is not None:
                bt_rows.append(res)
        if bt_rows:
            bt = pd.DataFrame(bt_rows)
            insights = insights.merge(bt, on="genre", how="left").rename(columns={
                "MAPE_base":"backtest_mape_baseline",
                "MAPE_with":"backtest_mape_with",
                "winner":"backtest_winner"
            })
        else:
            insights["backtest_mape_baseline"] = np.nan
            insights["backtest_mape_with"] = np.nan
            insights["backtest_winner"] = np.nan
    else:
        insights["backtest_mape_baseline"] = np.nan
        insights["backtest_mape_with"] = np.nan
        insights["backtest_winner"] = np.nan

    # Auto note (1-liner)
    notes = []
    for _, r in insights.iterrows():
        if r.get("baseline_sparse", False):
            note = f"{r['genre']}: baseline ≈ 0; focus on absolute delta (+{r['abs_delta']:.1f}/yr)."
        elif r["confidence_label"] == "High" and (r["cagr_pct"] is not None and r["cagr_pct"] > 0):
            note = f"{r['genre']}: solid signal, +{r['cagr_pct']:.1f}% CAGR; uplift {r['uplift_pct']:.1f}%."
        elif r["confidence_label"] == "Low":
            note = f"{r['genre']}: weak signal (n_obs={r['n_obs']}, relBW={r['rel_band_width']}). Treat as directional."
        else:
            note = f"{r['genre']}: steady; bands {r['rel_band_width']:.2f} relative."
        notes.append(note)
    insights["note"] = notes

    # Final ordering
    insights["created_at"] = datetime.now(timezone.utc).isoformat()
    insights["model_version"] = MODEL_VERSION
    col_order = [
        "genre",
        "avg_baseline","avg_with","abs_delta","uplift_pct","cagr_pct",
        "avg_band_width","rel_band_width","n_obs","model_share_non_naive",
        "baseline_sparse","confidence_label","confidence_score",
        "backtest_mape_baseline","backtest_mape_with","backtest_winner",
        "note","model_version","created_at"
    ]
    insights = insights[col_order].sort_values(
        ["confidence_label","uplift_pct","cagr_pct","abs_delta"],
        ascending=[True, False, False, False]
    )

    # Write
    tmp = OUT_FILE.with_suffix(".tmp.csv")
    insights.to_csv(tmp, index=False)
    tmp.replace(OUT_FILE)
    print(f"[OK] Wrote insights → {OUT_FILE}")
    try:
        print(insights[["genre","abs_delta","uplift_pct","cagr_pct","confidence_label","confidence_score","baseline_sparse"]].to_string(index=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()
