# app/dashboard/streamlit_app.py
# How to run locally (inside your venv):
#   streamlit run app/dashboard/streamlit_app.py
#
# This app expects the processed CSVs created in earlier phases:
#   data/processed/forecast_baseline.csv
#   data/processed/forecast_with_sentiment.csv
#   data/processed/insights_genre.csv
#   (optional) data/processed/forecast_joined.csv

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
BASE_FILE = DATA_DIR / "forecast_baseline.csv"
WITH_FILE = DATA_DIR / "forecast_with_sentiment.csv"
INSIGHTS_FILE = DATA_DIR / "insights_genre.csv"

APP_TITLE = "Netflix Genre Trend Forecasting — MVP"
DEFAULT_HORIZON = list(range(2021, 2026))  # 2021–2025

# ------------------ Helpers ------------------
@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Best-effort type fixes
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    for col in ("yhat","yhat_lower","yhat_upper"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def format_num(x, digits=1):
    try:
        x = float(x)
        if x >= 1000:
            return f"{x:,.0f}"
        return f"{x:.{digits}f}"
    except Exception:
        return str(x)

def confidence_color(label: str) -> str:
    return {"High":"#2ca02c", "Medium":"#ff7f0e", "Low":"#d62728"}.get(label, "#7f7f7f")

# ------------------ Load Data ------------------
base = load_csv(BASE_FILE).rename(columns={"yhat":"yhat_base","yhat_lower":"yhat_lower_base","yhat_upper":"yhat_upper_base"})
wth  = load_csv(WITH_FILE).rename(columns={"yhat":"yhat_with","yhat_lower":"yhat_lower_with","yhat_upper":"yhat_upper_with"})
ins  = load_csv(INSIGHTS_FILE)

# Guard rails
if ins.empty or wth.empty:
    st.error("Processed files not found. Please generate Phase 5–6 outputs first.")
    st.stop()

# Build year choices & genre canon
all_years = sorted(set(pd.concat([base.get("year", pd.Series(dtype="Int64")), wth["year"]]).dropna().astype(int)))
horizon_default = [y for y in DEFAULT_HORIZON if y in all_years] or all_years[-5:]
genres_all = sorted(wth["genre"].dropna().unique().tolist())

# ------------------ Sidebar ------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Interactive demo • Baseline vs With-Sentiment (Hype) • Plotly + Streamlit")

with st.sidebar:
    st.header("Filters")
    genres_sel = st.multiselect("Genres", options=genres_all, default=genres_all[:6])
    years_sel = st.multiselect("Years", options=all_years, default=horizon_default)
    variant = st.radio("Variant", options=["With-sentiment","Baseline","Compare"], index=0)
    conf_filter = st.selectbox("Min confidence", options=["All","Low","Medium","High"], index=0)
    show_bands = st.checkbox("Show bands (uncertainty)", value=True)
    st.divider()
    st.caption("Download data")
    st.download_button("Insights CSV", data=ins.to_csv(index=False), file_name="insights_genre.csv", mime="text/csv")
    st.download_button("With-sentiment Forecast CSV", data=wth.to_csv(index=False), file_name="forecast_with_sentiment.csv", mime="text/csv")
    if not base.empty:
        st.download_button("Baseline Forecast CSV", data=base.to_csv(index=False), file_name="forecast_baseline.csv", mime="text/csv")

# Apply confidence filter to insights
if conf_filter != "All":
    ins_view = ins[ins["confidence_label"] == conf_filter]
else:
    ins_view = ins.copy()

# ------------------ Overview: Top Genres ------------------
st.subheader("Top Genres (Avg 2021–2025)")
left, right = st.columns([2,1], gap="large")

with left:
    df_top = ins_view.copy()
    df_top = df_top.sort_values("avg_with", ascending=False).head(12)
    fig_bar = px.bar(
        df_top,
        x="avg_with", y="genre",
        color="confidence_label",
        color_discrete_map={"High":"#2ca02c","Medium":"#ff7f0e","Low":"#d62728"},
        orientation="h",
        hover_data={
            "avg_with":":.1f",
            "cagr_pct":":.1f",
            "uplift_pct":":.1f",
            "confidence_label":True
        },
        labels={"avg_with":"Avg titles/yr (with-sentiment)","genre":""},
        height=520
    )
    fig_bar.update_layout(yaxis={"categoryorder":"total ascending"}, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_bar, use_container_width=True)

with right:
    st.markdown("**Key metrics (selected genres)**")
    ins_sel = ins_view[ins_view["genre"].isin(genres_sel)].copy()
    if not ins_sel.empty:
        ins_sel_disp = ins_sel[["genre","avg_with","abs_delta","uplift_pct","cagr_pct","confidence_label"]].copy()
        ins_sel_disp["avg_with"] = ins_sel_disp["avg_with"].apply(format_num)
        ins_sel_disp["abs_delta"] = ins_sel_disp["abs_delta"].apply(lambda x: f"+{format_num(x)}")
        ins_sel_disp["uplift_pct"] = ins_sel_disp["uplift_pct"].apply(lambda x: f"{format_num(x,1)}%")
        ins_sel_disp["cagr_pct"] = ins_sel_disp["cagr_pct"].apply(lambda x: f"{format_num(x,1)}%")
        st.dataframe(ins_sel_disp.set_index("genre"), use_container_width=True, height=400)
    else:
        st.info("No genres match the confidence filter.")

st.divider()

# ------------------ Uplift vs Baseline ------------------
st.subheader("Uplift vs Baseline (symmetric %, bounded)")
df_uplift = ins_view.sort_values("uplift_pct", ascending=False).head(12)
fig_up = px.bar(
    df_uplift, x="uplift_pct", y="genre",
    color="baseline_sparse",
    color_discrete_map={True:"#1f77b4", False:"#9467bd"},
    orientation="h",
    hover_data={"uplift_pct":":.1f","abs_delta":":.1f","avg_with":":.1f","avg_baseline":":.1f","baseline_sparse":True},
    labels={"uplift_pct":"Uplift vs baseline (%, symmetric)","genre":""},
    height=500
)
fig_up.update_layout(yaxis={"categoryorder":"total ascending"}, margin=dict(l=10,r=10,t=40,b=10))
st.plotly_chart(fig_up, use_container_width=True)

st.divider()

# ------------------ Forecast Explorer ------------------
st.subheader("Forecast Explorer")
expl_a, expl_b = st.columns([2,1], gap="large")

def forecast_ribbon(df, y_col, lower_col, upper_col, name, color=None, show_bands=True):
    fig = go.Figure()
    # Bands
    if show_bands and lower_col in df.columns and upper_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df["year"], y=df[upper_col],
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=df["year"], y=df[lower_col],
            mode="lines", line=dict(width=0),
            fill="tonexty", name=f"{name} band",
            hoverinfo="skip", opacity=0.2
        ))
    # Mean
    fig.add_trace(go.Scatter(
        x=df["year"], y=df[y_col], mode="lines+markers",
        name=name
    ))
    fig.update_layout(xaxis_title="Year", yaxis_title="Titles (yhat)", margin=dict(l=10,r=10,t=10,b=10))
    return fig

with expl_a:
    g1, g2 = st.columns(2)
    genre_pick = st.selectbox("Genre", options=genres_all, index=genres_all.index(genres_sel[0]) if genres_sel else 0)
    if variant == "With-sentiment":
        df_g = wth[(wth["genre"]==genre_pick) & (wth["year"].isin(years_sel))].sort_values("year")
        fig = forecast_ribbon(df_g, "yhat_with","yhat_lower_with","yhat_upper_with", f"{genre_pick} — with-sentiment", show_bands=show_bands)
    elif variant == "Baseline":
        df_g = base[(base["genre"]==genre_pick) & (base["year"].isin(years_sel))].sort_values("year")
        fig = forecast_ribbon(df_g, "yhat_base","yhat_lower_base","yhat_upper_base", f"{genre_pick} — baseline", show_bands=show_bands)
    else:
        # Compare
        fig = go.Figure()
        df_gw = wth[(wth["genre"]==genre_pick) & (wth["year"].isin(years_sel))].sort_values("year")
        df_gb = base[(base["genre"]==genre_pick) & (base["year"].isin(years_sel))].sort_values("year")
        # with bands for with-sentiment only
        if show_bands and {"yhat_lower_with","yhat_upper_with"}.issubset(df_gw.columns):
            fig.add_trace(go.Scatter(
                x=df_gw["year"], y=df_gw["yhat_upper_with"], mode="lines", line=dict(width=0),
                showlegend=False, hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(
                x=df_gw["year"], y=df_gw["yhat_lower_with"], mode="lines", line=dict(width=0),
                fill="tonexty", name="with-sentiment band", hoverinfo="skip", opacity=0.15
            ))
        fig.add_trace(go.Scatter(x=df_gw["year"], y=df_gw["yhat_with"], mode="lines+markers", name="with-sentiment"))
        if not df_gb.empty:
            fig.add_trace(go.Scatter(x=df_gb["year"], y=df_gb["yhat_base"], mode="lines+markers", name="baseline"))
        fig.update_layout(xaxis_title="Year", yaxis_title="Titles (yhat)", margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

with expl_b:
    st.markdown("**Genre snapshot**")
    row = ins[ins["genre"]==genre_pick].head(1).to_dict(orient="records")
    if row:
        r = row[0]
        bullets = [
            f"**Avg (with-sentiment):** {format_num(r.get('avg_with',np.nan))}/yr",
            f"**Abs delta vs baseline:** +{format_num(r.get('abs_delta',np.nan))}/yr",
            f"**Uplift (symmetric):** {format_num(r.get('uplift_pct',np.nan))}%",
            f"**CAGR (2021–2025):** {format_num(r.get('cagr_pct',np.nan))}%",
            f"**Confidence:** {r.get('confidence_label','?')} ({format_num(r.get('confidence_score',np.nan),0)})",
            f"**Uncertainty (rel):** {format_num(r.get('rel_band_width',np.nan))}",
            f"**Note:** {r.get('note','')}"
        ]
        st.markdown("\n\n".join([f"- {b}" for b in bullets]))
    else:
        st.info("No insights row for selection.")

st.divider()

# ------------------ Heatmap ------------------
st.subheader("Year × Genre Heatmap (with-sentiment yhat)")
hm_df = wth[wth["year"].isin(years_sel)].pivot_table(index="genre", columns="year", values="yhat_with", aggfunc="mean")
hm_df = hm_df.reindex(index=genres_all)  # keep canonical order
fig_hm = px.imshow(
    hm_df,
    aspect="auto",
    color_continuous_scale="Blues",
    labels=dict(color="yhat"),
    height=520
)
st.plotly_chart(fig_hm, use_container_width=True)

# ------------------ Footer ------------------
st.caption("MVP • Forecasts are counts within the dataset scope, not global production totals.")
