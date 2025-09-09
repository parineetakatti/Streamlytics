# /app/etl/build_hype_index.py  (OK if your folder is /app/et1/)
from pathlib import Path
import pandas as pd
import numpy as np

# ---------- Paths ----------
# Go up two levels from this file: .../app/etl/build_hype_index.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_DIR       = PROJECT_ROOT / "data" / "external" / "hype_trends"
OUT_DIR      = PROJECT_ROOT / "data" / "processed"
OUT_MONTHLY  = OUT_DIR / "hype_index_monthly.csv"
OUT_ANNUAL   = OUT_DIR / "hype_index_annual.csv"

# ---------- Canon ----------
CANON_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Drama", "Horror",
    "Romance", "Sci-Fi", "Thriller", "Documentary", "Family", "Fantasy"
]

# Map filename keys (from hype_<key>.csv) to canonical names
FILENAME_TO_CANON = {
    "action": "Action",
    "adventure": "Adventure",
    "animation": "Animation",
    "comedy": "Comedy",
    "drama": "Drama",
    "horror": "Horror",
    "romance": "Romance",
    "sci-fi": "Sci-Fi",
    "scifi": "Sci-Fi",
    "sci fi": "Sci-Fi",
    "sci_fi": "Sci-Fi",
    "thriller": "Thriller",
    "documentary": "Documentary",
    "family": "Family",
    "fantasy": "Fantasy",
}

# ---------- Helpers ----------
def read_trends_csv(path: Path) -> pd.DataFrame:
    """
    Robustly read a Google Trends UI-exported CSV.
    Detects the real header row (starts with 'Month,' or 'Week,'),
    returns columns: date (Timestamp), value (float 0..100).
    Handles '<1' values and weekly->monthly resampling.
    """
    # Locate the header row Google adds somewhere after intro lines
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        head = line.strip().lower()
        if head.startswith("month,") or head.startswith("week,"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Could not find 'Month' or 'Week' header in {path.name}")

    df = pd.read_csv(path, skiprows=header_idx)

    # Date column may be 'Month' or 'Week'; value column is the other one
    date_col = "Month" if "Month" in df.columns else ("Week" if "Week" in df.columns else df.columns[0])
    value_cols = [c for c in df.columns if c != date_col]
    if not value_cols:
        raise ValueError(f"No value column found in {path.name}")
    val_col = value_cols[0]

    # Keep only date + value; standardize names
    df = df[[date_col, val_col]].rename(columns={date_col: "date", val_col: "value"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Clean value column:
    # - map '<1' (or '< 1') to 0.5
    # - strip spaces, commas, percent signs
    # - coerce to numeric, fill NaN with 0
    df["value"] = (
        df["value"]
        .astype(str)
        .str.strip()
        .str.replace(r"^<\s*1$", "0.5", regex=True)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
    )
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)

    # If weekly data, convert to monthly mean
    # Heuristic: many distinct days -> it's weekly
    if df["date"].dt.day.nunique() > 12:
        df = df.set_index("date").resample("M")["value"].mean().reset_index()

    return df.sort_values("date")


def main():
    if not IN_DIR.exists():
        raise SystemExit(f"[ERROR] Trends folder not found: {IN_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(IN_DIR.glob("hype_*.csv"))
    if not files:
        raise SystemExit(f"[ERROR] No files matching hype_*.csv in {IN_DIR}")

    monthly_frames = []
    for path in files:
        # Derive key from filename and map to canonical genre
        key = path.stem.replace("hype_", "").strip().lower()
        key = key.replace("_", " ").replace("-", " ")
        genre = FILENAME_TO_CANON.get(key, key.title())

        if genre not in CANON_GENRES:
            print(f"[WARN] '{path.name}' → '{genre}' not in canon; skipping.")
            continue

        df = read_trends_csv(path)

        # Keep data from 2010-01 onward
        df = df[df["date"] >= pd.Timestamp("2010-01-01")].copy()

        # Optional smoothing: 3-month rolling mean
        df["hype_index"] = df["value"].rolling(window=3, min_periods=1).mean()

        df["genre"] = genre
        monthly_frames.append(df[["genre", "date", "hype_index"]])

    if not monthly_frames:
        raise SystemExit("[ERROR] No valid monthly frames assembled.")

    # ----- Monthly output -----
    monthly = pd.concat(monthly_frames, ignore_index=True).sort_values(["genre", "date"])
    monthly.to_csv(OUT_MONTHLY, index=False)
    print(f"[OK] Wrote monthly hype index → {OUT_MONTHLY}")

    # ----- Annual output (mean, require >=8 months observed) -----
    ann = monthly.copy()
    ann["year"] = ann["date"].dt.year
    ann = (
        ann.groupby(["genre", "year"])
           .agg(months=("hype_index", "count"),
                hype_index=("hype_index", "mean"))
           .reset_index()
    )
    ann = ann[ann["months"] >= 8].drop(columns=["months"]).sort_values(["genre", "year"])
    ann.to_csv(OUT_ANNUAL, index=False)
    print(f"[OK] Wrote annual hype index → {OUT_ANNUAL}")

    # Tiny preview
    print("[PREVIEW] Monthly rows:", len(monthly))
    print("[PREVIEW] Annual rows :", len(ann))
    try:
        print(ann.groupby("genre")["hype_index"].mean().round(1).sort_values(ascending=False).head(5))
    except Exception:
        pass


if __name__ == "__main__":
    main()
