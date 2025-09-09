# /app/etl/build_genre_year_agg.py
from pathlib import Path
import re
import sys
import pandas as pd
import numpy as np

# --- Config ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "all_df.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_FILE = OUT_DIR / "genre_year_agg.csv"

# Canonical genres (must match your GENRE_CANON.md)
CANON_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Drama", "Horror",
    "Romance", "Sci-Fi", "Thriller", "Documentary", "Family", "Fantasy"
]

# Lightweight synonym map -> canonical label (lowercase keys)
GENRE_SYNONYMS = {
    "action": "Action",
    "adventure": "Adventure",
    "animation": "Animation",
    "animated": "Animation",
    "comedy": "Comedy",
    "drama": "Drama",
    "horror": "Horror",
    "romance": "Romance",
    "romantic": "Romance",
    "sci fi": "Sci-Fi",
    "sci-fi": "Sci-Fi",
    "science fiction": "Sci-Fi",
    "thriller": "Thriller",
    "psychological thriller": "Thriller",
    "documentary": "Documentary",
    "doc": "Documentary",
    "family": "Family",
    "kids": "Family",
    "fantasy": "Fantasy",
}

# Accept common column name variants
YEAR_CANDIDATES = ["year", "Year", "release_year", "Release Year"]
GENRE_CANDIDATES = ["genre", "genres", "Genre", "Genres", "listed_in"]
VOTES_CANDIDATES = ["votes", "Votes", "imdb_votes", "IMDB Votes"]

SPLIT_PATTERN = re.compile(r"\s*[|/,;&]\s*|\s+and\s+", re.IGNORECASE)
YEAR_EXTRACT = re.compile(r"(19|20)\d{2}")

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def clean_year(val):
    """
    Handles strings like 'TV Movie 2019' -> 2019, or plain ints.
    Returns int year or np.nan.
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, np.integer, float)) and not pd.isna(val):
        y = int(val)
        return y
    s = str(val)
    m = YEAR_EXTRACT.search(s)
    return int(m.group(0)) if m else np.nan

def normalize_genres(s):
    """
    Split multi-genre strings and map to canonical labels.
    Deduplicates within a row.
    """
    if pd.isna(s):
        return []
    s = str(s).strip()
    if not s:
        return []
    # Split on common separators and 'and'
    parts = [p.strip().lower() for p in SPLIT_PATTERN.split(s) if p.strip()]
    mapped = set()
    for p in parts:
        # Direct synonym hits
        if p in GENRE_SYNONYMS:
            mapped.add(GENRE_SYNONYMS[p])
            continue
        # Try to normalize punctuation/hyphen variants
        p_norm = p.replace("-", " ").replace("_", " ").strip()
        if p_norm in GENRE_SYNONYMS:
            mapped.add(GENRE_SYNONYMS[p_norm])
            continue
        # If the token already equals a canon (case-insensitive)
        for canon in CANON_GENRES:
            if p_norm == canon.lower():
                mapped.add(canon)
                break
    return sorted(mapped)

def main():
    if not RAW_PATH.exists():
        sys.exit(f"[ERROR] Could not find raw file: {RAW_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    # Identify columns
    year_col = find_col(df, YEAR_CANDIDATES)
    genre_col = find_col(df, GENRE_CANDIDATES)
    votes_col = find_col(df, VOTES_CANDIDATES)

    if year_col is None or genre_col is None:
        sys.exit(f"[ERROR] Required columns not found. Year in {YEAR_CANDIDATES}, Genre in {GENRE_CANDIDATES}")

    # Clean year
    df["year_clean"] = df[year_col].apply(clean_year)

    # Filter desired window
    df = df[(df["year_clean"] >= 1980) & (df["year_clean"] <= 2020)].copy()

    # Normalize genres and explode
    df["canon_genres"] = df[genre_col].apply(normalize_genres)
    df = df.explode("canon_genres", ignore_index=True)
    df = df[~df["canon_genres"].isna() & (df["canon_genres"] != "")]

    # Compute aggregates
    agg = (
        df.groupby(["canon_genres", "year_clean"])
          .agg(title_count=("canon_genres", "size"),
               votes_median=(votes_col, "median") if votes_col else ("canon_genres", "size"))
          .reset_index()
    )

    # If votes not present, drop placeholder column
    if votes_col is None:
        agg = agg.drop(columns=["votes_median"])

    # Tidy names
    agg = agg.rename(columns={"canon_genres": "genre", "year_clean": "year"})

    # Basic QA checks
    # Ensure year is int
    agg["year"] = agg["year"].astype(int)
    # Keep only canonical genres (safety)
    agg = agg[agg["genre"].isin(CANON_GENRES)].copy()

    # Save
    agg.sort_values(["genre", "year"]).to_csv(OUT_FILE, index=False)

    # Log a tiny preview
    print(f"[OK] Wrote {OUT_FILE}")
    print(agg.groupby("genre")["title_count"].sum().sort_values(ascending=False).head(5))
    print(agg.head(10))

if __name__ == "__main__":
    main()
