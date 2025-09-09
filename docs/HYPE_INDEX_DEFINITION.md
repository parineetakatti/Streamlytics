# Hype Index — Definition (MVP)

Source: Google Trends (Worldwide), Monthly, Jan 2010 → Present.
Per-genre input: one or two search terms from the canon’s keyword hints.
Normalization: use Trends’ 0–100 index per series.
Smoothing (optional): 3-month rolling average to reduce noise.
Annual alignment for modeling: mean of the 12 months per year.
Use: leading indicator to build a “with-sentiment” forecast variant for a subset of genres.

Caveats:
- Representativeness (search ≠ views), term ambiguity, region differences, news spikes.
- MVP uses Global; future work: US/India/EU splits.
# Hype Index — Definition (MVP)

**Source:** Google Trends (Worldwide)  
**Granularity:** Monthly, Jan 2010 → Present  
**Normalization:** Native 0–100 index from Google Trends  
**Smoothing (optional):** 3-month rolling average  
**Annual alignment (for modeling):** Mean of the 12 months in each calendar year  
**Use:** Leading indicator to create a “with-sentiment” forecast variant for selected genres

**Caveats:** Search interest ≠ viewing; term ambiguity; regional differences; news spikes.  
MVP uses **Worldwide**; future: US/India/EU splits.

---

## Final queries (pick **Topic** when available; fallback to Search term)

| Genre        | Trends selection to use (preferred)                          | Type   | Fallback search term (if Topic not shown) |
|--------------|---------------------------------------------------------------|--------|-------------------------------------------|
| Action       | **Action film (Film genre)**                                  | Topic  | action movie                              |
| Adventure    | **Adventure film (Film genre)**                               | Topic  | adventure movie                           |
| Animation    | **Animated film (Film genre)** *(or “Animation (Film genre)”)*| Topic  | animated movie                            |
| Comedy       | **Comedy film (Film genre)**                                  | Topic  | comedy movie                              |
| Drama        | **Drama film (Film genre)**                                   | Topic  | drama movie                               |
| Horror       | **Horror film (Film genre)**                                  | Topic  | horror movie                              |
| Romance      | **Romance film (Film genre)**                                 | Topic  | romantic movie                            |
| Sci-Fi       | **Science fiction film (Film genre)** *(or “Science fiction” Topic)* | Topic  | science fiction                           |
| Thriller     | **Thriller film (Film genre)**                                | Topic  | thriller movie                            |
| Documentary  | **Documentary film (Film genre)**                             | Topic  | documentary                               |
| Family       | **Family film (Film genre)**                                  | Topic  | family movie                              |
| Fantasy      | **Fantasy film (Film genre)**                                 | Topic  | fantasy movie                             |

> **How to choose:** In Google Trends, type the genre name and pick the item labeled **Topic** that clearly refers to the **film genre**. If a Topic isn’t available, use the fallback **Search term**.

---

## Export instructions (one CSV per genre)

1. Go to **trends.google.com**.  
2. Set filters at the top:
   - **Region:** Worldwide  
   - **Time range:** 2010–present  
   - **Search type:** Web Search  
   - **Category:** *(optional)* Arts & Entertainment
3. Ensure **only one series** is shown (remove any comparisons).
4. Click the **download icon** on “Interest over time” → **CSV**.
5. Rename the file to: `hype_<genre>.csv`  
   Examples: `hype_sci-fi.csv`, `hype_thriller.csv`, `hype_drama.csv`
6. Move each file into: `/data/external/hype_trends/`.

---

## With-sentiment subset (MVP)

We will compute the **with-sentiment** forecast variant first for these genres:

**Sci-Fi, Thriller, Action, Drama, Horror, Romance**

---

## Export log (fill this as you download)

| Genre       | Chosen query text (exactly as shown in Trends) | Type (Topic/Search term) | File name            | Date range (e.g., Jan 2010 – Sep 2025) | Notes |
|-------------|--------------------------------------------------|---------------------------|----------------------|-----------------------------------------|-------|
| Action      |                                                  |                           | hype_action.csv      |                                         |       |
| Adventure   |                                                  |                           | hype_adventure.csv   |                                         |       |
| Animation   |                                                  |                           | hype_animation.csv   |                                         |       |
| Comedy      |                                                  |                           | hype_comedy.csv      |                                         |       |
| Drama       |                                                  |                           | hype_drama.csv       |                                         |       |
| Horror      |                                                  |                           | hype_horror.csv      |                                         |       |
| Romance     |                                                  |                           | hype_romance.csv     |                                         |       |
| Sci-Fi      |                                                  |                           | hype_sci-fi.csv      |                                         |       |
| Thriller    |                                                  |                           | hype_thriller.csv    |                                         |       |
| Documentary |                                                  |                           | hype_documentary.csv |                                         |       |
| Family      |                                                  |                           | hype_family.csv      |                                         |       |
| Fantasy     |                                                  |                           | hype_fantasy.csv     |                                         |       |
## Export log

| Genre        | Chosen query text (exactly as shown in Trends) | Type (Topic/Search term) | File name            | Date range (e.g., Jan 2010 – Sep 2025) | Notes |
|--------------|--------------------------------------------------|---------------------------|----------------------|-----------------------------------------|-------|
| Action       |                                                  |                           | hype_action.csv      |                                         |       |
| Adventure    |                                                  |                           | hype_adventure.csv   |                                         |       |
| Animation    |                                                  |                           | hype_animation.csv   |                                         |       |
| Comedy       |                                                  |                           | hype_comedy.csv      |                                         |       |
| Drama        |                                                  |                           | hype_drama.csv       |                                         |       |
| Horror       |                                                  |                           | hype_horror.csv      |                                         |       |
| Romance      |                                                  |                           | hype_romance.csv     |                                         |       |
| Sci-Fi       |                                                  |                           | hype_sci-fi.csv      |                                         |       |
| Thriller     |                                                  |                           | hype_thriller.csv    |                                         |       |
| Documentary  |                                                  |                           | hype_documentary.csv |                                         |       |
| Family       |                                                  |                           | hype_family.csv      |                                         |       |
| Fantasy      |                                                  |                           | hype_fantasy.csv     |                                         |       |
