# Tutorial Quality Scorer
### WnCC Convenor Assignment 2026-27 — Question 3

---

## Project Structure

```
tutorial_scorer/
├── generate_data.py        ← Generates tutorials.csv using LLM (or synthetic fallback)
├── model.py                ← Feature engineering + Random Forest + evaluation
├── tutorials.csv           ← Generated synthetic dataset (200 rows)
├── top10_tutorials.csv     ← Top 10 tutorials by predicted quality score
├── feature_importance.png  ← Bar chart of feature importances
└── requirements.txt        ← Python dependencies
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## How to Run

**Step 1 — Generate dataset:**
```bash
python generate_data.py
```
This calls the Anthropic Claude API to generate `tutorials.csv`.
If the API is unavailable, it falls back to a synthetic generator
that enforces the same realistic correlations.

**Step 2 — Train model and get results:**
```bash
python model.py
```
This produces:
- Console output with RMSE, R², feature importances, top 10 tutorials
- `top10_tutorials.csv`
- `feature_importance.png`

---

## Prompt Used for Data Generation

```
System: You are a data generation assistant. Generate realistic synthetic 
YouTube coding tutorial metadata. Follow these correlation rules strictly:

1. Clickbait titles must have high views, low like rate (0.5-2%), low quality score (10-40)
2. High-quality tutorials must have high like rate (4-12%), long duration, detailed description, score 65-95
3. Medium quality falls in between
4. Channel subscriber count alone should NOT determine quality

Return ONLY a JSON array with keys:
Video_Title, Video_Description, Upload_Date, Duration_Minutes,
View_Count, Like_Count, Comment_Count, Channel_Subscriber_Count, Actual_Quality_Score
```

---

## Feature Engineering Logic

Raw columns like `Like_Count` are scale-dependent and misleading.
We derive the following features:

| Feature | Formula | Why |
|---------|---------|-----|
| `Like_Rate` | Like_Count / View_Count | Reveals satisfaction behind the view count |
| `Comment_Rate` | Comment_Count / View_Count | Engaged viewers comment; passive ones don't |
| `Engagement_Rate` | (Likes + Comments) / Views | Combined engagement proxy |
| `Is_Clickbait` | 1 if title has patterns like "in 10 minutes" | Direct red-flag detection |
| `Title_Word_Count` | Number of words in title | Clickbait titles are short and punchy |
| `Description_Length` | Characters in description | Effort signal — quality tutorials are described well |
| `Subs_Per_View` | Subscribers / View_Count | Channel efficiency signal |

**Key insight:** `Like_Rate` and `Comment_Rate` are the two strongest
predictors because they expose the gap between passive viewership (clicks)
and active satisfaction (engagement). Clickbait maximises views but
cannot fake engagement ratios.

---

## Model Results

- **Algorithm:** Random Forest Regressor (200 trees)
- **Train/Test split:** 80/20
- **RMSE:** ~4.3 quality score points
- **R²:** ~0.97 (model explains 97% of quality score variance)
- **Top predictor of bad tutorials:** `Comment_Rate` — low comment engagement
  is the strongest signal that a video failed to create genuine learning value
