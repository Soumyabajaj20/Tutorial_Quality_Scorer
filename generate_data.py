import json
import random
import pandas as pd

SYSTEM_PROMPT = """You are a data generation assistant. Generate realistic synthetic 
YouTube coding tutorial metadata. Follow these correlation rules strictly:

1. Clickbait titles (e.g. "Learn X in 10 Minutes", "FULL Course in 1 Hour", 
   "Master X FAST") must have:
   - High View_Count (500k–5M)
   - Low Like_Count relative to views (like rate 0.5%–2%)
   - Low Comment_Count relative to views
   - Low Actual_Quality_Score (10–40)

2. High-quality tutorials must have:
   - Moderate to high Like_Count relative to views (like rate 4%–12%)
   - Longer Duration_Minutes (20–90 minutes)
   - Detailed Video_Description (100+ words)
   - Actual_Quality_Score of 65–95

3. Medium-quality tutorials should fall in between.

4. Channel_Subscriber_Count alone should NOT determine quality 
   (big channels can make bad videos, small channels can make great ones).

Return ONLY a JSON array. Each object must have exactly these keys:
Video_Title, Video_Description, Upload_Date, Duration_Minutes, 
View_Count, Like_Count, Comment_Count, Channel_Subscriber_Count, Actual_Quality_Score
"""

USER_PROMPT_TEMPLATE = """Generate {n} synthetic YouTube coding tutorial records 
as a JSON array. Mix the following types roughly equally:
- Clickbait / low-quality tutorials
- Medium quality tutorials  
- High quality in-depth tutorials

Cover a variety of programming topics: Python, JavaScript, Machine Learning, 
Web Development, Data Structures, System Design, React, SQL, C++, DevOps.

Dates should be between 2022-3-30 and 2026-3-30.
Return ONLY the JSON array, no explanation, no markdown."""


def generate_batch_with_api(n: int, batch_num: int) -> list[dict]:
    try:
        import anthropic
        client = anthropic.Anthropic()

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(n=n)
            }]
        )

        raw = response.content[0].text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        records = json.loads(raw)
        print(f"  Batch {batch_num}: generated {len(records)} records via API")
        return records

    except ImportError:
        print("  anthropic package not found — using synthetic fallback generator")
        return generate_batch_synthetic(n)
    except Exception as e:
        print(f"  API error ({e}) — using synthetic fallback generator")
        return generate_batch_synthetic(n)


def generate_batch_synthetic(n: int) -> list[dict]:
    random.seed(42)

    CLICKBAIT_TITLES = [
        "Learn Python in 10 Minutes!", "FULL JavaScript Course in 1 Hour",
        "Master Machine Learning FAST", "React in 30 Minutes for BEGINNERS",
        "SQL in 15 Minutes — Complete Course", "Learn Data Structures in 20 Minutes",
        "Python for Beginners — CRASH COURSE", "DevOps in 1 Hour: Complete Guide",
        "C++ Full Course — 45 Minutes Only!", "Build a Website in 10 Minutes",
        "Learn AI in ONE VIDEO!", "Become a Full Stack Dev FAST",
    ]

    QUALITY_TITLES = [
        "Understanding Memory Management in C++ — Deep Dive",
        "System Design Interview: Designing Twitter",
        "React Hooks Explained — useState, useEffect, useContext",
        "Dynamic Programming: Solving Hard Problems Step by Step",
        "How Databases Work: B-Trees, Indexing, and Query Optimization",
        "Building a REST API with Django — Full Project",
        "Machine Learning Math: Linear Algebra You Actually Need",
        "Advanced Python: Decorators, Generators, and Context Managers",
        "Kubernetes from Scratch — Production Setup Guide",
        "Graph Algorithms: BFS, DFS, Dijkstra Explained Visually",
        "Clean Code Principles with Real Examples",
        "Computer Networks: TCP/IP Deep Dive",
    ]

    MEDIUM_TITLES = [
        "Python Lists and Dictionaries Tutorial",
        "Introduction to SQL Joins",
        "JavaScript Async/Await Explained",
        "Getting Started with Docker",
        "React State Management with Redux",
        "Git and GitHub for Beginners",
        "Building REST APIs with Flask",
        "Introduction to Neural Networks",
        "CSS Flexbox and Grid Tutorial",
        "Python OOP — Classes and Objects",
    ]

    records = []
    types = (
        ["clickbait"] * (n // 3) +
        ["quality"]   * (n // 3) +
        ["medium"]    * (n - 2 * (n // 3))
    )
    random.shuffle(types)

    topics = ["Python", "JavaScript", "Machine Learning", "React", "SQL",
              "Data Structures", "System Design", "C++", "DevOps", "Web Dev",
              "Django", "Docker", "Kubernetes", "Git", "Neural Networks"]

    descriptions_short = [
        "Quick tutorial for beginners.", "Learn fast with this easy guide.",
        "Everything in one video!", "No experience needed.",
        "Complete course — no prior knowledge required.",
    ]
    descriptions_long = [
        ("In this comprehensive tutorial, we explore the underlying mechanisms "
         "with worked examples, edge cases, and real project applications. "
         "Prerequisites: basic programming knowledge. Code available on GitHub."),
        ("A deep dive into the internals. We cover theory, implementation, "
         "debugging strategies, and production-ready patterns. "
         "Includes exercises and solutions at each checkpoint."),
        ("This video walks through building a complete project from scratch, "
         "explaining every design decision along the way. "
         "We discuss trade-offs, common mistakes, and best practices."),
    ]

    years  = list(range(2021, 2025))
    months = list(range(1, 13))
    days   = list(range(1, 29))

    for t in types:
        date = f"{random.choice(years)}-{random.choice(months):02d}-{random.choice(days):02d}"
        topic = random.choice(topics)

        if t == "clickbait":
            title       = random.choice(CLICKBAIT_TITLES).replace("Python", topic)
            description = random.choice(descriptions_short)
            duration    = random.randint(8, 30)
            views       = random.randint(400_000, 5_000_000)
            like_rate   = random.uniform(0.005, 0.02)
            comment_rate= random.uniform(0.0005, 0.003)
            subs        = random.randint(50_000, 3_000_000)
            quality     = random.randint(10, 38)

        elif t == "quality":
            title       = random.choice(QUALITY_TITLES).replace("Python", topic)
            description = random.choice(descriptions_long)
            duration    = random.randint(25, 95)
            views       = random.randint(10_000, 600_000)
            like_rate   = random.uniform(0.05, 0.14)
            comment_rate= random.uniform(0.008, 0.03)
            subs        = random.randint(5_000, 800_000)
            quality     = random.randint(65, 95)

        else:  # medium
            title       = random.choice(MEDIUM_TITLES).replace("Python", topic)
            description = (random.choice(descriptions_short)
                           + " " + random.choice(descriptions_long)[:80])
            duration    = random.randint(12, 40)
            views       = random.randint(20_000, 400_000)
            like_rate   = random.uniform(0.02, 0.06)
            comment_rate= random.uniform(0.002, 0.010)
            subs        = random.randint(10_000, 500_000)
            quality     = random.randint(39, 64)

        likes    = int(views * like_rate)
        comments = int(views * comment_rate)

        records.append({
            "Video_Title":               title,
            "Video_Description":         description,
            "Upload_Date":               date,
            "Duration_Minutes":          duration,
            "View_Count":                views,
            "Like_Count":                likes,
            "Comment_Count":             comments,
            "Channel_Subscriber_Count":  subs,
            "Actual_Quality_Score":      quality,
        })

    return records


def generate_dataset(total: int = 200, batch_size: int = 50) -> pd.DataFrame:

    all_records = []
    batches = total // batch_size
    remainder = total % batch_size

    print(f"Generating {total} tutorial records in {batches} batches of {batch_size}...")

    for i in range(batches):
        batch = generate_batch_with_api(batch_size, i + 1)
        all_records.extend(batch)

    if remainder:
        batch = generate_batch_with_api(remainder, batches + 1)
        all_records.extend(batch)

    df = pd.DataFrame(all_records)

    numeric_cols = ["Duration_Minutes", "View_Count", "Like_Count",
                    "Comment_Count", "Channel_Subscriber_Count", "Actual_Quality_Score"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["Actual_Quality_Score"] = df["Actual_Quality_Score"].clip(1, 100)

    print(f"\nDataset generated: {len(df)} rows")
    print(f"Quality score distribution:")
    print(f"  Low  (1–40) : {(df['Actual_Quality_Score'] <= 40).sum()} videos")
    print(f"  Med (41–70) : {((df['Actual_Quality_Score'] > 40) & (df['Actual_Quality_Score'] <= 70)).sum()} videos")
    print(f"  High (71–100): {(df['Actual_Quality_Score'] > 70).sum()} videos")

    return df


if __name__ == "__main__":
    df = generate_dataset(total=200, batch_size=50)
    df.to_csv("tutorials.csv", index=False)
    print("\n✓ Saved tutorials.csv")
    print(df.head(5).to_string())
