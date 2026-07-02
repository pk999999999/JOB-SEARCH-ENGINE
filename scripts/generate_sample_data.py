#!/usr/bin/env python3
"""
Generate synthetic candidate data for testing and demoing the ranking pipeline.

This produces a CSV file matching the schema expected by CandidateLoader
(src/data_loader.py), with a realistic mix of strong matches, weak matches,
and a small fraction of "honeypot" / low-quality profiles so the full
pipeline (skill match, experience fit, location fit, education fit,
coherence, behavioral modifier, honeypot detection) has something
meaningful to do.

Usage:
    python scripts/generate_sample_data.py --n 2000 --output data/candidates.csv
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

TITLES_STRONG = [
    "Senior AI Engineer", "Machine Learning Engineer", "Senior ML Engineer",
    "AI Research Engineer", "Deep Learning Engineer", "NLP Engineer",
    "Senior Data Scientist", "Applied Scientist", "ML Platform Engineer",
]
TITLES_MODERATE = [
    "Software Engineer", "Backend Engineer", "Data Engineer",
    "Full Stack Developer", "Data Analyst", "DevOps Engineer",
]
TITLES_WEAK = [
    "QA Engineer", "Business Analyst", "Sales Executive",
    "HR Manager", "Graphic Designer", "Content Writer",
]

COMPANIES = [
    "Acme AI", "Nimbus Labs", "Quanta Systems", "Vertex Analytics",
    "Bluepeak Tech", "Orbit Software", "Crestline Data", "Helix AI",
    "Northstar Robotics", "Indus Cloud", "Skyline ML", "Pinnacle Soft",
]

REQUIRED_SKILLS = [
    "python", "machine learning", "deep learning",
    "natural language processing", "transformers", "pytorch",
    "tensorflow", "data pipelines", "mlops", "model deployment",
]
PREFERRED_SKILLS = [
    "retrieval systems", "ranking algorithms", "recommendation systems",
    "llm", "large language models", "vector databases", "embeddings",
    "sentence transformers", "kubernetes", "docker", "aws", "gcp",
    "azure", "sql", "spark", "airflow", "ray", "fastapi", "ci/cd",
    "a/b testing",
]
UNRELATED_SKILLS = [
    "excel", "photoshop", "salesforce", "accounting", "copywriting",
    "customer service", "negotiation", "event planning",
]

LOCATIONS_PREFERRED = ["Pune", "Noida"]
LOCATIONS_GOOD = ["Mumbai", "Hyderabad", "Delhi NCR", "Gurugram"]
LOCATIONS_OK = ["Bengaluru", "Chennai", "Kolkata", "Ahmedabad", "Jaipur"]
LOCATIONS_OUTSIDE = ["San Francisco, USA", "London, UK", "Singapore", "Toronto, Canada"]

DEGREES = ["B.Tech", "M.Tech", "B.E.", "M.S.", "PhD", "MBA", "B.Sc"]
FIELDS = [
    "Computer Science", "Artificial Intelligence", "Electrical Engineering",
    "Mathematics", "Information Technology", "Mechanical Engineering",
    "Statistics",
]
INSTITUTIONS = [
    "IIT Bombay", "IIT Delhi", "NIT Trichy", "BITS Pilani",
    "Anna University", "Delhi University", "VIT Vellore",
]


def _make_skills(rng: random.Random, profile: str) -> list[dict]:
    n_required = {"strong": (6, 10), "moderate": (2, 5), "weak": (0, 2)}[profile]
    n_preferred = {"strong": (3, 8), "moderate": (1, 4), "weak": (0, 1)}[profile]

    chosen = set(rng.sample(REQUIRED_SKILLS, k=rng.randint(*n_required)))
    chosen |= set(rng.sample(PREFERRED_SKILLS, k=rng.randint(*n_preferred)))
    if profile == "weak":
        chosen |= set(rng.sample(UNRELATED_SKILLS, k=rng.randint(1, 4)))

    skills = []
    for name in chosen:
        skills.append({
            "name": name,
            "endorsements": rng.randint(0, 25),
            "duration_months": rng.randint(3, 60),
            "proficiency": rng.choice(["beginner", "intermediate", "advanced", "expert"]),
        })
    return skills


def _make_career_history(rng: random.Random, years_of_experience: float, title_pool: list[str]) -> list[dict]:
    n_roles = max(1, min(5, int(years_of_experience // 2) + 1))
    entries = []
    remaining_months = int(years_of_experience * 12)
    year = 2024
    for i in range(n_roles):
        is_current = i == 0
        dur = max(3, remaining_months // (n_roles - i)) if n_roles - i > 0 else remaining_months
        dur = min(dur, remaining_months) if remaining_months > 0 else dur
        remaining_months -= dur
        end_year = year
        start_year = max(end_year - max(1, dur // 12), end_year - 6)
        entries.append({
            "title": rng.choice(title_pool),
            "company": rng.choice(COMPANIES),
            "start_date": f"{start_year}-01-01",
            "end_date": "" if is_current else f"{end_year}-12-31",
            "duration_months": dur,
            "is_current": is_current,
            "description": rng.choice([
                "Built and shipped production ML pipelines.",
                "Led a team delivering NLP-based search features.",
                "Owned end-to-end deployment of recommendation models.",
                "Worked on data infrastructure and ETL pipelines.",
                "Contributed to internal tooling and APIs.",
                "",
            ]),
        })
        year = start_year - 1
    return entries


def _make_behavioral_signals(rng: random.Random, quality: str) -> dict:
    if quality == "high":
        return {
            "last_active_date": "2026-06-15",
            "recruiter_response_rate": round(rng.uniform(0.7, 1.0), 2),
            "avg_response_time_hours": round(rng.uniform(1, 24), 1),
            "interview_completion_rate": round(rng.uniform(0.7, 1.0), 2),
            "offer_acceptance_rate": round(rng.uniform(0.5, 1.0), 2),
            "verified_email": True,
            "verified_phone": rng.random() < 0.8,
            "linkedin_connected": rng.random() < 0.8,
        }
    if quality == "low":
        return {
            "last_active_date": rng.choice(["2022-01-01", "2021-06-01", ""]),
            "recruiter_response_rate": round(rng.uniform(0.0, 0.2), 2),
            "avg_response_time_hours": round(rng.uniform(100, 300), 1),
            "interview_completion_rate": round(rng.uniform(0.0, 0.2), 2),
            "offer_acceptance_rate": round(rng.uniform(0.0, 0.2), 2),
            "verified_email": False,
            "verified_phone": False,
            "linkedin_connected": False,
        }
    return {
        "last_active_date": "2026-04-01",
        "recruiter_response_rate": round(rng.uniform(0.3, 0.6), 2),
        "avg_response_time_hours": round(rng.uniform(24, 72), 1),
        "interview_completion_rate": round(rng.uniform(0.4, 0.7), 2),
        "offer_acceptance_rate": round(rng.uniform(0.3, 0.6), 2),
        "verified_email": rng.random() < 0.6,
        "verified_phone": rng.random() < 0.5,
        "linkedin_connected": rng.random() < 0.6,
    }


def generate_candidates(n: int, seed: int = 42, honeypot_fraction: float = 0.03) -> list[dict]:
    rng = random.Random(seed)
    records = []

    for i in range(n):
        cand_id = f"CAND_{i:06d}"

        # Pick an overall quality profile
        roll = rng.random()
        if roll < 0.30:
            profile, title_pool = "strong", TITLES_STRONG
        elif roll < 0.70:
            profile, title_pool = "moderate", TITLES_MODERATE
        else:
            profile, title_pool = "weak", TITLES_WEAK

        years_of_experience = round(max(0.0, rng.gauss(7, 4)), 1)

        loc_roll = rng.random()
        if loc_roll < 0.20:
            location = rng.choice(LOCATIONS_PREFERRED)
        elif loc_roll < 0.40:
            location = rng.choice(LOCATIONS_GOOD)
        elif loc_roll < 0.80:
            location = rng.choice(LOCATIONS_OK)
        else:
            location = rng.choice(LOCATIONS_OUTSIDE)

        skills = _make_skills(rng, profile)
        career_history = _make_career_history(rng, years_of_experience, title_pool)
        current_title = career_history[0]["title"] if career_history else rng.choice(title_pool)

        education = [{
            "degree": rng.choice(DEGREES),
            "field_of_study": rng.choice(FIELDS),
            "institution": rng.choice(INSTITUTIONS),
            "year": rng.randint(2008, 2023),
        }]

        beh_quality = rng.choices(["high", "medium", "low"], weights=[0.4, 0.4, 0.2])[0]
        behavioral_signals = _make_behavioral_signals(rng, beh_quality)

        headline = f"{current_title} | {rng.choice(COMPANIES)}"
        summary = (
            f"{years_of_experience:.1f} years of experience in "
            f"{rng.choice(['machine learning', 'software engineering', 'data science', 'AI systems'])}. "
            f"Skilled in {', '.join(s['name'] for s in skills[:4])}."
            if skills else f"{years_of_experience:.1f} years of professional experience."
        )

        salary_min = round(rng.uniform(800000, 2500000), -3)
        salary_max = salary_min + round(rng.uniform(200000, 1000000), -3)

        record = {
            "candidate_id": cand_id,
            "headline": headline,
            "summary": summary,
            "current_title": current_title,
            "skills": json.dumps(skills),
            "career_history": json.dumps(career_history),
            "education": json.dumps(education),
            "location": location,
            "years_of_experience": years_of_experience,
            "behavioral_signals": json.dumps(behavioral_signals),
            "salary_min": salary_min,
            "salary_max": salary_max,
        }

        # Inject honeypot signals into a small fraction of candidates
        if rng.random() < honeypot_fraction:
            kind = rng.choice(["expert_overclaim", "yoe_mismatch", "salary_inversion"])
            if kind == "expert_overclaim":
                extra = [{
                    "name": rng.choice(REQUIRED_SKILLS),
                    "endorsements": rng.randint(0, 3),
                    "duration_months": rng.randint(1, 3),
                    "proficiency": "expert",
                } for _ in range(2)]
                bad_skills = json.loads(record["skills"]) + extra
                record["skills"] = json.dumps(bad_skills)
            elif kind == "yoe_mismatch":
                record["years_of_experience"] = round(years_of_experience + rng.uniform(15, 25), 1)
            else:
                record["salary_min"], record["salary_max"] = record["salary_max"], record["salary_min"]

        records.append(record)

    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic candidate data.")
    parser.add_argument("--n", type=int, default=2000, help="Number of candidates to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output", type=str, default="data/candidates.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    records = generate_candidates(args.n, seed=args.seed)
    df = pd.DataFrame(records)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} synthetic candidates to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
