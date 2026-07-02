from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import (
    BehavioralSignals,
    Candidate,
    CandidateLoader,
    CareerEntry,
    EducationEntry,
    SkillEntry,
)
from src.jd_parser import JDParser
from src.ranker import RankingPipeline

ROOT = Path(__file__).parent.parent


def make_candidates(n: int = 50, seed: int = 7) -> list[Candidate]:
    rng = np.random.default_rng(seed)
    titles = ["Senior AI Engineer", "ML Engineer", "Software Engineer", "Data Analyst"]
    locations = ["Pune", "Mumbai", "Bengaluru", "San Francisco, USA"]
    skills_pool = [
        "python", "machine learning", "deep learning", "nlp", "pytorch",
        "tensorflow", "docker", "kubernetes", "aws", "sql",
    ]
    candidates = []
    for i in range(n):
        n_skills = rng.integers(1, 6)
        skills = [
            SkillEntry(
                name=str(s),
                endorsements=int(rng.integers(0, 20)),
                duration_months=int(rng.integers(1, 48)),
                proficiency=rng.choice(["beginner", "intermediate", "advanced", "expert"]),
            )
            for s in rng.choice(skills_pool, size=n_skills, replace=False)
        ]
        yoe = float(rng.uniform(0, 15))
        candidates.append(Candidate(
            candidate_id=f"CAND_{i:04d}",
            headline=f"{rng.choice(titles)} candidate",
            summary="Experienced professional.",
            current_title=str(rng.choice(titles)),
            skills=skills,
            career_history=[
                CareerEntry(
                    title=str(rng.choice(titles)), company="Acme",
                    is_current=True, duration_months=int(rng.integers(6, 60)),
                )
            ],
            education=[EducationEntry(degree="B.Tech", field_of_study="Computer Science")],
            location=str(rng.choice(locations)),
            years_of_experience=yoe,
            behavioral_signals=BehavioralSignals(
                recruiter_response_rate=float(rng.uniform(0, 1)),
                avg_response_time_hours=float(rng.uniform(1, 100)),
            ),
            salary_min=1_000_000.0,
            salary_max=2_000_000.0,
        ))
    return candidates


@pytest.fixture
def pipeline_factory(fake_artifacts_factory, jd):
    """Returns a callable that builds a RankingPipeline with fake (but
    well-formed) precomputed artifacts for a given candidate list.
    """

    def _build(candidates: list[Candidate], top_k: int = 10) -> RankingPipeline:
        artifacts_dir = fake_artifacts_factory([c.candidate_id for c in candidates])
        return RankingPipeline(
            candidates=candidates,
            jd=jd,
            artifacts_dir=artifacts_dir,
            weights_config=ROOT / "config" / "weights.yaml",
            top_k=top_k,
        )

    return _build


class TestOutputCSVSchema:
    """Verify the output of RankingPipeline.rank() has exactly the
    required columns, correct types, and is properly sorted.
    """

    REQUIRED_COLUMNS = ["candidate_id", "rank", "score", "reasoning"]

    def test_required_columns_present(self, pipeline_factory):
        candidates = make_candidates(50)
        df = pipeline_factory(candidates).rank()
        for col in self.REQUIRED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_extra_columns(self, pipeline_factory):
        candidates = make_candidates(20)
        df = pipeline_factory(candidates).rank()
        assert list(df.columns) == self.REQUIRED_COLUMNS

    def test_candidate_id_is_string(self, pipeline_factory):
        df = pipeline_factory(make_candidates(20)).rank()
        assert df["candidate_id"].dtype == object

    def test_rank_starts_at_one_and_is_consecutive(self, pipeline_factory):
        df = pipeline_factory(make_candidates(20), top_k=10).rank()
        assert list(df["rank"]) == list(range(1, len(df) + 1))

    def test_reasoning_is_nonempty_string(self, pipeline_factory):
        df = pipeline_factory(make_candidates(20), top_k=5).rank()
        assert all(isinstance(r, str) and len(r) > 10 for r in df["reasoning"])

    def test_scores_descending(self, pipeline_factory):
        df = pipeline_factory(make_candidates(50)).rank()
        scores = df["score"].tolist()
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1] - 1e-9, (
                f"Score not descending at position {i}: {scores[i]} < {scores[i+1]}"
            )

    def test_no_duplicate_candidate_ids(self, pipeline_factory):
        df = pipeline_factory(make_candidates(50)).rank()
        assert df["candidate_id"].nunique() == len(df)

    def test_no_null_values_in_required_columns(self, pipeline_factory):
        df = pipeline_factory(make_candidates(30)).rank()
        for col in self.REQUIRED_COLUMNS:
            assert df[col].isnull().sum() == 0, f"Null values found in {col}"

    def test_top_k_respected(self, pipeline_factory):
        candidates = make_candidates(50)
        for k in [5, 10, 20]:
            df = pipeline_factory(candidates, top_k=k).rank()
            assert len(df) <= k

    def test_top_k_larger_than_pool_returns_all(self, pipeline_factory):
        candidates = make_candidates(10)
        df = pipeline_factory(candidates, top_k=100).rank()
        assert len(df) == 10

    def test_write_and_reload_csv(self, pipeline_factory):
        pipeline = pipeline_factory(make_candidates(20), top_k=5)
        df = pipeline.rank()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "top.csv"
            pipeline.export_csv(df, out_path)
            assert out_path.exists()
            reloaded = pd.read_csv(out_path)
            assert list(reloaded.columns) == self.REQUIRED_COLUMNS
            assert len(reloaded) == len(df)


class TestHoneypotIntegration:
    """Verify candidates flagged as honeypots in the precomputed
    artifacts never appear in the final ranking output.
    """

    def test_flagged_honeypot_excluded_from_output(self, fake_artifacts_factory, jd):
        candidates = make_candidates(20)
        artifacts_dir = fake_artifacts_factory([c.candidate_id for c in candidates])

        # Mark the first candidate as a honeypot in the artifacts.
        honeypot_path = artifacts_dir / "honeypot_flags.parquet"
        hp_df = pd.read_parquet(honeypot_path)
        hp_df.loc[0, "is_honeypot"] = True
        hp_df.loc[0, "gate_value"] = 0.0
        hp_df.to_parquet(honeypot_path, index=False)
        excluded_id = hp_df.loc[0, "candidate_id"]

        pipeline = RankingPipeline(
            candidates=candidates, jd=jd, artifacts_dir=artifacts_dir,
            weights_config=ROOT / "config" / "weights.yaml", top_k=20,
        )
        df = pipeline.rank()
        assert df[df["candidate_id"] == excluded_id]["score"].iloc[0] == pytest.approx(0.0)


class TestCandidateLoaderRoundTrip:
    """Validate that CandidateLoader correctly reads back a CSV produced
    by the sample-data generator (and CSVs in general).
    """

    def test_csv_roundtrip(self, tmp_path):
        import json

        rows = []
        for i in range(15):
            rows.append({
                "candidate_id": f"CSV_{i:03d}",
                "headline": "ML Engineer",
                "summary": "Built ML pipelines.",
                "current_title": "ML Engineer",
                "skills": json.dumps([{"name": "python", "endorsements": 5, "duration_months": 12}]),
                "career_history": json.dumps([{
                    "title": "ML Engineer", "company": "Acme",
                    "is_current": True, "duration_months": 24,
                }]),
                "education": json.dumps([{"degree": "B.Tech", "field_of_study": "CS"}]),
                "location": "Pune",
                "years_of_experience": 5.0,
                "behavioral_signals": json.dumps({"recruiter_response_rate": 0.5}),
            })
        path = tmp_path / "candidates.csv"
        pd.DataFrame(rows).to_csv(path, index=False)

        loader = CandidateLoader(path)
        candidates = loader.load()
        assert len(candidates) == 15
        assert candidates[0].candidate_id == "CSV_000"
        assert candidates[0].get_skill_names() == ["python"]

    def test_missing_required_column_raises(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame({"headline": ["X"]}).to_csv(path, index=False)
        loader = CandidateLoader(path)
        with pytest.raises(ValueError):
            loader.load_dataframe()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CandidateLoader(tmp_path / "does_not_exist.csv")


class TestJDParserBasics:
    def test_required_and_preferred_skills_loaded(self, jd):
        assert "python" in jd.get_required_skills_lower()
        assert len(jd.preferred_skills) > 0

    def test_job_title_loaded(self, jd):
        assert jd.job_title == "Senior AI Engineer"

    def test_get_jd_text_includes_skills(self, jd):
        text = jd.get_jd_text()
        assert "python" in text.lower()
