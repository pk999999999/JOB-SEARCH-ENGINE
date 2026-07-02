from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader, generate_synthetic_candidates
from src.jd_parser import JDParser
from src.ranker import Ranker


def load_weights() -> dict:
    with open("config/weights.yaml") as f:
        return yaml.safe_load(f)


def make_small_candidates(n: int = 50) -> list:
    records = generate_synthetic_candidates(n=n, seed=7)
    loader = DataLoader(strict=False)
    return loader.load_from_records(records)


class TestOutputCSVSchema:
    """Verify the output CSV has exactly the required columns and types."""

    REQUIRED_COLUMNS = ["candidate_id", "rank", "score", "reasoning"]

    def setup_method(self):
        self.jd = JDParser("config/jd_requirements.yaml").parse()
        self.weights = load_weights()
        self.candidates = make_small_candidates(50)
        self.ranker = Ranker(
            jd=self.jd,
            weights_cfg=self.weights,
            artifacts_dir="artifacts/",
        )

    def _get_output_df(self, top_k: int = 10) -> pd.DataFrame:
        return self.ranker.rank(self.candidates, top_k=top_k, generate_reasoning=True)

    def test_required_columns_present(self):
        df = self._get_output_df()
        for col in self.REQUIRED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_candidate_id_is_string(self):
        df = self._get_output_df()
        assert df["candidate_id"].dtype == object

    def test_rank_is_integer(self):
        df = self._get_output_df()
        assert df["rank"].dtype in (int, np.int64, np.int32)

    def test_score_is_float(self):
        df = self._get_output_df()
        assert df["score"].dtype in (float, np.float32, np.float64)

    def test_reasoning_is_string(self):
        df = self._get_output_df()
        assert df["reasoning"].dtype == object
        assert all(isinstance(r, str) for r in df["reasoning"])

    def test_rank_starts_at_one(self):
        df = self._get_output_df()
        assert df["rank"].iloc[0] == 1

    def test_rank_is_consecutive(self):
        df = self._get_output_df(top_k=10)
        assert list(df["rank"]) == list(range(1, len(df) + 1))

    def test_scores_descending(self):
        df = self._get_output_df()
        scores = df["score"].tolist()
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Score not descending at position {i}: {scores[i]} < {scores[i+1]}"
            )

    def test_scores_in_valid_range(self):
        df = self._get_output_df()
        assert df["score"].min() >= 0.0
        assert df["score"].max() <= 1.0

    def test_no_duplicate_candidate_ids(self):
        df = self._get_output_df()
        assert df["candidate_id"].nunique() == len(df)

    def test_no_null_values_in_required_cols(self):
        df = self._get_output_df()
        for col in self.REQUIRED_COLUMNS:
            assert df[col].isnull().sum() == 0, f"Null values in {col}"

    def test_top_k_respected(self):
        for k in [5, 10, 20]:
            df = self._get_output_df(top_k=k)
            assert len(df) <= k

    def test_write_and_reload_csv(self):
        df = self._get_output_df(top_k=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "top.csv"
            self.ranker.write_output_csv(df, out_path)
            assert out_path.exists()
            reloaded = pd.read_csv(out_path)
            assert list(reloaded.columns) == self.REQUIRED_COLUMNS
            assert len(reloaded) == len(df)

    def test_reasoning_not_empty_for_top_k(self):
        df = self._get_output_df(top_k=5)
        assert all(len(r) > 20 for r in df["reasoning"])

    def test_reasoning_contains_score(self):
        df = self._get_output_df(top_k=3)
        for r in df["reasoning"]:
            assert "score" in r.lower() or any(
                char.isdigit() for char in r
            ), f"Reasoning lacks numeric content: {r[:80]}"


class TestDataLoaderValidation:
    """Validate schema enforcement in DataLoader."""

    def test_valid_records_loaded(self):
        records = generate_synthetic_candidates(n=20, seed=1)
        loader = DataLoader(strict=False)
        candidates = loader.load_from_records(records)
        assert len(candidates) == 20

    def test_invalid_record_skipped_in_lenient_mode(self):
        records = [{"candidate_id": "GOOD", "years_of_experience": 5.0},
                   {"not_a_valid_field": True}]
        loader = DataLoader(strict=False)
        candidates = loader.load_from_records(records)
        assert len(candidates) >= 1

    def test_narrative_text_built(self):
        records = generate_synthetic_candidates(n=5, seed=3)
        loader = DataLoader(strict=False)
        candidates = loader.load_from_records(records)
        for c in candidates:
            assert len(c.narrative_text) > 0

    def test_years_of_experience_non_negative(self):
        records = [{"candidate_id": "TEST", "years_of_experience": -5}]
        loader = DataLoader(strict=False)
        candidates = loader.load_from_records(records)
        assert all(c.years_of_experience >= 0 for c in candidates)

    def test_csv_roundtrip(self):
        import json
        import tempfile
        records = generate_synthetic_candidates(n=10, seed=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            rows = []
            for r in records:
                row = dict(r)
                for f in ["skills", "career_history", "education", "behavioral_signals"]:
                    row[f] = json.dumps(row[f])
                rows.append(row)
            pd.DataFrame(rows).to_csv(str(path), index=False)
            loader = DataLoader(strict=False)
            loaded = loader.load(path)
            assert len(loaded) == 10