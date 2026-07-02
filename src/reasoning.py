"""
Reasoning generation module.

Creates grounded, template-based explanations for top-ranked candidates.
Uses only actual candidate field values — never hallucinates.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.data_loader import Candidate

logger = logging.getLogger(__name__)


class ReasoningGenerator:
    """Generate human-readable reasoning for top-ranked candidates.

    All reasoning is grounded in actual candidate data fields.
    No hallucination — every claim maps to a concrete field value.
    """

    # Title relevance keywords
    STRONG_TITLE_KEYWORDS = {
        "ai", "machine learning", "ml", "deep learning", "nlp",
        "data scientist", "research scientist",
    }
    MODERATE_TITLE_KEYWORDS = {
        "software engineer", "data engineer", "backend",
        "full stack", "platform", "devops",
    }

    # Skill category mappings for narrative
    SKILL_CATEGORIES = {
        "ml_core": {"machine learning", "deep learning", "pytorch", "tensorflow", "keras"},
        "nlp": {"nlp", "natural language processing", "transformers", "bert", "gpt", "llm"},
        "data": {"data pipelines", "spark", "airflow", "etl", "data engineering"},
        "mlops": {"mlops", "docker", "kubernetes", "ci/cd", "model deployment"},
        "cloud": {"aws", "gcp", "azure", "cloud"},
        "retrieval": {"retrieval systems", "ranking algorithms", "recommendation systems",
                      "vector databases", "embeddings", "search"},
    }

    def generate(
        self,
        candidate: Candidate,
        score_breakdown: Dict[str, float],
        rank: int,
    ) -> str:
        """Generate reasoning for a single ranked candidate.

        Args:
            candidate: The candidate profile.
            score_breakdown: Score component breakdown.
            rank: The candidate's rank (1-indexed).

        Returns:
            Multi-sentence reasoning string.
        """
        parts: List[str] = []

        # 1. Background strength
        parts.append(self._describe_background(candidate))

        # 2. Experience
        parts.append(self._describe_experience(candidate))

        # 3. Skill alignment
        parts.append(self._describe_skills(candidate, score_breakdown))

        # 4. Semantic alignment
        if score_breakdown.get("career_match", 0) > 0.6:
            parts.append(
                "High semantic alignment with target role requirements."
            )
        elif score_breakdown.get("career_match", 0) > 0.4:
            parts.append(
                "Moderate semantic alignment with target role requirements."
            )

        # 5. Activity / availability
        parts.append(self._describe_availability(candidate, score_breakdown))

        # 6. Concerns
        concerns = self._identify_concerns(candidate, score_breakdown)
        if concerns:
            parts.append(f"Concern: {'; '.join(concerns)}.")

        # Filter empty parts
        reasoning = " ".join(p for p in parts if p)

        return reasoning

    def _describe_background(self, candidate: Candidate) -> str:
        """Describe the candidate's professional background.

        Args:
            candidate: The candidate.

        Returns:
            Background description sentence.
        """
        title = candidate.current_title or candidate.headline

        if not title:
            return "Professional with relevant experience."

        title_lower = title.lower()

        # Check title strength
        for keyword in self.STRONG_TITLE_KEYWORDS:
            if keyword in title_lower:
                return f"Strong {title.strip()} background."

        for keyword in self.MODERATE_TITLE_KEYWORDS:
            if keyword in title_lower:
                return f"Relevant {title.strip()} background with transferable skills."

        return f"{title.strip()} with applicable experience."

    def _describe_experience(self, candidate: Candidate) -> str:
        """Describe years of experience.

        Args:
            candidate: The candidate.

        Returns:
            Experience description sentence.
        """
        yoe = candidate.years_of_experience

        if yoe >= 5 and yoe <= 9:
            return f"{yoe:.1f} years experience — within ideal range."
        elif yoe > 9:
            return f"{yoe:.1f} years experience — senior-level depth."
        elif yoe >= 3:
            return f"{yoe:.1f} years experience — approaching target range."
        else:
            return f"{yoe:.1f} years experience."

    def _describe_skills(
        self,
        candidate: Candidate,
        score_breakdown: Dict[str, float],
    ) -> str:
        """Describe skill alignment.

        Args:
            candidate: The candidate.
            score_breakdown: Score component values.

        Returns:
            Skill description sentence.
        """
        skill_names = set(candidate.get_skill_names())

        # Find matched categories
        matched_categories: List[str] = []
        for category, keywords in self.SKILL_CATEGORIES.items():
            if skill_names & keywords:
                matched_categories.append(category)

        if not matched_categories:
            return ""

        # Map category keys to readable names
        readable_map = {
            "ml_core": "ML/DL frameworks",
            "nlp": "NLP and language models",
            "data": "data engineering",
            "mlops": "MLOps and deployment",
            "cloud": "cloud infrastructure",
            "retrieval": "retrieval and ranking",
        }

        readable = [readable_map.get(c, c) for c in matched_categories[:3]]

        if len(readable) == 1:
            return f"Skills in {readable[0]}."
        elif len(readable) == 2:
            return f"Skills spanning {readable[0]} and {readable[1]}."
        else:
            return f"Skills spanning {', '.join(readable[:-1])}, and {readable[-1]}."

    def _describe_availability(
        self,
        candidate: Candidate,
        score_breakdown: Dict[str, float],
    ) -> str:
        """Describe availability indicators.

        Args:
            candidate: The candidate.
            score_breakdown: Score component values.

        Returns:
            Availability description sentence.
        """
        behavioral = score_breakdown.get("behavioral_modifier", 1.0)

        if behavioral >= 1.05:
            return "Recent activity and strong engagement signals indicate high availability."
        elif behavioral >= 0.90:
            return "Moderate engagement signals suggest availability."
        elif behavioral >= 0.70:
            return "Limited recent activity."
        else:
            return "Low engagement signals — may not be actively seeking."

    def _identify_concerns(
        self,
        candidate: Candidate,
        score_breakdown: Dict[str, float],
    ) -> List[str]:
        """Identify concerns about the candidate.

        Args:
            candidate: The candidate.
            score_breakdown: Score component values.

        Returns:
            List of concern strings.
        """
        concerns: List[str] = []

        # Low skill match
        if score_breakdown.get("skill_match", 1.0) < 0.3:
            concerns.append("Limited skill overlap with requirements")

        # Low experience fit
        if score_breakdown.get("experience_fit", 1.0) < 0.5:
            yoe = candidate.years_of_experience
            if yoe < 3:
                concerns.append(f"Below minimum experience ({yoe:.1f} years)")
            elif yoe > 12:
                concerns.append(f"May be overqualified ({yoe:.1f} years)")

        # Low location fit
        if score_breakdown.get("location_fit", 1.0) < 0.5:
            if candidate.location:
                concerns.append(f"Location ({candidate.location}) not preferred")

        # Low coherence
        if score_breakdown.get("coherence_factor", 1.0) < 0.85:
            concerns.append("Profile sections show inconsistency")

        # Behavioral concerns
        behavioral = score_breakdown.get("behavioral_modifier", 1.0)
        if behavioral < 0.75:
            bs = candidate.behavioral_signals
            if bs.recruiter_response_rate < 0.3:
                concerns.append("Below-average recruiter responsiveness")
            if bs.avg_response_time_hours > 72:
                concerns.append("Slow response time")

        return concerns

    def generate_batch(
        self,
        candidates: List[Candidate],
        score_breakdowns: List[Dict[str, float]],
        ranks: List[int],
    ) -> List[str]:
        """Generate reasoning for a batch of ranked candidates.

        Args:
            candidates: List of candidate objects.
            score_breakdowns: Score breakdowns per candidate.
            ranks: Rank numbers (1-indexed).

        Returns:
            List of reasoning strings.
        """
        return [
            self.generate(candidate, breakdown, rank)
            for candidate, breakdown, rank in zip(candidates, score_breakdowns, ranks)
        ]
