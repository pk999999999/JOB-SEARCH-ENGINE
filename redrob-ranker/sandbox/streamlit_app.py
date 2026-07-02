from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml

# Make sure project root is importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# -------------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Redrob AI Ranker",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------------
# Helper: load configs
# -------------------------------------------------------------------------

@st.cache_data
def load_weights_cfg() -> dict:
    with open(ROOT / "config" / "weights.yaml") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_jd():
    from src.jd_parser import JDParser
    return JDParser(ROOT / "config" / "jd_requirements.yaml").parse()


@st.cache_resource
def get_ranker(artifacts_dir: str):
    from src.jd_parser import JDParser
    from src.ranker import Ranker
    jd = JDParser(ROOT / "config" / "jd_requirements.yaml").parse()
    weights = load_weights_cfg()
    return Ranker(jd=jd, weights_cfg=weights, artifacts_dir=artifacts_dir)


# -------------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------------

def render_sidebar() -> dict:
    st.sidebar.image("https://via.placeholder.com/200x60?text=Redrob+AI", use_container_width=True)
    st.sidebar.title("⚙️ Configuration")

    st.sidebar.markdown("### Scoring Weights")
    weights = {
        "career_match": st.sidebar.slider("Career Match", 0.0, 0.5, 0.30, 0.05),
        "skill_match": st.sidebar.slider("Skill Match", 0.0, 0.5, 0.25, 0.05),
        "experience_fit": st.sidebar.slider("Experience Fit", 0.0, 0.3, 0.15, 0.05),
        "location_fit": st.sidebar.slider("Location Fit", 0.0, 0.3, 0.10, 0.05),
        "education_fit": st.sidebar.slider("Education Fit", 0.0, 0.2, 0.05, 0.05),
        "rule_adjustments": st.sidebar.slider("Rule Adjustments", 0.0, 0.3, 0.15, 0.05),
    }

    total = sum(weights.values())
    if abs(total - 1.0) > 0.01:
        st.sidebar.warning(f"⚠️ Weights sum to {total:.2f} (should be 1.0)")
    else:
        st.sidebar.success(f"✓ Weights sum to {total:.2f}")

    st.sidebar.markdown("---")
    top_k = st.sidebar.number_input("Top K candidates", min_value=1, max_value=100, value=20)
    use_reasoning = st.sidebar.checkbox("Generate reasoning", value=True)
    use_synthetic = st.sidebar.checkbox("Use synthetic data (demo)", value=True)
    n_synthetic = st.sidebar.number_input("# Synthetic candidates", 100, 10000, 1000, step=100)

    artifacts_dir = str(ROOT / "artifacts")

    return {
        "weights": weights,
        "top_k": top_k,
        "use_reasoning": use_reasoning,
        "use_synthetic": use_synthetic,
        "n_synthetic": n_synthetic,
        "artifacts_dir": artifacts_dir,
    }


# -------------------------------------------------------------------------
# Score bar component
# -------------------------------------------------------------------------

def score_bar(label: str, value: float, color: str = "#3B82F6") -> str:
    pct = int(value * 100)
    return f"""
    <div style="margin-bottom:6px">
        <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:2px">
            <span>{label}</span><span>{value:.2f}</span>
        </div>
        <div style="background:#e5e7eb;border-radius:4px;height:8px">
            <div style="background:{color};width:{pct}%;height:8px;border-radius:4px"></div>
        </div>
    </div>
    """


SCORE_COLORS = {
    "career": "#6366F1",
    "skills": "#10B981",
    "experience": "#F59E0B",
    "location": "#EF4444",
    "education": "#8B5CF6",
    "behavioral": "#06B6D4",
}


# -------------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------------

def main() -> None:
    st.title("🤖 Redrob AI Candidate Ranker")
    st.caption("Production-grade AI/ML candidate ranking for Senior AI Engineer roles")

    cfg = render_sidebar()

    # -----------------------------------------------------------------------
    # Data input
    # -----------------------------------------------------------------------
    st.header("1. Load Candidates")

    candidates = None
    upload_tab, synthetic_tab = st.tabs(["📁 Upload File", "🎲 Synthetic Data"])

    with upload_tab:
        uploaded = st.file_uploader(
            "Upload candidate file (JSON / JSONL / CSV)",
            type=["json", "jsonl", "csv"],
        )
        if uploaded is not None:
            with st.spinner("Loading candidates..."):
                from src.data_loader import DataLoader
                content = uploaded.read().decode("utf-8")
                suffix = Path(uploaded.name).suffix.lower()
                with open("/tmp/_upload" + suffix, "w") as f:
                    f.write(content)
                loader = DataLoader(strict=False)
                try:
                    candidates = loader.load("/tmp/_upload" + suffix)
                    st.success(f"✓ Loaded {len(candidates)} candidates")
                    if loader.validation_errors:
                        st.warning(f"{len(loader.validation_errors)} records had validation errors")
                except Exception as e:
                    st.error(f"Failed to load file: {e}")

    with synthetic_tab:
        if cfg["use_synthetic"] or candidates is None:
            col1, col2 = st.columns([3, 1])
            with col2:
                gen_btn = st.button("Generate", type="primary", use_container_width=True)
            with col1:
                st.info(f"Will generate {cfg['n_synthetic']} synthetic candidates for demo")

            if gen_btn or (candidates is None and "synthetic_candidates" in st.session_state):
                if gen_btn:
                    with st.spinner(f"Generating {cfg['n_synthetic']} synthetic candidates..."):
                        from src.data_loader import DataLoader, generate_synthetic_candidates
                        records = generate_synthetic_candidates(n=cfg["n_synthetic"], seed=42)
                        loader = DataLoader(strict=False)
                        st.session_state["synthetic_candidates"] = loader.load_from_records(records)
                candidates = st.session_state.get("synthetic_candidates")
                if candidates:
                    st.success(f"✓ {len(candidates)} synthetic candidates ready")

    # -----------------------------------------------------------------------
    # Ranking
    # -----------------------------------------------------------------------
    st.header("2. Run Ranking")

    if candidates is None:
        st.info("👆 Load or generate candidates above to continue.")
        return

    # Preview stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Candidates", len(candidates))
    avg_yoe = np.mean([c.years_of_experience for c in candidates])
    col2.metric("Avg Experience", f"{avg_yoe:.1f} years")
    locations = [c.location for c in candidates]
    from collections import Counter
    top_loc = Counter(locations).most_common(1)[0][0] if locations else "N/A"
    col3.metric("Top Location", top_loc)
    col4.metric("Target Top-K", cfg["top_k"])

    run_btn = st.button("🚀 Run Ranking", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Running ranking pipeline..."):
            t0 = time.perf_counter()

            # Build custom weights config
            weights_cfg = load_weights_cfg().copy()
            weights_cfg["base_score"] = cfg["weights"]

            from src.jd_parser import JDParser
            from src.ranker import Ranker

            jd = JDParser(ROOT / "config" / "jd_requirements.yaml").parse()
            ranker = Ranker(
                jd=jd,
                weights_cfg=weights_cfg,
                artifacts_dir=cfg["artifacts_dir"],
            )

            df = ranker.rank(
                candidates,
                top_k=cfg["top_k"],
                generate_reasoning=cfg["use_reasoning"],
            )
            elapsed = time.perf_counter() - t0

        st.success(f"✓ Ranked {len(candidates)} candidates in {elapsed:.2f}s")
        st.session_state["ranked_df"] = df
        st.session_state["candidates_map"] = {c.candidate_id: c for c in candidates}

    # -----------------------------------------------------------------------
    # Results display
    # -----------------------------------------------------------------------
    if "ranked_df" not in st.session_state:
        return

    df = st.session_state["ranked_df"]
    candidates_map = st.session_state.get("candidates_map", {})

    st.header("3. Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Top Score", f"{df['score'].iloc[0]:.4f}")
    col2.metric("Rank-10 Score", f"{df['score'].iloc[min(9, len(df)-1)]:.4f}")
    excluded = len(df[df.get("_honeypot_gate", pd.Series([1]*len(df))) == 0.0])
    col3.metric("Honeypot Excluded", excluded)
    col4.metric("Returned", len(df))

    # Score distribution
    if "_base_score" in df.columns:
        st.subheader("Score Distribution")
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df["score"], nbinsx=20, name="Final Score",
                marker_color="#6366F1", opacity=0.7
            ))
            fig.update_layout(
                xaxis_title="Score", yaxis_title="Count",
                height=200, margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.bar_chart(df["score"].value_counts().sort_index())

    # Score breakdown heatmap (top-20)
    if "_career_match" in df.columns:
        st.subheader("Score Breakdown (Top 20)")
        breakdown_cols = [
            "_career_match", "_skill_match", "_experience_fit",
            "_location_fit", "_education_fit", "_behavioral"
        ]
        avail = [c for c in breakdown_cols if c in df.columns]
        if avail:
            display_df = df[["candidate_id", "rank", "score"] + avail].head(20).copy()
            display_df.columns = [c.lstrip("_").replace("_", " ").title()
                                   for c in display_df.columns]
            st.dataframe(
                display_df.style.background_gradient(
                    subset=display_df.columns[3:], cmap="RdYlGn", vmin=0, vmax=1
                ).format("{:.3f}", subset=display_df.columns[2:]),
                use_container_width=True,
                height=400,
            )

    # Candidate cards
    st.subheader("Top Candidates")
    show_n = min(10, len(df))

    for _, row in df.head(show_n).iterrows():
        cand = candidates_map.get(row["candidate_id"])

        with st.expander(
            f"#{row['rank']}  {row['candidate_id']}  —  score: {row['score']:.4f}"
        ):
            left, right = st.columns([3, 2])

            with left:
                if cand:
                    st.markdown(f"**{cand.current_title}**")
                    st.markdown(f"📍 {cand.location} &nbsp;|&nbsp; 🗓️ {cand.years_of_experience:.0f} years")
                    top_skills = [
                        s["name"] for s in sorted(
                            cand.skills,
                            key=lambda x: x.get("endorsements", 0), reverse=True
                        )[:6]
                    ]
                    if top_skills:
                        st.markdown("**Skills:** " + " · ".join(
                            [f"`{s}`" for s in top_skills]
                        ))

                if row.get("reasoning"):
                    st.markdown("**Reasoning:**")
                    st.info(row["reasoning"])

            with right:
                if "_career_match" in row:
                    html = ""
                    html += score_bar("Career Match", row.get("_career_match", 0), SCORE_COLORS["career"])
                    html += score_bar("Skill Match", row.get("_skill_match", 0), SCORE_COLORS["skills"])
                    html += score_bar("Experience", row.get("_experience_fit", 0), SCORE_COLORS["experience"])
                    html += score_bar("Location", row.get("_location_fit", 0), SCORE_COLORS["location"])
                    html += score_bar("Education", row.get("_education_fit", 0), SCORE_COLORS["education"])
                    html += score_bar("Behavioral", row.get("_behavioral", 0), SCORE_COLORS["behavioral"])
                    st.markdown(html, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Download
    # -----------------------------------------------------------------------
    st.header("4. Export")
    col1, col2 = st.columns(2)

    with col1:
        required_df = df[["candidate_id", "rank", "score", "reasoning"]]
        csv_bytes = required_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Top-K CSV (required format)",
            data=csv_bytes,
            file_name=f"top{len(df)}_candidates.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        full_csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Full Diagnostic CSV",
            data=full_csv,
            file_name=f"top{len(df)}_diagnostic.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # JD info
    with st.expander("📋 Job Description Details"):
        jd = load_jd()
        st.markdown(f"**Role:** {jd.title} @ {jd.company}")
        st.markdown(f"**Experience:** {jd.experience_min}–{jd.experience_max} years (ideal: {jd.experience_ideal}y)")
        st.markdown(f"**Preferred Locations:** {', '.join(jd.preferred_locations)}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Required Skills:**")
            for sk in jd.required_skills:
                st.markdown(f"• {sk}")
        with col2:
            st.markdown("**Preferred Skills:**")
            for sk in jd.preferred_skills[:10]:
                st.markdown(f"• {sk}")


if __name__ == "__main__":
    main()