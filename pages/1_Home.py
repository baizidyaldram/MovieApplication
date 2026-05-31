# pages/1_Home.py

```python
import streamlit as st
from src.recommender import movies

st.title("🏠 Home")

st.markdown("""
Welcome to the Hybrid Movie Recommendation System.
""")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Movies",
        len(movies)
    )

with col2:
    st.metric(
        "Features",
        len(movies.columns)
    )

with col3:
    st.metric(
        "Models",
        "SBERT + SVD + XGB"
    )

with col4:
    st.metric(
        "Recommendation",
        "Hybrid"
    )

st.markdown("---")

st.subheader("Dataset Preview")

st.dataframe(
    movies[
        [
            "title",
            "year",
            "vote_average",
            "popularity"
        ]
    ].head(20),
    use_container_width=True
)

st.markdown("---")

st.subheader("Architecture")

st.markdown("""
1. SBERT Content Similarity
2. Latent Semantic Similarity
3. Reciprocal Rank Fusion
4. XGBoost Ranking
5. Hybrid Scoring
6. AI Explanation
""")