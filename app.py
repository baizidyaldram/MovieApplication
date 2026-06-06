import streamlit as st

st.set_page_config(
    page_title="Hybrid Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.block-container {
    padding-top: 1rem;
}

.metric-card {
    background-color: #262730;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

st.title("🎬 Hybrid Movie Recommendation System")

st.markdown("""
### AI-Powered Movie Recommendations

This project combines:

- Sentence-BERT Semantic Similarity
- Latent Semantic Analysis (SVD)
- XGBoost Re-Ranking
- Reciprocal Rank Fusion (RRF)
- Generative AI Explanations (OpenRouter)

Use the navigation menu on the left to explore the system.
""")

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Models",
        "4"
    )

with col2:
    st.metric(
        "Pipeline",
        "Hybrid"
    )

with col3:
    st.metric(
        "LLM",
        "OpenRouter"
    )

with col4:
    st.metric(
        "Deployment",
        "Streamlit"
    )

st.markdown("---")

st.subheader("🏗 Recommendation Pipeline")

st.markdown(""")
```text
Movie Query
     │
     ▼
SBERT Similarity
     │
     ▼
SVD Latent Similarity
     │
     ▼
Reciprocal Rank Fusion
     │
     ▼
XGBoost Re-Ranking
     │
     ▼
Hybrid Score
     │
     ▼
Top Recommendations
     │
     ▼
OpenRouter AI Explanation
