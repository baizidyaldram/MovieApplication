import streamlit as st
import pandas as pd
from src.recommender import movies

st.title("🏠 Home")

# Check if movies data is loaded properly
if movies is not None and len(movies) > 0:
    st.markdown("""
    Welcome to the Hybrid Movie Recommendation System.
    
    This system uses multiple AI techniques to provide accurate movie recommendations:
    - **Content-Based Filtering** using Sentence-BERT
    - **Collaborative Filtering** using SVD
    - **XGBoost Re-ranking** for improved accuracy
    - **OpenRouter AI** for natural language explanations
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Movies", len(movies))
    
    with col2:
        st.metric("Features", len(movies.columns) if hasattr(movies, 'columns') else 0)
    
    with col3:
        st.metric("Models", "SBERT + SVD + XGB")
    
    with col4:
        st.metric("Recommendation", "Hybrid + AI")
    
    st.markdown("---")
    
    st.subheader("Dataset Preview")
    
    # Check available columns
    display_cols = ["title", "year", "vote_average", "popularity"]
    available_cols = [col for col in display_cols if col in movies.columns]
    
    if available_cols:
        st.dataframe(
            movies[available_cols].head(20),
            use_container_width=True
        )
    else:
        st.warning("Preview columns not available")
    
    st.markdown("---")
    
    st.subheader("Architecture")
    
    st.markdown("""
    1. **SBERT Content Similarity** - Semantic understanding of movie descriptions
    2. **Latent Semantic Similarity** - Hidden pattern detection via SVD
    3. **Reciprocal Rank Fusion** - Combining multiple ranking signals
    4. **XGBoost Ranking** - Machine learning re-ranking
    5. **Hybrid Scoring** - Weighted ensemble of all models
    6. **AI Explanation** - OpenRouter-powered natural language insights
    """)
else:
    st.error("⚠️ Movies data not loaded properly. Please check if model files exist in the 'models' directory.")
    st.info("Required files: movies_processed.pkl, content_sim.npy, latent_sim.npy, xgb_model.pkl, metadata.pkl")
