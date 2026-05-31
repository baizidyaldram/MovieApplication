import streamlit as st
import pandas as pd
from src.recommender import movies, load_movies

# Load movies safely
try:
    if 'movies' not in locals() and movies is None:
        movies = load_movies()
    st.title("🏠 Home")
    
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
        st.metric("Features", len(movies.columns))
    
    with col3:
        st.metric("Models", "SBERT + SVD + XGB")
    
    with col4:
        st.metric("Recommendation", "Hybrid + AI")
    
    st.markdown("---")
    
    st.subheader("Dataset Preview")
    
    # Check available columns
    display_cols = ["title", "year", "vote_average", "popularity"]
    available_cols = [col for col in display_cols if col in movies.columns]
    
    st.dataframe(
        movies[available_cols].head(20),
        use_container_width=True
    )
    
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
    
except Exception as e:
    st.error(f"Error loading home page: {str(e)}")
    st.info("Please make sure all required data files are present.")
