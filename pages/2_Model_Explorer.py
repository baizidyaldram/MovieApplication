import streamlit as st
import numpy as np

from src.recommender import (
    movies,
    movie_to_idx,
    content_sim,
    latent_sim
)

st.title("🔬 Model Explorer")

try:
    movie = st.selectbox(
        "Choose a Movie",
        sorted(movies["title"].tolist())
    )
    
    if movie:
        
        idx = movie_to_idx[movie]
        
        tab1, tab2, tab3 = st.tabs([
            "SBERT Similarity",
            "Latent Similarity",
            "Hybrid Score"
        ])
        
        with tab1:
            st.markdown("### Content-Based Similarity (SBERT)")
            st.markdown("Finds movies with similar descriptions using semantic understanding.")
            
            sims = content_sim[idx]
            top = sims.argsort()[::-1][1:11]
            
            rows = []
            for i in top:
                rows.append({
                    "Movie": movies.iloc[i]["title"],
                    "Similarity Score": round(float(sims[i]), 4),
                    "Year": movies.iloc[i].get("year", "N/A"),
                    "Rating": movies.iloc[i].get("vote_average", "N/A")
                })
            
            st.dataframe(rows, use_container_width=True)
        
        with tab2:
            st.markdown("### Latent Semantic Similarity (SVD)")
            st.markdown("Captures hidden patterns and latent features in movie preferences.")
            
            sims = latent_sim[idx]
            top = sims.argsort()[::-1][1:11]
            
            rows = []
            for i in top:
                rows.append({
                    "Movie": movies.iloc[i]["title"],
                    "Similarity Score": round(float(sims[i]), 4),
                    "Year": movies.iloc[i].get("year", "N/A"),
                    "Rating": movies.iloc[i].get("vote_average", "N/A")
                })
            
            st.dataframe(rows, use_container_width=True)
        
        with tab3:
            st.markdown("### Hybrid Recommendation Score")
            st.markdown("Combines SBERT and SVD using Reciprocal Rank Fusion.")
            
            # Display current movie info
            st.info(f"**Selected Movie:** {movie}")
            current_movie = movies.iloc[idx]
            if "overview" in movies.columns:
                st.write(f"**Overview:** {current_movie.get('overview', 'No description available')[:200]}...")
            
except Exception as e:
    st.error(f"Error in Model Explorer: {str(e)}")
    st.info("Please ensure all model files are properly loaded.")
