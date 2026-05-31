import numpy as np
import pandas as pd
import streamlit as st

from src.recommender import (
    movies,
    movie_to_idx,
    get_recommendations,
    content_sim
)

def diversity_at_k(recommendations, k=5):
    """Calculate diversity of recommendations at k"""
    if len(recommendations) == 0:
        return 0
    
    indices = []
    for m in recommendations[:k]:
        if m["title"] in movie_to_idx:
            indices.append(movie_to_idx[m["title"]])
    
    if len(indices) < 2:
        return 0
    
    sims = []
    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            if indices[i] < len(content_sim) and indices[j] < len(content_sim):
                sims.append(content_sim[indices[i]][indices[j]])
    
    if len(sims) == 0:
        return 0
    
    return float(1 - np.mean(sims))

def run_evaluation():
    """Run comprehensive evaluation on test movies"""
    
    test_movies = [
        "The Dark Knight",
        "Inception",
        "Iron Man",
        "Avatar",
        "Titanic",
        "The Godfather",
        "Pulp Fiction",
        "Interstellar",
        "Fight Club"
    ]
    
    results = []
    progress_bar = st.progress(0)
    
    for i, movie in enumerate(test_movies):
        if movie not in movie_to_idx:
            st.warning(f"Movie '{movie}' not found, skipping...")
            continue
        
        # Get recommendations
        recs = get_recommendations(movie, 10)
        
        if len(recs) == 0:
            continue
        
        # Calculate metrics
        diversity = diversity_at_k(recs)
        
        quality = np.mean([r["rating"] for r in recs[:5]]) if len(recs) >= 5 else 0
        
        # Calculate average similarity
        similarities = []
        src_idx = movie_to_idx[movie]
        for r in recs[:5]:
            if r["title"] in movie_to_idx:
                tgt_idx = movie_to_idx[r["title"]]
                if src_idx < len(content_sim) and tgt_idx < len(content_sim):
                    similarities.append(content_sim[src_idx][tgt_idx])
        
        similarity = np.mean(similarities) if similarities else 0
        
        results.append({
            "Movie": movie,
            "AvgSim@5": round(similarity, 4),
            "Diversity@5": round(diversity, 4),
            "VoteQuality@5": round(quality, 2)
        })
        
        progress_bar.progress((i + 1) / len(test_movies))
    
    progress_bar.empty()
    
    if len(results) == 0:
        st.warning("No evaluation results generated")
        return pd.DataFrame()
    
    return pd.DataFrame(results)

def detailed_evaluation():
    """Run detailed evaluation with additional metrics"""
    results = run_evaluation()
    
    if len(results) == 0:
        return None
    
    # Calculate overall statistics
    overall_stats = {
        "Average Similarity@5": results["AvgSim@5"].mean(),
        "Average Diversity@5": results["Diversity@5"].mean(),
        "Average Vote Quality@5": results["VoteQuality@5"].mean(),
        "Min Similarity": results["AvgSim@5"].min(),
        "Max Similarity": results["AvgSim@5"].max(),
        "Std Diversity": results["Diversity@5"].std()
    }
    
    return {
        "per_movie": results,
        "overall_stats": overall_stats
    }
