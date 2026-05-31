import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import os

@st.cache_resource
def load_assets():
    """Load all pre-trained models and data"""
    try:
        movies = pd.read_pickle("models/movies_processed.pkl")
        content_sim = np.load("models/content_sim.npy")
        latent_sim = np.load("models/latent_sim.npy")
        ranker = joblib.load("models/xgb_model.pkl")
        
        with open("models/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        
        return (movies, content_sim, latent_sim, ranker, metadata)
    except FileNotFoundError as e:
        st.error(f"Model files not found: {str(e)}")
        st.info("Please ensure all model files are in the 'models/' directory")
        raise
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        raise

# Load assets with error handling
try:
    movies, content_sim, latent_sim, ranker, metadata = load_assets()
    movie_to_idx = metadata["movie_to_idx"]
    lower_to_title = metadata["lower_to_title"]
    year_min = metadata["year_min"]
    year_max = metadata["year_max"]
except Exception as e:
    st.warning(f"Could not load all models: {str(e)}")
    # Create placeholder variables to avoid crashes
    movies = pd.DataFrame()
    content_sim = np.array([])
    latent_sim = np.array([])
    ranker = None
    metadata = {}
    movie_to_idx = {}
    lower_to_title = {}
    year_min = 0
    year_max = 1

def normalize_year(year):
    """Normalize year to 0-1 range"""
    if year_max == year_min:
        return 0.5
    return (year - year_min) / (year_max - year_min)

def fuzzy_find_movie(query):
    """Find movie by fuzzy matching"""
    if not query or len(movie_to_idx) == 0:
        return None
    
    if query in movie_to_idx:
        return query
    
    q = query.lower()
    
    if q in lower_to_title:
        return lower_to_title[q]
    
    for title in lower_to_title:
        if title.startswith(q):
            return lower_to_title[title]
    
    return None

def franchise_similarity(i, j):
    """Calculate franchise similarity between two movies"""
    if i >= len(movies) or j >= len(movies):
        return 0.0
    
    fi = movies.iloc[i].get("franchise", "")
    fj = movies.iloc[j].get("franchise", "")
    
    if len(fi) >= 5 and fi == fj:
        return 1.0
    
    return 0.0

@st.cache_resource
def build_keyword_matrix():
    """Build keyword similarity matrix"""
    if len(movies) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    
    n = len(movies)
    matrix = np.zeros((n, n), dtype=np.float32)
    
    keyword_sets = []
    for i in range(n):
        keywords = movies.iloc[i].get("keywords", [])
        if isinstance(keywords, list):
            keyword_sets.append(set(keywords))
        else:
            keyword_sets.append(set())
    
    for i in range(n):
        if not keyword_sets[i]:
            continue
        
        for j in range(i + 1, n):
            if not keyword_sets[j]:
                continue
            
            inter = len(keyword_sets[i] & keyword_sets[j])
            union = len(keyword_sets[i] | keyword_sets[j])
            
            if union > 0:
                score = inter / union
                matrix[i][j] = score
                matrix[j][i] = score
    
    return matrix

# Build keyword matrix with error handling
try:
    kw_sim_mat = build_keyword_matrix()
except Exception as e:
    st.warning(f"Could not build keyword matrix: {str(e)}")
    kw_sim_mat = np.zeros((0, 0), dtype=np.float32)

def build_feature(src_idx, cand_idx):
    """Build feature vector for XGBoost ranking"""
    if src_idx >= len(movies) or cand_idx >= len(movies):
        return [0.0] * 10
    
    m = movies.iloc[cand_idx]
    
    # Safe feature extraction with fallbacks
    popularity = m.get("popularity", 0)
    popularity_norm = popularity / (movies["popularity"].max() + 1e-9) if len(movies) > 0 else 0
    
    vote_avg = m.get("vote_average", 0)
    vote_norm = vote_avg / 10.0
    
    runtime = m.get("runtime", 0)
    runtime_norm = runtime / (movies["runtime"].max() + 1e-9) if len(movies) > 0 else 0
    
    year = m.get("year", 2000)
    year_norm = normalize_year(year)
    
    genres = m.get("genres", [])
    genres_count = len(genres) / 10.0 if isinstance(genres, list) else 0
    
    keywords = m.get("keywords", [])
    keywords_count = len(keywords) / 20.0 if isinstance(keywords, list) else 0
    
    kw_sim = kw_sim_mat[src_idx][cand_idx] if src_idx < len(kw_sim_mat) and cand_idx < len(kw_sim_mat) else 0
    content_sim_val = content_sim[src_idx][cand_idx] if src_idx < len(content_sim) and cand_idx < len(content_sim) else 0
    latent_sim_val = latent_sim[src_idx][cand_idx] if src_idx < len(latent_sim) and cand_idx < len(latent_sim) else 0
    franchise_sim = franchise_similarity(src_idx, cand_idx)
    
    return [
        popularity_norm,
        vote_norm,
        runtime_norm,
        year_norm,
        genres_count,
        keywords_count,
        kw_sim,
        content_sim_val,
        latent_sim_val,
        franchise_sim
    ]

CANDIDATE_POOL = 300

def safe_normalize(arr):
    """Safely normalize array to 0-1 range"""
    if len(arr) == 0 or arr.max() == arr.min():
        return np.zeros_like(arr)
    return MinMaxScaler().fit_transform(arr.reshape(-1, 1)).flatten()

def get_recommendations(movie_name, top_n=10):
    """Get hybrid recommendations for a movie"""
    if len(movies) == 0:
        st.error("Movies data not loaded")
        return []
    
    canonical = fuzzy_find_movie(movie_name)
    
    if canonical is None:
        st.warning(f"Movie '{movie_name}' not found in database")
        return []
    
    idx = movie_to_idx[canonical]
    
    # Get candidate movies from content and latent similarity
    content_ranked = np.argsort(content_sim[idx])[::-1][1:CANDIDATE_POOL+1]
    latent_ranked = np.argsort(latent_sim[idx])[::-1][1:CANDIDATE_POOL+1]
    
    # Reciprocal Rank Fusion
    rrf_scores = defaultdict(float)
    
    for rank, c in enumerate(content_ranked):
        rrf_scores[int(c)] += (1.0 / (60 + rank))
    
    for rank, c in enumerate(latent_ranked):
        rrf_scores[int(c)] += (1.0 / (60 + rank))
    
    candidates = list(rrf_scores.keys())
    
    if len(candidates) == 0:
        return []
    
    # Calculate raw scores
    raw_content = np.array([content_sim[idx][c] for c in candidates])
    raw_latent = np.array([latent_sim[idx][c] for c in candidates])
    raw_rrf = np.array([rrf_scores[c] for c in candidates])
    
    # XGBoost predictions (if ranker is available)
    if ranker is not None:
        raw_xgb = np.array([
            ranker.predict(np.array([build_feature(idx, c)]))[0]
            for c in candidates
        ])
    else:
        raw_xgb = raw_content  # Fallback to content similarity
    
    # Hybrid scoring
    hybrid = (
        0.50 * safe_normalize(raw_content) +
        0.25 * safe_normalize(raw_xgb) +
        0.15 * safe_normalize(raw_latent) +
        0.10 * safe_normalize(raw_rrf)
    )
    
    top_idx = np.argsort(hybrid)[::-1][:top_n]
    
    results = []
    for i in top_idx:
        movie_row = movies.iloc[candidates[i]]
        
        results.append({
            "title": movie_row.get("title", "Unknown"),
            "year": movie_row.get("year", "N/A"),
            "genres": movie_row.get("genres", []),
            "rating": movie_row.get("vote_average", 0),
            "match": round(hybrid[i] * 100, 2)
        })
    
    return results
