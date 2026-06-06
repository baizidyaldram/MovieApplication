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
# Add to src/recommender.py (at the very end)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Cache SBERT model to avoid reloading
@st.cache_resource
def load_sbert_model():
    """Load SBERT model for text embeddings"""
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_recommendations_by_text(query_text: str, n_recommendations: int = 5) -> list:
    """
    Get movie recommendations based on text description.
    Uses SBERT to find movies whose plots/summaries match the query.
    """
    if len(movies) == 0:
        st.error("Movies data not loaded")
        return []
    
    # Load SBERT model
    model = load_sbert_model()
    
    # Get movies with plot summaries
    movies_with_plots = movies.copy()
    
    # Create text representation for each movie
    def get_movie_text(row):
        title = str(row.get('title', ''))
        genres = row.get('genres', [])
        genres_str = " ".join(genres) if isinstance(genres, list) else ""
        overview = str(row.get('overview', ''))
        
        # Combine title, genres, and overview for better matching
        return f"{title} {genres_str} {overview}"
    
    movies_with_plots['search_text'] = movies_with_plots.apply(get_movie_text, axis=1)
    
    # Encode query
    query_embedding = model.encode([query_text])
    
    # Encode movie texts (cache after first run)
    if not hasattr(get_recommendations_by_text, 'cached_embeddings'):
        movie_texts = movies_with_plots['search_text'].fillna('').tolist()
        get_recommendations_by_text.cached_embeddings = model.encode(movie_texts)
    
    movie_embeddings = get_recommendations_by_text.cached_embeddings
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_embedding, movie_embeddings)[0]
    
    # Get top recommendations (get more for filtering)
    top_indices = np.argsort(similarities)[::-1][:n_recommendations * 3]
    
    recommendations = []
    for idx in top_indices:
        movie = movies.iloc[idx]
        
        # Boost score for high-rated movies (rating 7+ gets up to 30% boost)
        rating = movie.get('vote_average', 0)
        rating_boost = max(0, (rating - 6) / 4)  # 6=0 boost, 10=1 boost
        final_score = similarities[idx] * 0.7 + rating_boost * 0.3
        
        # Boost for popularity
        popularity = movie.get('popularity', 0)
        pop_norm = min(1.0, popularity / 100)  # Cap at 100
        final_score = final_score * 0.9 + pop_norm * 0.1
        
        recommendations.append({
            'title': movie.get('title', 'Unknown'),
            'rating': round(rating, 1),
            'year': movie.get('year', 'N/A'),
            'genres': movie.get('genres', []) if isinstance(movie.get('genres'), list) else [],
            'match': round(final_score * 100, 1),
            'similarity_score': round(similarities[idx], 3),
            'vote_count': movie.get('vote_count', 0)
        })
    
    # Remove duplicates and sort by match score
    seen_titles = set()
    unique_recs = []
    for rec in recommendations:
        if rec['title'] not in seen_titles:
            seen_titles.add(rec['title'])
            unique_recs.append(rec)
    
    # Sort by match score
    unique_recs.sort(key=lambda x: x['match'], reverse=True)
    
    # Filter to only high-quality matches (above 15% similarity threshold)
    quality_recs = [r for r in unique_recs if r['similarity_score'] > 0.15]
    
    if len(quality_recs) >= n_recommendations:
        return quality_recs[:n_recommendations]
    else:
        return unique_recs[:n_recommendations]


def filter_recommendations_by_text(recommendations: list, filter_text: str) -> list:
    """
    Filter existing recommendations based on additional text preferences.
    Example: filter_text = "funny" will prioritize comedy movies
    """
    if not filter_text or len(filter_text.strip()) < 3 or len(recommendations) == 0:
        return recommendations
    
    model = load_sbert_model()
    
    # Encode filter text
    filter_embedding = model.encode([filter_text])
    
    # Calculate relevance for each recommendation
    for rec in recommendations:
        # Combine title and genres as text
        genres_str = " ".join(rec.get('genres', []))
        movie_text = f"{rec['title']} {genres_str}"
        movie_embedding = model.encode([movie_text])
        
        relevance = cosine_similarity(filter_embedding, movie_embedding)[0][0]
        
        # Adjust match score (70% original, 30% relevance)
        rec['match'] = rec['match'] * 0.7 + relevance * 100 * 0.3
        rec['relevance_score'] = round(relevance, 3)
    
    # Re-sort by adjusted match score
    recommendations.sort(key=lambda x: x['match'], reverse=True)
    return recommendations


def get_movie_by_title(title: str) -> dict:
    """Get movie details by title"""
    canonical = fuzzy_find_movie(title)
    
    if canonical is None:
        return None
    
    idx = movie_to_idx[canonical]
    movie = movies.iloc[idx]
    
    return {
        'title': movie.get('title', 'Unknown'),
        'year': movie.get('year', 'N/A'),
        'genres': movie.get('genres', []) if isinstance(movie.get('genres'), list) else [],
        'rating': movie.get('vote_average', 0),
        'overview': movie.get('overview', 'No overview available'),
        'popularity': movie.get('popularity', 0)
    }
