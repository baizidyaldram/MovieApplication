from openai import OpenAI
import streamlit as st
from typing import Optional

from src.recommender import (
    movie_to_idx,
    movies,
    content_sim
)

# Global client variable
_client = None
MODEL = "openai/gpt-3.5-turbo"  # Changed to stable model

def initialize_llm(api_key: str):
    """Initialize OpenRouter client with API key"""
    global _client
    try:
        _client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        return True
    except Exception as e:
        st.error(f"Failed to initialize OpenRouter: {str(e)}")
        return False

def get_client():
    """Get or create OpenRouter client"""
    global _client
    
    if _client is not None:
        return _client
    
    # Try to get API key from session state or secrets
    api_key = None
    if 'openrouter_api_key' in st.session_state and st.session_state.openrouter_api_key:
        api_key = st.session_state.openrouter_api_key
    elif "OPENROUTER_API_KEY" in st.secrets:
        api_key = st.secrets["OPENROUTER_API_KEY"]
    
    if api_key:
        initialize_llm(api_key)
        return _client
    
    return None

def get_movie_overview(title: str, max_chars: int = 200) -> str:
    """Get movie overview from dataset"""
    if title not in movie_to_idx:
        return "No overview available"
    
    idx = movie_to_idx[title]
    overview = str(movies.iloc[idx].get("overview", ""))
    
    if len(overview) > max_chars:
        overview = overview[:max_chars] + "..."
    
    return overview if overview else "No overview available"

def explain_movie(source: str, target: str) -> str:
    """Generate AI explanation for why target movie is recommended to source movie fans"""
    
    # Check if source and target movies exist
    if source not in movie_to_idx:
        return f"Movie '{source}' not found in database"
    
    if target not in movie_to_idx:
        return f"Movie '{target}' not found in database"
    
    src_idx = movie_to_idx[source]
    tgt_idx = movie_to_idx[target]
    
    # Calculate similarity score
    similarity = round(content_sim[src_idx][tgt_idx] * 100)
    
    # Get movie details for context
    source_genres = movies.iloc[src_idx].get("genres", [])
    target_genres = movies.iloc[tgt_idx].get("genres", [])
    source_year = movies.iloc[src_idx].get("year", "Unknown")
    target_year = movies.iloc[tgt_idx].get("year", "Unknown")
    source_overview = get_movie_overview(source, 150)
    target_overview = get_movie_overview(target, 150)
    
    prompt = f"""Explain why "{target}" is recommended to a fan of "{source}".

Similarity Score: {similarity}%

Movie Details:
- {source} ({source_year}): {source_overview}
- Genres: {', '.join(source_genres[:3]) if source_genres else 'Various'}

- {target} ({target_year}): {target_overview}
- Genres: {', '.join(target_genres[:3]) if target_genres else 'Various'}

Provide a concise explanation (3-4 sentences) covering:
1. Shared themes or plot elements
2. Genre similarities
3. Why fans of the source movie would enjoy the target

Keep it engaging and informative."""
    
    try:
        client = get_client()
        if client is None:
            return "⚠️ OpenRouter API key not configured. Please add your API key in the sidebar to get AI explanations."
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a movie recommendation expert who provides clear, concise explanations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        explanation = response.choices[0].message.content
        return explanation
    
    except Exception as e:
        # Fallback explanation without AI
        return f"""📽️ Recommendation Insight:

• {target} shares {similarity}% similarity with {source}
• Both movies appeal to similar audiences
• Fans of {source} often enjoy {target} for its comparable quality and themes

Note: AI explanation unavailable: {str(e)}"""

def batch_explain_movies(source: str, targets: list) -> dict:
    """Generate explanations for multiple movies (useful for batch processing)"""
    explanations = {}
    for target in targets:
        explanations[target] = explain_movie(source, target)
    return explanations
