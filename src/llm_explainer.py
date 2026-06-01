from typing import Optional, List
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
# Use ONLY the free GPT-OSS-120B model
MODEL = "openai/gpt-oss-120b:free"  # Free model only

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
        
        # Explicitly specify the free model
        response = client.chat.completions.create(
            model=MODEL,  # Uses "openai/gpt-oss-120b:free"
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
            max_tokens=250,
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
# Add to src/llm_explainer.py (at the very end)

def explain_movie_by_query(user_query: str, recommended_movie: str) -> str:
    """
    Generate an AI explanation for why a movie matches a user's text query.
    """
    # Check if recommended movie exists
    from src.recommender import fuzzy_find_movie, movie_to_idx, movies
    
    canonical = fuzzy_find_movie(recommended_movie)
    if canonical is None:
        return f"Movie '{recommended_movie}' not found in database"
    
    idx = movie_to_idx[canonical]
    movie_data = movies.iloc[idx]
    
    # Get movie details
    movie_title = movie_data.get('title', recommended_movie)
    movie_year = movie_data.get('year', 'Unknown')
    movie_genres = movie_data.get('genres', [])
    genres_str = ', '.join(movie_genres[:3]) if movie_genres else 'Various'
    movie_overview = get_movie_overview(recommended_movie, 200)
    movie_rating = movie_data.get('vote_average', 0)
    
    prompt = f"""User is looking for movies like this: "{user_query}"

We recommended: "{movie_title}" ({movie_year})

Movie Details:
- Genres: {genres_str}
- Rating: {movie_rating}/10
- Overview: {movie_overview}

Explain in 2-3 sentences why this movie is a good match for what the user described.
Focus on how the movie's plot, genre, tone, or style matches the user's request.
Be conversational, enthusiastic, and helpful.

Example good response: "This movie is perfect for you because it combines thrilling action with clever comedy. The main character's witty one-liners and the fast-paced fight scenes match exactly what you're looking for. Plus, it has an impressive 8.5/10 rating from critics!"

Keep it concise and engaging."""
    
    try:
        client = get_client()
        if client is None:
            return self._get_fallback_text_explanation(user_query, movie_title, genres_str, movie_rating)
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a enthusiastic movie recommendation expert who gives concise, helpful explanations."
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
        return self._get_fallback_text_explanation(user_query, movie_title, genres_str, movie_rating)


def _get_fallback_text_explanation(user_query: str, movie_title: str, genres: str, rating: float) -> str:
    """Generate fallback explanation without API call"""
    # Simple keyword matching for fallback
    query_lower = user_query.lower()
    movie_lower = movie_title.lower()
    
    explanations = []
    
    # Genre-based matching
    if 'action' in query_lower and 'action' in genres.lower():
        explanations.append("• Delivers the action-packed experience you're looking for")
    if 'comedy' in query_lower or 'funny' in query_lower:
        if 'comedy' in genres.lower():
            explanations.append("• Filled with humor and comedic moments")
    if 'romantic' in query_lower or 'romance' in query_lower:
        if 'romance' in genres.lower():
            explanations.append("• Features romantic elements and emotional depth")
    if 'scary' in query_lower or 'horror' in query_lower:
        if 'horror' in genres.lower():
            explanations.append("• Delivers the thrills and scares you want")
    if 'high rating' in query_lower or 'best' in query_lower:
        if rating >= 7.5:
            explanations.append(f"• Critically acclaimed with {rating}/10 rating")
    
    if explanations:
        return f"📽️ Why {movie_title} matches your request:\n\n" + "\n".join(explanations[:3])
    else:
        return f"📽️ **{movie_title}** was recommended because it aligns with your search for '{user_query[:50]}...' with its {genres} genre profile and solid {rating}/10 rating."


def batch_explain_by_query(query: str, movies_list: list) -> dict:
    """Generate explanations for multiple movies based on a text query"""
    explanations = {}
    for movie in movies_list:
        explanations[movie] = explain_movie_by_query(query, movie)
    return explanations
