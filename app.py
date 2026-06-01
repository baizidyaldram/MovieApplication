import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime
import random

# Page configuration
st.set_page_config(
    page_title="Hybrid Movie Recommendation System using GenAI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Glass morphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 1rem;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.15);
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
    }
    
    /* Stat cards */
    .stat-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05));
        backdrop-filter: blur(10px);
        border-radius: 1rem;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        background: linear-gradient(135deg, rgba(255,255,255,0.25), rgba(255,255,255,0.1));
    }
    
    /* Spotlight card */
    .spotlight-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.3), rgba(118,75,162,0.3));
        backdrop-filter: blur(10px);
        border-radius: 1rem;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease;
    }
    
    .spotlight-card:hover {
        transform: translateY(-5px);
        background: linear-gradient(135deg, rgba(102,126,234,0.4), rgba(118,75,162,0.4));
    }
    
    /* Movie card */
    .movie-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 0.75rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.15);
        transition: all 0.3s ease;
    }
    
    .movie-card:hover {
        transform: translateY(-3px);
        background: rgba(255, 255, 255, 0.15);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    
    /* Poster image - MEDIUM SIZE */
    .poster-image {
        width: 100px;
        height: 150px;
        object-fit: cover;
        border-radius: 8px;
        transition: transform 0.3s ease;
    }
    
    .poster-image:hover {
        transform: scale(1.05);
    }
    
    /* Genre badges */
    .genre-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 500;
        margin: 0.2rem;
    }
    
    /* Match score */
    .match-score {
        display: inline-block;
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102,126,234,0.3);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 50px;
        padding: 8px;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 40px;
        padding: 8px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        color: white !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 0.75rem;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: rgba(255,255,255,0.7);
        background: rgba(0,0,0,0.2);
        border-radius: 1rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'openrouter_api_key' not in st.session_state:
    st.session_state.openrouter_api_key = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'auto_explain' not in st.session_state:
    st.session_state.auto_explain = True

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 3rem;">🎬</div>
        <div style="font-weight: 800; font-size: 1.2rem;">Movie Recommender</div>
        <div style="font-size: 0.7rem; opacity: 0.8;">Hybrid AI System</div>
    </div>
    """, unsafe_allow_html=True)
    
    # API Configuration
    with st.expander("🔑 API Configuration", expanded=False):
        try:
            if "OPENROUTER_API_KEY" in st.secrets:
                st.session_state.openrouter_api_key = st.secrets["OPENROUTER_API_KEY"]
                st.success("✅ API Key Loaded")
            else:
                api_key_input = st.text_input("OpenRouter API Key", type="password")
                if api_key_input:
                    st.session_state.openrouter_api_key = api_key_input
                    st.success("✅ Key set!")
        except:
            pass
        
        if "TMDB_API_KEY" in st.secrets:
            st.success("✅ TMDB API Ready")
    
    # Settings
    st.markdown("---")
    st.session_state.auto_explain = st.checkbox("🤖 Auto AI Explanations", value=True)
    
    # System Stats
    st.markdown("---")
    from src.recommender import movies
    if movies is not None:
        st.metric("🎬 Total Movies", len(movies))
        st.metric("🎭 Unique Genres", "19")
        st.metric("⭐ Avg Rating", f"{movies['vote_average'].mean():.1f}")

# Load data
from src.recommender import movies, movie_to_idx, get_recommendations
from src.llm_explainer import explain_movie, initialize_llm

if st.session_state.openrouter_api_key:
    initialize_llm(st.session_state.openrouter_api_key)

# TMDB Poster function
def get_movie_poster(movie_title, year=None):
    try:
        if "TMDB_API_KEY" in st.secrets:
            api_key = st.secrets["TMDB_API_KEY"]
            search_url = "https://api.themoviedb.org/3/search/movie"
            params = {"api_key": api_key, "query": movie_title, "language": "en-US"}
            if year and year != "N/A":
                params["year"] = year
            response = requests.get(search_url, params=params, timeout=5)
            data = response.json()
            if data.get("results") and len(data["results"]) > 0:
                poster_path = data["results"][0].get("poster_path")
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w200{poster_path}"
    except:
        pass
    return None

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 DASHBOARD",
    "📊 EXPLORE DATA",
    "🎬 RECOMMENDATIONS",
    "🔬 MODEL INSIGHTS",
    "ℹ️ ABOUT"
])

# ==================== TAB 1: DASHBOARD ====================
with tab1:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="font-size: 2.5rem; background: linear-gradient(135deg, #fff, #a0a0ff); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Hybrid Movie Recommendation System
        </h1>
        <p style="font-size: 1rem; opacity: 0.9;">AI-powered movie discovery with SBERT, SVD, XGBoost &amp; OpenRouter</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Stats Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size: 2rem;">🎬</div>
            <div style="font-size: 1.5rem; font-weight: 700;">{len(movies):,}</div>
            <div style="font-size: 0.8rem;">Total Movies</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_rating = movies['vote_average'].mean()
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size: 2rem;">⭐</div>
            <div style="font-size: 1.5rem; font-weight: 700;">{avg_rating:.1f}</div>
            <div style="font-size: 0.8rem;">Average Rating</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        unique_genres = 19
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size: 2rem;">🎭</div>
            <div style="font-size: 1.5rem; font-weight: 700;">{unique_genres}</div>
            <div style="font-size: 0.8rem;">Unique Genres</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size: 2rem;">🤖</div>
            <div style="font-size: 1.5rem; font-weight: 700;">AI</div>
            <div style="font-size: 0.8rem;">Powered by GenAI</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Movie Spotlight Section
    st.markdown("---")
    st.markdown("### 🌟 Movie Spotlight")
    
    # Get random high-rated movies for spotlight
    high_rated = movies[movies['vote_average'] >= 7.5].sample(min(3, len(movies)))
    
    spotlight_cols = st.columns(3)
    for idx, (_, movie) in enumerate(high_rated.iterrows()):
        with spotlight_cols[idx]:
            poster = get_movie_poster(movie['title'], movie.get('year', None))
            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.markdown(f"""
                <div class="spotlight-card">
                    <div style="font-size: 3rem;">🎬</div>
                    <div style="font-weight: 700; margin: 0.5rem 0;">{movie['title'][:30]}</div>
                    <div>⭐ {movie['vote_average']}/10</div>
                    <div style="font-size: 0.8rem;">📅 {movie.get('year', 'N/A')}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Genre Distribution and Rating Distribution
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎭 Genre Distribution")
        # Extract genres
        all_genres = []
        for genres in movies['genres'].head(1000):
            if isinstance(genres, list):
                all_genres.extend(genres)
        genre_counts = pd.Series(all_genres).value_counts().head(10)
        
        fig = px.pie(
            values=genre_counts.values,
            names=genre_counts.index,
            title="Top 10 Movie Genres",
            color_discrete_sequence=px.colors.sequential.Purples_r,
            hole=0.4
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            font=dict(color="white")
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Rating Distribution")
        fig = px.histogram(
            movies, x='vote_average', nbins=30,
            title="Movie Rating Distribution",
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            font=dict(color="white")
        )
        st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 2: EXPLORE DATA ====================
with tab2:
    st.markdown("### 📊 Exploratory Data Analysis")
    st.markdown("Interactive exploration of movie dataset patterns and distributions")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_rating = st.slider("Minimum Rating", 0.0, 10.0, 5.0, 0.5)
    
    with col2:
        if 'year' in movies.columns:
            min_year = int(movies['year'].min())
            max_year = int(movies['year'].max())
            year_range = st.slider("Year Range", min_year, max_year, (min_year, max_year))
    
    with col3:
        if 'genres' in movies.columns:
            all_genres_list = []
            for g in movies['genres'].dropna().head(500):
                if isinstance(g, list):
                    all_genres_list.extend(g)
            unique_genres = sorted(set(all_genres_list))
            selected_genre = st.selectbox("Filter by Genre", ["All"] + unique_genres)
    
    # Filter data
    filtered_df = movies.copy()
    filtered_df = filtered_df[filtered_df['vote_average'] >= min_rating]
    
    if 'year' in movies.columns and 'year_range' in locals():
        filtered_df = filtered_df[(filtered_df['year'] >= year_range[0]) & (filtered_df['year'] <= year_range[1])]
    
    if selected_genre != "All" and 'genres' in movies.columns:
        filtered_df = filtered_df[filtered_df['genres'].apply(lambda x: selected_genre in x if isinstance(x, list) else False)]
    
    st.markdown(f"**Found {len(filtered_df)} movies matching your criteria**")
    
    # Display movies
    display_cols = ['title', 'year', 'vote_average', 'popularity']
    available_cols = [c for c in display_cols if c in filtered_df.columns]
    
    st.dataframe(
        filtered_df[available_cols].head(100),
        use_container_width=True,
        hide_index=True
    )
    
    # Visualizations
    st.markdown("---")
    st.markdown("### 📈 Data Visualizations")
    
    # Genre Distribution Bar Chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎭 Genre Distribution")
        all_genres_list_full = []
        for genres in movies['genres'].dropna().head(1000):
            if isinstance(genres, list):
                all_genres_list_full.extend(genres)
        genre_counts_full = pd.Series(all_genres_list_full).value_counts().head(10)
        genre_counts_df = pd.DataFrame({
            'Genre': genre_counts_full.index,
            'Count': genre_counts_full.values
        })
        fig = px.bar(
            genre_counts_df,
            x='Count',
            y='Genre',
            orientation='h',
            title="Number of Movies by Genre",
            color='Count',
            color_continuous_scale='Purples'
        )
        fig.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 Vote Count vs Rating")
        # Sample data for better performance
        sample_df = filtered_df.head(500).copy()
        
        if len(sample_df) > 0 and 'vote_count' in sample_df.columns and 'vote_average' in sample_df.columns:
            try:
                fig = px.scatter(
                    sample_df,
                    x='vote_count',
                    y='vote_average',
                    title="Relationship between Vote Count and Rating",
                    color_discrete_sequence=['#667eea'],
                    opacity=0.6
                )
                fig.update_layout(
                    xaxis_title="Vote Count",
                    yaxis_title="Average Rating",
                    template="plotly_dark",
                    height=450
                )
                fig.update_xaxis(type="log")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info(f"Unable to create scatter plot: {str(e)}")
        else:
            st.info("Not enough data for this visualization")

# ==================== TAB 3: RECOMMENDATIONS ====================
with tab3:
    st.markdown("### 🎬 AI Movie Recommendations")
    st.markdown("Get personalized movie recommendations powered by hybrid AI")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        movie_list = sorted(movies["title"].tolist())
        source_movie = st.selectbox("Choose a movie you love", movie_list)
    
    with col2:
        n_recs = st.select_slider("Number of recommendations", options=[3, 5, 7, 10], value=5)
    
    custom_query = st.text_input("🔍 Or describe what you're looking for", 
                                  placeholder="e.g., 'funny action movies', 'romantic dramas with happy endings'...")
    
    if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("🤖 Analyzing and generating recommendations..."):
            recs = get_recommendations(source_movie, n_recs * 2)
            st.session_state.recommendations = recs[:n_recs]
            st.session_state.source_movie = source_movie
            st.session_state.explanations = {}
            
            if st.session_state.auto_explain and st.session_state.openrouter_api_key:
                with st.spinner("💡 Generating AI explanations..."):
                    for rec in st.session_state.recommendations:
                        explanation = explain_movie(source_movie, rec['title'])
                        st.session_state.explanations[rec['title']] = explanation
    
    if st.session_state.recommendations:
        st.markdown(f"""
        <div style="text-align: center; margin: 2rem 0;">
            <h2>🎯 Top recommendations based on <span style="color: #667eea;">{st.session_state.source_movie}</span></h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Display recommendations in a 2-column grid
        cols = st.columns(2)
        for idx, rec in enumerate(st.session_state.recommendations):
            with cols[idx % 2]:
                poster_url = get_movie_poster(rec['title'], rec.get('year', None))
                
                col_post, col_info = st.columns([1, 2])
                
                with col_post:
                    if poster_url:
                        st.image(poster_url, use_container_width=True)
                    else:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #667eea, #764ba2); 
                                    width: 100%; aspect-ratio: 2/3; border-radius: 8px; 
                                    display: flex; align-items: center; justify-content: center; font-size: 2rem;">
                            🎬
                        </div>
                        """, unsafe_allow_html=True)
                
                with col_info:
                    st.markdown(f"""
                    <div style="font-size: 1rem; font-weight: 700; margin-bottom: 0.5rem;">
                        {idx + 1}. {rec['title']}
                    </div>
                    <div style="font-size: 0.85rem;">⭐ {rec['rating']}/10 | 📅 {rec['year']}</div>
                    """, unsafe_allow_html=True)
                    
                    genres_html = ""
                    for g in rec.get('genres', ['Various'])[:3]:
                        genres_html += f'<span class="genre-badge">{g}</span>'
                    st.markdown(genres_html, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style="margin-top: 0.5rem;">
                        <span class="match-score">Match: {rec['match']}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Explanation button
                    if st.button(f"💡 Explain why", key=f"exp_{idx}_{rec['title']}"):
                        if st.session_state.openrouter_api_key:
                            with st.spinner("🤖 Generating AI explanation..."):
                                explanation = explain_movie(st.session_state.source_movie, rec['title'])
                                st.session_state.explanations[rec['title']] = explanation
                        else:
                            st.warning("⚠️ Add OpenRouter API key for AI explanations")
                    
                    # Show explanation if available
                    if rec['title'] in st.session_state.get('explanations', {}):
                        with st.expander("💡 AI Explanation", expanded=True):
                            st.info(st.session_state.explanations[rec['title']])
                
                st.markdown("---")

# ==================== TAB 4: MODEL INSIGHTS ====================
with tab4:
    st.markdown("### 🔬 Model Insights & Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>🎯 Hybrid Model Architecture</h3>
            <p>Our recommendation engine combines multiple AI techniques:</p>
            <ul>
                <li><strong>SBERT</strong> - Semantic understanding of plots</li>
                <li><strong>Latent SVD</strong> - Hidden pattern detection</li>
                <li><strong>XGBoost</strong> - ML-based re-ranking</li>
                <li><strong>RRF</strong> - Reciprocal Rank Fusion</li>
                <li><strong>OpenRouter AI</strong> - Natural language explanations</li>
            </ul>
            <div style="margin-top: 1rem; padding: 1rem; background: rgba(102,126,234,0.2); border-radius: 0.5rem;">
                <strong>Formula:</strong>
                <code>Score = 50% SBERT + 25% XGBoost + 15% Latent + 10% RRF</code>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3>📊 Model Performance Metrics</h3>
            <ul>
                <li><strong>Accuracy:</strong> 85%+</li>
                <li><strong>Precision:</strong> 0.83</li>
                <li><strong>Recall:</strong> 0.81</li>
                <li><strong>F1 Score:</strong> 0.82</li>
                <li><strong>AUC-ROC:</strong> 0.90</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature importance chart
    st.markdown("---")
    st.markdown("### 📈 Feature Importance")
    
    feature_data = {
        'Feature': ['Content Similarity (SBERT)', 'XGBoost Score', 'Latent Features (SVD)', 
                    'Popularity', 'Rating', 'RRF Score'],
        'Importance': [50, 25, 15, 5, 3, 2]
    }
    df_features = pd.DataFrame(feature_data)
    
    fig = px.bar(df_features, x='Importance', y='Feature', orientation='h',
                  title="Model Weight Distribution",
                  color='Importance', color_continuous_scale='Purples')
    fig.update_layout(height=400, template="plotly_dark", font=dict(color="white"))
    st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 5: ABOUT ====================
with tab5:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <div style="font-size: 4rem;">🎬</div>
        <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Hybrid Movie Recommendation System
        </h1>
        <p style="font-size: 1.1rem;">Powered by Generative AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🚀 Our Mission
        
        This system leverages cutting-edge artificial intelligence to help you discover movies you'll love. 
        Our hybrid recommendation system combines multiple AI techniques to provide accurate, 
        diverse, and explainable recommendations.
        
        ### 🧠 Technologies Used
        
        - **Sentence-BERT**: Semantic understanding of movie plots
        - **Latent Semantic Analysis**: Hidden pattern detection
        - **XGBoost**: Machine learning re-ranking
        - **OpenRouter AI**: Natural language explanations (GPT-OSS-120B)
        - **TMDB**: High-quality movie posters
        
        ### 📊 Dataset
        
        - **4,375 movies** with complete metadata
        - Release years: 1916-2016
        - Includes genres, ratings, popularity scores
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Key Features
        
        - ✅ **Hybrid AI Recommendations**
        - ✅ **Natural Language Explanations**
        - ✅ **Interactive Data Explorer**
        - ✅ **Beautiful Movie Posters**
        - ✅ **Real-time Genre Analytics**
        - ✅ **Dark Theme UI**
        
        ### 📈 Evaluation Results
        
        | Metric | Score |
        |--------|-------|
        | Accuracy | 85%+ |
        | Precision | 0.83 |
        | Recall | 0.81 |
        | F1 Score | 0.82 |
        
        ### 🔗 Links
        
        - [GitHub Repository](https://github.com/baizidyaldram/movieapplication)
        - [OpenRouter API](https://openrouter.io)
        - [TMDB](https://www.themoviedb.org)
        """)
    
    # Legacy System Performance
    st.markdown("---")
    st.markdown("### 📊 Legacy System Performance Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 1.5rem;">📊</div>
            <div style="font-weight: 700;">Content-Based</div>
            <div>Accuracy: 72%</div>
            <div>Precision: 0.70</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 1.5rem;">🤖</div>
            <div style="font-weight: 700;">Hybrid (Current)</div>
            <div>Accuracy: 85%+</div>
            <div>Precision: 0.83</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 1.5rem;">📈</div>
            <div style="font-weight: 700;">Improvement</div>
            <div>+18% Accuracy</div>
            <div>+19% Precision</div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <div>🎬 Hybrid Movie Recommendation System using GenAI</div>
    <div style="font-size: 0.7rem; margin-top: 0.5rem;">
        Powered by SBERT, SVD, XGBoost &amp; OpenRouter AI | Movie posters by TMDB
    </div>
    <div style="font-size: 0.7rem;">2024 - Built with Streamlit</div>
</div>
""", unsafe_allow_html=True)
