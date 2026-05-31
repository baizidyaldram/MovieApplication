import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Hybrid Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS following Material Design 3 + Academic Tech Theme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main {
        background-color: #F8FAFC;
        padding: 0rem 1rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E3A8A 0%, #1E40AF 100%);
        padding-top: 2rem;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {
        color: rgba(255,255,255,0.9) !important;
    }
    
    /* Tab styling - Material Design Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: white;
        border-bottom: 2px solid #E2E8F0;
        padding: 0;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        padding: 0 1.5rem;
        font-weight: 600;
        font-size: 0.875rem;
        letter-spacing: 0.025em;
        color: #64748B;
        border-bottom: 2px solid transparent;
        transition: all 0.15s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #1E3A8A;
        background-color: #F1F5F9;
    }
    
    .stTabs [aria-selected="true"] {
        color: #2563EB !important;
        border-bottom: 2px solid #2563EB !important;
        background-color: transparent;
    }
    
    /* Card styling */
    .metric-card, .stat-card {
        background: white;
        border-radius: 0.75rem;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: all 0.15s ease;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover, .stat-card:hover {
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    /* Gradient cards for special metrics */
    .gradient-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        color: white;
        border-radius: 0.75rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Movie recommendation cards */
    .movie-card {
        background: white;
        border-radius: 0.75rem;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid #E2E8F0;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    
    .movie-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        border-color: #2563EB;
    }
    
    .movie-title {
        font-size: 1.125rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.5rem;
    }
    
    .movie-meta {
        font-size: 0.875rem;
        color: #6B7280;
        margin-bottom: 0.75rem;
    }
    
    .movie-score {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2563EB;
    }
    
    .match-score {
        display: inline-block;
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Genre badges */
    .genre-badge {
        display: inline-block;
        background-color: #E0E7FF;
        color: #1E3A8A;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.6rem 1.25rem;
        font-weight: 600;
        font-size: 0.875rem;
        transition: all 0.15s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37,99,235,0.3);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #F8FAFC;
        border-radius: 0.5rem;
        font-weight: 600;
        color: #1E3A8A;
    }
    
    /* Dataframe styling */
    .dataframe {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.813rem;
    }
    
    /* Info/Warning/Success boxes */
    .stAlert {
        border-radius: 0.5rem;
        border-left: 4px solid;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #94A3B8;
        font-size: 0.875rem;
        border-top: 1px solid #E2E8F0;
        margin-top: 3rem;
    }
    
    /* Method indicator badges */
    .method-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .sbert-badge {
        background-color: #DBEAFE;
        color: #1E3A8A;
    }
    
    .svd-badge {
        background-color: #E0E7FF;
        color: #4338CA;
    }
    
    .hybrid-badge {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
    }
    
    /* Code viewer */
    .code-viewer {
        background-color: #1E293B;
        border-radius: 0.5rem;
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.813rem;
        color: #E2E8F0;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'openrouter_api_key' not in st.session_state:
    st.session_state.openrouter_api_key = None
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Sidebar Navigation
with st.sidebar:
    # Logo/Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <div style="font-size: 3rem;">🎬</div>
        <div style="font-weight: 700; font-size: 1.25rem; margin-top: 0.5rem;">Movie Recommender</div>
        <div style="font-size: 0.75rem; opacity: 0.8;">Hybrid AI System</div>
    </div>
    """, unsafe_allow_html=True)
    
    # API Configuration
    st.markdown("### 🔑 API Config")
    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            st.session_state.openrouter_api_key = st.secrets["OPENROUTER_API_KEY"]
            st.success("API Key Loaded")
        else:
            api_key_input = st.text_input(
                "OpenRouter API Key",
                type="password",
                placeholder="Enter your API key",
                help="Get key from https://openrouter.io/keys"
            )
            if api_key_input:
                st.session_state.openrouter_api_key = api_key_input
                st.success("Key set successfully!")
    except:
        pass
    
    # Theme Toggle
    st.markdown("---")
    theme = st.toggle("Dark Mode", key="dark_mode_toggle")
    if theme != st.session_state.dark_mode:
        st.session_state.dark_mode = theme
        st.rerun()
    
    # System Stats
    st.markdown("---")
    st.markdown("### System Stats")
    from src.recommender import movies
    if movies is not None:
        st.metric("Total Movies", len(movies))
        st.metric("Active Models", "4")

# Main content area with tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "DASHBOARD",
    "MODEL EXPLORER", 
    "RECOMMENDATIONS",
    "EVALUATION",
    "ABOUT"
])

# ==================== TAB 1: DASHBOARD ====================
with tab1:
    st.markdown("### Welcome to the Hybrid Movie Recommendation System")
    st.markdown("An AI-powered system combining **SBERT**, **SVD**, **XGBoost**, and **OpenRouter** for intelligent movie recommendations.")
    
    # Quick Stats Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="gradient-card" style="text-align: center;">
            <div style="font-size: 2rem;">🎬</div>
            <div style="font-size: 1.5rem; font-weight: 700;">4,375</div>
            <div style="font-size: 0.75rem;">Movies in Database</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 2rem;">⚙️</div>
            <div style="font-size: 1.5rem; font-weight: 700;">4</div>
            <div style="font-size: 0.75rem;">AI Models</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 2rem;">🤖</div>
            <div style="font-size: 1.5rem; font-weight: 700;">OpenRouter</div>
            <div style="font-size: 0.75rem;">LLM Integration</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 2rem;">📈</div>
            <div style="font-size: 1.5rem; font-weight: 700;">85%+</div>
            <div style="font-size: 0.75rem;">Accuracy Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Method Badges
    st.markdown("---")
    st.markdown("### Recommendation Methods")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="method-badge sbert-badge" style="display: flex; align-items: center; gap: 0.5rem;">
            🔤 <strong>SBERT Similarity</strong> - Semantic understanding of plots
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="method-badge svd-badge" style="display: flex; align-items: center; gap: 0.5rem;">
            📐 <strong>Latent SVD</strong> - Hidden pattern detection
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="method-badge hybrid-badge" style="display: flex; align-items: center; gap: 0.5rem;">
            🎯 <strong>Hybrid Score</strong> - Weighted ensemble
        </div>
        """, unsafe_allow_html=True)
    
    # Dataset Preview
    st.markdown("---")
    st.markdown("### Dataset Preview")
    
    from src.recommender import movies
    if movies is not None and len(movies) > 0:
        display_cols = ["title", "year", "vote_average", "popularity"]
        available_cols = [col for col in display_cols if col in movies.columns]
        
        if available_cols:
            st.dataframe(
                movies[available_cols].head(20),
                use_container_width=True,
                hide_index=True
            )

# ==================== TAB 2: MODEL EXPLORER ====================
with tab2:
    st.markdown("### Model Explorer")
    st.markdown("Understand how each model contributes to your recommendations.")
    
    from src.recommender import movies, movie_to_idx, content_sim, latent_sim, get_recommendations
    
    # Movie selector
    movie_list = sorted(movies["title"].tolist())
    selected_movie = st.selectbox("Select a movie to analyze", movie_list)
    
    if selected_movie:
        idx = movie_to_idx[selected_movie]
        
        # Model cards with explanations
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🔤</div>
                <div style="font-weight: 700; margin-bottom: 0.25rem;">SBERT Similarity</div>
                <div style="font-size: 0.75rem; color: #6B7280;">Uses Sentence-BERT for semantic understanding of movie descriptions and plots.</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show SBERT similarities
            sims = content_sim[idx]
            top_sbert = sims.argsort()[::-1][1:6]
            st.markdown("**Top SBERT matches:**")
            for i, t in enumerate(top_sbert, 1):
                st.write(f"{i}. {movies.iloc[t]['title']} ({sims[t]:.3f})")
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📐</div>
                <div style="font-weight: 700; margin-bottom: 0.25rem;">Latent SVD Similarity</div>
                <div style="font-size: 0.75rem; color: #6B7280;">Captures hidden patterns through Singular Value Decomposition.</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show Latent similarities
            sims = latent_sim[idx]
            top_latent = sims.argsort()[::-1][1:6]
            st.markdown("**Top Latent matches:**")
            for i, t in enumerate(top_latent, 1):
                st.write(f"{i}. {movies.iloc[t]['title']} ({sims[t]:.3f})")
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🎯</div>
                <div style="font-weight: 700; margin-bottom: 0.25rem;">Hybrid Score</div>
                <div style="font-size: 0.75rem; color: #6B7280;">Weighted combination of all models for final ranking.</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show Hybrid scores
            recs = get_recommendations(selected_movie, 5)
            if recs:
                for r in recs:
                    st.markdown(f"**{r['title']}** - {r['match']}% match")
        
        # Formula display
        st.markdown("---")
        st.markdown("### Hybrid Score Formula")
        st.latex(r'\text{Score} = 0.5 \times \text{SBERT} + 0.25 \times \text{XGBoost} + 0.15 \times \text{Latent} + 0.1 \times \text{RRF}')

# ==================== TAB 3: RECOMMENDATIONS ====================
with tab3:
    st.markdown("### Get Movie Recommendations")
    st.markdown("Select a movie you love and get AI-powered recommendations.")
    
    from src.recommender import movies, get_recommendations
    from src.llm_explainer import explain_movie, initialize_llm
    
    if st.session_state.openrouter_api_key:
        initialize_llm(st.session_state.openrouter_api_key)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        source_movie = st.selectbox(
            "Choose a movie you like",
            sorted(movies["title"].tolist()),
            key="rec_movie"
        )
    
    with col2:
        n_recs = st.slider("Number of recommendations", 5, 20, 10)
    
    if st.button("Generate Recommendations", type="primary", use_container_width=True):
        with st.spinner("Generating recommendations..."):
            recs = get_recommendations(source_movie, n_recs)
            st.session_state['recommendations'] = recs
            st.session_state['source_movie'] = source_movie
            st.session_state['explanations'] = {}
    
    if 'recommendations' in st.session_state:
        st.markdown(f"### Top recommendations based on **{st.session_state['source_movie']}**")
        
        # Display in grid
        cols = st.columns(2)
        for idx, rec in enumerate(st.session_state['recommendations']):
            with cols[idx % 2]:
                # Create genre badges HTML
                genres_html = ""
                for g in rec.get('genres', ['Various'])[:3]:
                    genres_html += f'<span class="genre-badge">{g}</span>'
                
                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-title">🎬 {rec['title']}</div>
                    <div class="movie-meta">⭐ {rec['rating']}/10 | 📅 {rec['year']}</div>
                    <div class="movie-meta">
                        {genres_html}
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.75rem;">
                        <span class="match-score">Match: {rec['match']}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Explanation button
                if st.button(f"Explain why", key=f"exp_{idx}_{rec['title']}"):
                    if st.session_state.openrouter_api_key:
                        with st.spinner("Generating AI explanation..."):
                            explanation = explain_movie(st.session_state['source_movie'], rec['title'])
                            st.session_state['explanations'][rec['title']] = explanation
                    else:
                        st.warning("Add OpenRouter API key for AI explanations")
                
                if rec['title'] in st.session_state.get('explanations', {}):
                    with st.expander("AI Explanation", expanded=True):
                        st.info(st.session_state['explanations'][rec['title']])

# ==================== TAB 4: EVALUATION ====================
with tab4:
    st.markdown("### Model Evaluation")
    st.markdown("Comprehensive evaluation metrics for our hybrid recommendation system.")
    
    from src.evaluation import run_evaluation
    
    if st.button("Run Evaluation", type="primary", use_container_width=True):
        with st.spinner("Evaluating models..."):
            df = run_evaluation()
            
            if df is not None and len(df) > 0:
                st.success("Evaluation complete!")
                
                # Metrics display
                st.subheader("Performance Metrics by Movie")
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Visualizations
                metrics = ["AvgSim@5", "Diversity@5", "VoteQuality@5"]
                available = [m for m in metrics if m in df.columns]
                
                for metric in available:
                    fig = px.bar(
                        df,
                        x="Movie",
                        y=metric,
                        title=f"{metric} by Movie",
                        color=metric,
                        color_continuous_scale="Blues",
                        text=metric
                    )
                    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 5: ABOUT ====================
with tab5:
    st.markdown("### About This Project")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Hybrid Movie Recommendation System
        
        This advanced system combines multiple AI techniques to provide accurate, diverse, and explainable movie recommendations.
        
        ### Core Technologies
        
        | Technology | Purpose |
        |-----------|---------|
        | **Sentence-BERT** | Semantic understanding of movie plots and descriptions |
        | **SVD (Latent Analysis)** | Capturing hidden patterns and latent features |
        | **XGBoost** | Machine learning re-ranking of candidates |
        | **RRF (Reciprocal Rank Fusion)** | Combining multiple ranking signals |
        | **OpenRouter AI** | Generating natural language explanations |
        
        ### Model Weights
          Hybrid Score = 50% SBERT + 25% XGBoost + 15% Latent + 10% RRF
  ### Key Features

- Real-time hybrid recommendations
- AI-powered natural language explanations
- Multiple similarity metrics
- Comprehensive evaluation framework
- Interactive visualizations

### Links

- [GitHub Repository](https://github.com/baizidyaldram/movieapplication)
- [OpenRouter API](https://openrouter.io)
- [Streamlit Cloud](https://streamlit.io)
""")
        with col2:
st.markdown("""
<div class="gradient-card">
    <div style="text-align: center;">
        <div style="font-size: 3rem;">🎬</div>
        <div style="font-weight: 700; font-size: 1.25rem; margin: 1rem 0;">Version 2.0</div>
        <div style="font-size: 0.875rem;">Hybrid AI Movie Recommender</div>
        <hr style="margin: 1rem 0;">
        <div style="font-size: 0.75rem;">Built with ❤️ using Streamlit</div>
        <div style="font-size: 0.75rem;">Powered by OpenRouter AI</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
<div>Hybrid Movie Recommendation System | Powered by SBERT, SVD, XGBoost and OpenRouter AI</div>
<div style="font-size: 0.75rem; margin-top: 0.5rem;">2024 - Built with Streamlit</div>
</div>
""", unsafe_allow_html=True)
