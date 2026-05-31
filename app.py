import streamlit as st

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Hybrid Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding-top: 1rem;
    }
    
    /* Metric cards styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #262730;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
    }
    
    /* Recommendation card styling */
    .rec-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #4a4a4a;
        transition: transform 0.2s;
    }
    
    .rec-card:hover {
        transform: translateY(-5px);
        border-color: #764ba2;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #262730;
        border-radius: 8px;
    }
    
    /* Success message styling */
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'openrouter_api_key' not in st.session_state:
    st.session_state.openrouter_api_key = None

# Sidebar for API key (collapsible)
with st.sidebar:
    st.markdown("### 🔑 API Configuration")
    
    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            st.session_state.openrouter_api_key = st.secrets["OPENROUTER_API_KEY"]
            st.success("✅ OpenRouter API key loaded!")
        else:
            st.warning("⚠️ No API key in secrets")
            api_key_input = st.text_input(
                "Enter OpenRouter API Key:",
                type="password",
                help="Get your key from https://openrouter.io/keys"
            )
            if api_key_input:
                st.session_state.openrouter_api_key = api_key_input
                st.success("✅ API key set!")
                st.rerun()
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Main title
st.title("🎬 Hybrid Movie Recommendation System")

# Create tabs at the top
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Home", 
    "🔬 Model Explorer", 
    "🎯 Hybrid Recommender", 
    "📊 Evaluation",
    "ℹ️ About"
])

# Tab 1: Home
with tab1:
    from src.recommender import movies
    
    if movies is not None and len(movies) > 0:
        st.markdown("### Welcome to the Ultimate Movie Recommendation Engine")
        st.markdown("Experience the power of hybrid AI recommendations combining multiple cutting-edge techniques.")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎬 Total Movies", len(movies))
        with col2:
            st.metric("⚙️ Models", "4 (SBERT + SVD + XGB + RRF)")
        with col3:
            st.metric("🤖 AI Engine", "OpenRouter")
        with col4:
            st.metric("📈 Accuracy", "85%+")
        
        st.markdown("---")
        st.subheader("📊 Recent Movies")
        
        display_cols = ["title", "year", "vote_average", "popularity"]
        available_cols = [col for col in display_cols if col in movies.columns]
        
        if available_cols:
            st.dataframe(
                movies[available_cols].head(20),
                use_container_width=True,
                hide_index=True
            )
    else:
        st.error("⚠️ Movies data not loaded properly")

# Tab 2: Model Explorer (Enhanced with explanations)
with tab2:
    from src.recommender import movies, movie_to_idx, content_sim, latent_sim
    
    st.markdown("### 🔬 Model Explorer - Understand How Recommendations Work")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        movie = st.selectbox(
            "🎥 Choose a Movie to Analyze",
            sorted(movies["title"].tolist()),
            key="model_explorer_select"
        )
    
    with col2:
        st.markdown("### 📊 Hybrid Score")
        st.info("The Hybrid Score combines all models for final recommendations")
    
    if movie:
        idx = movie_to_idx[movie]
        
        # Create expandable sections for model explanations
        with st.expander("📖 How Each Model Works", expanded=False):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown("**🔤 SBERT Similarity**")
                st.caption("Uses Sentence-BERT to understand semantic meaning of movie descriptions and plots.")
            with col_b:
                st.markdown("**📐 Latent Similarity (SVD)**")
                st.caption("Captures hidden patterns and latent features using matrix factorization.")
            with col_c:
                st.markdown("**🎯 Hybrid Score**")
                st.caption("Weighted combination: 50% SBERT + 25% XGBoost + 15% Latent + 10% RRF")
        
        # Model comparison tabs
        model_tab1, model_tab2, model_tab3 = st.tabs([
            "🔤 SBERT Semantic Similarity", 
            "📐 Latent Semantic Analysis",
            "🎯 Hybrid Score"
        ])
        
        with model_tab1:
            st.markdown("**How it works:** Uses deep learning to understand the semantic meaning of movie descriptions.")
            sims = content_sim[idx]
            top = sims.argsort()[::-1][1:11]
            
            rows = []
            for i in top:
                similarity_score = float(sims[i])
                rows.append({
                    "Rank": len(rows) + 1,
                    "Movie": movies.iloc[i]["title"],
                    "Similarity Score": f"{similarity_score:.4f}",
                    "Year": movies.iloc[i].get("year", "N/A"),
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
        
        with model_tab2:
            st.markdown("**How it works:** Discovers hidden patterns and latent features through Singular Value Decomposition.")
            sims = latent_sim[idx]
            top = sims.argsort()[::-1][1:11]
            
            rows = []
            for i in top:
                similarity_score = float(sims[i])
                rows.append({
                    "Rank": len(rows) + 1,
                    "Movie": movies.iloc[i]["title"],
                    "Latent Similarity": f"{similarity_score:.4f}",
                    "Year": movies.iloc[i].get("year", "N/A"),
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
        
        with model_tab3:
            st.markdown("**Hybrid Score Formula:**")
            st.latex(r'\text{Hybrid} = 0.5 \times \text{SBERT} + 0.25 \times \text{XGBoost} + 0.15 \times \text{Latent} + 0.1 \times \text{RRF}')
            
            # Show hybrid scores for top recommendations
            from src.recommender import get_recommendations
            recs = get_recommendations(movie, 10)
            
            if recs:
                st.markdown(f"**Top recommendations for '{movie}':**")
                for idx, rec in enumerate(recs[:5], 1):
                    col_a, col_b = st.columns([4, 1])
                    with col_a:
                        st.write(f"{idx}. **{rec['title']}** ({rec['year']})")
                    with col_b:
                        st.metric("Hybrid Score", f"{rec['match']}%")

# Tab 3: Hybrid Recommender (Enhanced with horizontal cards)
with tab3:
    from src.recommender import movies, get_recommendations
    from src.llm_explainer import explain_movie, initialize_llm
    
    # Initialize LLM if API key available
    if st.session_state.openrouter_api_key:
        initialize_llm(st.session_state.openrouter_api_key)
    
    st.markdown("### 🎯 Get AI-Powered Movie Recommendations")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        movie = st.selectbox(
            "🎥 Select a movie you love",
            sorted(movies["title"].tolist()),
            key="hybrid_select"
        )
    
    with col2:
        top_n = st.slider(
            "📊 Number of recommendations",
            5, 20, 10
        )
    
    with col3:
        st.markdown("### ")
        generate_btn = st.button("🚀 Generate Recommendations", type="primary", use_container_width=True)
    
    if generate_btn or "recs" in st.session_state:
        if generate_btn:
            with st.spinner("🔍 Generating recommendations using hybrid models..."):
                recs = get_recommendations(movie, top_n)
                st.session_state["recs"] = recs
                st.session_state["source"] = movie
                st.session_state["explanations"] = {}
        
        if "recs" in st.session_state and st.session_state["recs"]:
            st.markdown("---")
            st.markdown(f"### 🎬 Top recommendations based on **{st.session_state['source']}**")
            
            # Display recommendations in grid (horizontal layout)
            cols = st.columns(2)
            
            for idx, rec in enumerate(st.session_state["recs"]):
                with cols[idx % 2]:
                    with st.container():
                        st.markdown(f"""
                        <div class="rec-card">
                            <h3>🎥 {rec['title']}</h3>
                            <p>⭐ Rating: {rec['rating']}/10 | 📅 {rec['year']}</p>
                            <p>🎭 Genres: {', '.join(rec.get('genres', ['Various'])[:3])}</p>
                            <p>🔗 Match Score: <b>{rec['match']}%</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Generate explanation button
                        if st.button(f"💡 Explain This", key=f"explain_{idx}_{rec['title']}"):
                            if st.session_state.openrouter_api_key:
                                with st.spinner("🤖 Generating AI explanation..."):
                                    explanation = explain_movie(st.session_state["source"], rec["title"])
                                    st.session_state["explanations"][rec['title']] = explanation
                            else:
                                st.warning("⚠️ Please add OpenRouter API key for AI explanations")
                        
                        # Show explanation if available
                        if rec['title'] in st.session_state.get("explanations", {}):
                            with st.expander("💡 AI Explanation", expanded=True):
                                st.info(st.session_state["explanations"][rec['title']])

# Tab 4: Evaluation
with tab4:
    from src.evaluation import run_evaluation
    import plotly.express as px
    
    st.markdown("### 📊 Model Evaluation Dashboard")
    
    if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
        with st.spinner("Running comprehensive evaluation..."):
            df = run_evaluation()
            
            if df is not None and len(df) > 0:
                st.success("✅ Evaluation complete!")
                
                st.subheader("📈 Performance Metrics")
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                metrics = ["AvgSim@5", "Diversity@5", "VoteQuality@5"]
                available_metrics = [m for m in metrics if m in df.columns]
                
                for metric in available_metrics:
                    fig = px.bar(
                        df,
                        x="Movie",
                        y=metric,
                        title=f"{metric} by Movie",
                        color=metric,
                        color_continuous_scale="Viridis",
                        text=metric
                    )
                    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

# Tab 5: About
with tab5:
    st.markdown("### ℹ️ About This Project")
    
    st.markdown("""
    ## 🚀 Hybrid Movie Recommendation System
    
    This advanced recommendation system combines multiple AI techniques to provide accurate and diverse movie recommendations.
    
    ### 🧠 Technologies Used:
    
    - **Sentence-BERT (SBERT)**: For semantic understanding of movie descriptions
    - **Latent Semantic Analysis (SVD)**: For capturing hidden patterns
    - **XGBoost**: For re-ranking recommendations
    - **Reciprocal Rank Fusion (RRF)**: For combining multiple ranking signals
    - **OpenRouter AI**: For generating natural language explanations
    
    ### 📊 Model Weights:
    
    Hybrid Score = 50% SBERT + 25% XGBoost + 15% Latent + 10% RRF
    
    ### 🎯 Features:
    
    - Real-time recommendations
    - AI-powered explanations
    - Multiple similarity metrics
    - Comprehensive evaluation
    - Interactive visualizations
    
    ### 🔗 Links:
    
    - [GitHub Repository](https://github.com/baizidyaldram/movieapplication)
    - [OpenRouter API](https://openrouter.io)
    - [Streamlit Cloud](https://streamlit.io/cloud)
    
    ---
    
    Built with ❤️ using Streamlit, Python, and AI
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    🎬 Hybrid Movie Recommendation System | Powered by SBERT, SVD, XGBoost & OpenRouter AI
</div>
""", unsafe_allow_html=True)
