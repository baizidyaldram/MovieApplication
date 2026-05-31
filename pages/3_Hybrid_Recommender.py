import streamlit as st

from src.recommender import (
    movies,
    get_recommendations
)

from src.llm_explainer import (
    explain_movie,
    initialize_llm
)

# Initialize LLM with API key from session state
if 'openrouter_api_key' in st.session_state and st.session_state.openrouter_api_key:
    initialize_llm(st.session_state.openrouter_api_key)

st.title("🎬 Hybrid Movie Recommendation")

# Check if API key is available
api_available = st.session_state.get('openrouter_api_key') is not None

if not api_available:
    st.warning("⚠️ OpenRouter API key not configured. AI explanations will be disabled. Please add your API key in the sidebar.")
    st.info("You can still get recommendations, but without AI explanations.")

movie = st.selectbox(
    "Select a Movie You Like",
    sorted(movies["title"].tolist())
)

top_n = st.slider(
    "Number of Recommendations",
    5,
    20,
    10
)

if st.button(
    "🎯 Generate Recommendations",
    use_container_width=True,
    type="primary"
):

    with st.spinner("Generating recommendations using hybrid models..."):
        recs = get_recommendations(movie, top_n)
        st.session_state["recs"] = recs
        st.session_state["source"] = movie
        st.success(f"✅ Found {len(recs)} recommendations!")

if "recs" in st.session_state:

    st.markdown("---")
    st.subheader(f"📽️ Top Recommendations based on '{st.session_state['source']}'")
    
    for idx, rec in enumerate(st.session_state["recs"], 1):
        
        with st.container(border=True):
            
            col1, col2, col3 = st.columns([3,1,1])
            
            with col1:
                st.markdown(f"**{idx}. {rec['title']}**")
                st.write(f"⭐ Rating: {rec['rating']}")
                st.write(f"📅 Year: {rec['year']}")
                if rec.get('genres'):
                    st.write(f"🎭 Genres: {', '.join(rec['genres'][:3])}")
            
            with col2:
                st.metric("Match Score", f"{rec['match']}%")
            
            with col3:
                # Only show explain button if API is available
                if api_available:
                    if st.button(f"🤖 Explain", key=f"explain_{idx}_{rec['title']}"):
                        with st.spinner(f"Generating AI explanation for {rec['title']}..."):
                            explanation = explain_movie(
                                st.session_state["source"],
                                rec["title"]
                            )
                            st.session_state[f"explanation_{idx}"] = explanation
                
                # Display explanation if available
                if f"explanation_{idx}" in st.session_state:
                    with st.expander("💡 AI Explanation", expanded=True):
                        st.info(st.session_state[f"explanation_{idx}"])
