import streamlit as st
import requests
import pandas as pd
from typing import List, Dict, Any
import json

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Hybrid Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.block-container {
    padding-top: 1rem;
}

.metric-card {
    background-color: #262730;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state for API key
if 'openrouter_api_key' not in st.session_state:
    st.session_state.openrouter_api_key = None

# Sidebar for API key input (only shown if not in secrets)
with st.sidebar:
    st.header("🔑 API Configuration")
    
    # Try to get API key from secrets first
    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            st.session_state.openrouter_api_key = st.secrets["OPENROUTER_API_KEY"]
            st.success("✅ OpenRouter API key loaded from secrets!")
        else:
            st.warning("⚠️ No API key found in secrets")
            api_key_input = st.text_input(
                "Enter OpenRouter API Key:",
                type="password",
                help="Get your key from https://openrouter.io/keys"
            )
            if api_key_input:
                st.session_state.openrouter_api_key = api_key_input
                st.success("✅ API key set!")
    except Exception as e:
        st.error(f"Error loading secrets: {str(e)}")
        api_key_input = st.text_input(
            "Enter OpenRouter API Key:",
            type="password",
            help="Get your key from https://openrouter.io/keys"
        )
        if api_key_input:
            st.session_state.openrouter_api_key = api_key_input
            st.success("✅ API key set!")

# Main content
st.title("🎬 Hybrid Movie Recommendation System")

st.markdown("""
### AI-Powered Movie Recommendations

This project combines:

- Sentence-BERT Semantic Similarity
- Latent Semantic Analysis (SVD)
- XGBoost Re-Ranking
- Reciprocal Rank Fusion (RRF)
- Generative AI Explanations (OpenRouter)

Use the navigation menu on the left to explore the system.
""")

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Models", "4")

with col2:
    st.metric("Pipeline", "Hybrid")

with col3:
    st.metric("LLM", "OpenRouter")

with col4:
    st.metric("Deployment", "Streamlit")

st.markdown("---")

st.subheader("🏗 Recommendation Pipeline")

st.markdown("""
```text
Movie Query
     │
     ▼
SBERT Similarity
     │
     ▼
SVD Latent Similarity
     │
     ▼
Reciprocal Rank Fusion
     │
     ▼
XGBoost Re-Ranking
     │
     ▼
Hybrid Score
     │
     ▼
Top Recommendations
     │
     ▼
OpenRouter AI Explanation
""")
