import streamlit as st

from src.recommender import (
    movies,
    get_recommendations
)

from src.llm_explainer import (
    explain_movie
)

st.title("🎬 Hybrid Movie Recommendation")

movie = st.selectbox(
    "Select Movie",
    sorted(
        movies["title"].tolist()
    )
)

top_n = st.slider(
    "Number of Recommendations",
    5,
    20,
    10
)

if st.button(
    "Generate Recommendations",
    use_container_width=True
):

    recs = get_recommendations(
        movie,
        top_n
    )

    st.session_state["recs"] = recs
    st.session_state["source"] = movie

if "recs" in st.session_state:

    st.markdown("---")

    for rec in st.session_state["recs"]:

        with st.container(border=True):

            col1, col2 = st.columns([3,1])

            with col1:

                st.subheader(
                    rec["title"]
                )

                st.write(
                    f"⭐ Rating: {rec['rating']}"
                )

                st.write(
                    f"📅 Year: {rec['year']}"
                )

                st.write(
                    ", ".join(
                        rec["genres"]
                    )
                )

            with col2:

                st.metric(
                    "Match %",
                    rec["match"]
                )

            if st.button(
                f"Explain {rec['title']}"
            ):

                with st.spinner(
                    "Generating explanation..."
                ):

                    explanation = explain_movie(
                        st.session_state[
                            "source"
                        ],
                        rec["title"]
                    )

                    st.success(
                        explanation
                    )