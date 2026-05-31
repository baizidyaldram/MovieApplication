import streamlit as st

from src.recommender import (
    movies,
    movie_to_idx,
    content_sim,
    latent_sim
)

st.title("🔬 Model Explorer")

movie = st.selectbox(
    "Choose a Movie",
    sorted(movies["title"].tolist())
)

if movie:

    idx = movie_to_idx[movie]

    tab1, tab2 = st.tabs([
        "SBERT Similarity",
        "Latent Similarity"
    ])

    with tab1:

        sims = content_sim[idx]

        top = sims.argsort()[::-1][1:11]

        rows = []

        for i in top:

            rows.append({

                "Movie":
                movies.iloc[i]["title"],

                "Similarity":
                round(
                    float(sims[i]),
                    4
                )
            })

        st.dataframe(
            rows,
            use_container_width=True
        )

    with tab2:

        sims = latent_sim[idx]

        top = sims.argsort()[::-1][1:11]

        rows = []

        for i in top:

            rows.append({

                "Movie":
                movies.iloc[i]["title"],

                "Similarity":
                round(
                    float(sims[i]),
                    4
                )
            })

        st.dataframe(
            rows,
            use_container_width=True
        )