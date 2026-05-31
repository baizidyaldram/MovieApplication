import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler


@st.cache_resource
def load_assets():

    movies = pd.read_pickle(
        "models/movies_processed.pkl"
    )

    content_sim = np.load(
        "models/content_sim.npy"
    )

    latent_sim = np.load(
        "models/latent_sim.npy"
    )

    ranker = joblib.load(
        "models/xgb_model.pkl"
    )

    with open(
        "models/metadata.pkl",
        "rb"
    ) as f:

        metadata = pickle.load(f)

    return (
        movies,
        content_sim,
        latent_sim,
        ranker,
        metadata
    )


movies,\
content_sim,\
latent_sim,\
ranker,\
metadata = load_assets()

movie_to_idx = metadata["movie_to_idx"]
lower_to_title = metadata["lower_to_title"]

year_min = metadata["year_min"]
year_max = metadata["year_max"]


def normalize_year(year):

    if year_max == year_min:
        return 0.5

    return (
        year - year_min
    ) / (
        year_max - year_min
    )


def fuzzy_find_movie(query):

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

    fi = movies.iloc[i]["franchise"]
    fj = movies.iloc[j]["franchise"]

    if len(fi) >= 5 and fi == fj:
        return 1.0

    return 0.0


@st.cache_resource
def build_keyword_matrix():

    n = len(movies)

    matrix = np.zeros(
        (n, n),
        dtype=np.float32
    )

    keyword_sets = [
        set(movies.iloc[i]["keywords"])
        for i in range(n)
    ]

    for i in range(n):

        if not keyword_sets[i]:
            continue

        for j in range(i + 1, n):

            if not keyword_sets[j]:
                continue

            inter = len(
                keyword_sets[i]
                & keyword_sets[j]
            )

            union = len(
                keyword_sets[i]
                | keyword_sets[j]
            )

            if union > 0:

                score = inter / union

                matrix[i][j] = score
                matrix[j][i] = score

    return matrix


kw_sim_mat = build_keyword_matrix()


def build_feature(
    src_idx,
    cand_idx
):

    m = movies.iloc[cand_idx]

    return [

        m["popularity"] /
        (movies["popularity"].max() + 1e-9),

        m["vote_average"] / 10.0,

        m["runtime"] /
        (movies["runtime"].max() + 1e-9),

        normalize_year(
            m["year"]
        ),

        len(
            m["genres"]
        ) / 10.0,

        len(
            m["keywords"]
        ) / 20.0,

        float(
            kw_sim_mat[src_idx][cand_idx]
        ),

        float(
            content_sim[src_idx][cand_idx]
        ),

        float(
            latent_sim[src_idx][cand_idx]
        ),

        franchise_similarity(
            src_idx,
            cand_idx
        )
    ]


CANDIDATE_POOL = 300


def safe_normalize(arr):

    if arr.max() == arr.min():
        return np.zeros_like(arr)

    return MinMaxScaler()\
        .fit_transform(
            arr.reshape(-1, 1)
        )\
        .flatten()


def get_recommendations(
    movie_name,
    top_n=10
):

    canonical = fuzzy_find_movie(
        movie_name
    )

    if canonical is None:
        return []

    idx = movie_to_idx[
        canonical
    ]

    content_ranked = np.argsort(
        content_sim[idx]
    )[::-1][1:CANDIDATE_POOL+1]

    latent_ranked = np.argsort(
        latent_sim[idx]
    )[::-1][1:CANDIDATE_POOL+1]

    rrf_scores = defaultdict(float)

    for rank, c in enumerate(
        content_ranked
    ):
        rrf_scores[int(c)] += (
            1.0 / (60 + rank)
        )

    for rank, c in enumerate(
        latent_ranked
    ):
        rrf_scores[int(c)] += (
            1.0 / (60 + rank)
        )

    candidates = list(
        rrf_scores.keys()
    )

    raw_content = np.array([
        content_sim[idx][c]
        for c in candidates
    ])

    raw_latent = np.array([
        latent_sim[idx][c]
        for c in candidates
    ])

    raw_rrf = np.array([
        rrf_scores[c]
        for c in candidates
    ])

    raw_xgb = np.array([

        ranker.predict(
            np.array([
                build_feature(
                    idx,
                    c
                )
            ])
        )[0]

        for c in candidates
    ])

    hybrid = (

        0.50 *
        safe_normalize(raw_content)

        +

        0.25 *
        safe_normalize(raw_xgb)

        +

        0.15 *
        safe_normalize(raw_latent)

        +

        0.10 *
        safe_normalize(raw_rrf)

    )

    top_idx = np.argsort(
        hybrid
    )[::-1][:top_n]

    results = []

    for i in top_idx:

        movie_row = movies.iloc[
            candidates[i]
        ]

        results.append({

            "title":
                movie_row["title"],

            "year":
                movie_row["year"],

            "genres":
                movie_row["genres"],

            "rating":
                movie_row["vote_average"],

            "match":
                round(
                    hybrid[i] * 100,
                    2
                )
        })

    return results