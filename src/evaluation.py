import numpy as np
import pandas as pd

from src.recommender import (
    movies,
    movie_to_idx,
    get_recommendations,
    content_sim
)


def diversity_at_k(
    recommendations,
    k=5
):

    indices = [

        movie_to_idx[
            m["title"]
        ]

        for m in recommendations[:k]
    ]

    if len(indices) < 2:
        return 0

    sims = []

    for i in range(
        len(indices)
    ):
        for j in range(
            i+1,
            len(indices)
        ):

            sims.append(
                content_sim[
                    indices[i]
                ][
                    indices[j]
                ]
            )

    return float(
        1 - np.mean(sims)
    )


def run_evaluation():

    test_movies = [

        "The Dark Knight",
        "Inception",
        "Iron Man",
        "Avatar",
        "Titanic",
        "The Godfather",
        "Pulp Fiction",
        "Interstellar",
        "Fight Club"
    ]

    results = []

    for movie in test_movies:

        if movie not in movie_to_idx:
            continue

        recs = get_recommendations(
            movie,
            10
        )

        diversity = diversity_at_k(
            recs
        )

        quality = np.mean([

            r["rating"]

            for r in recs[:5]

        ])

        similarity = np.mean([

            content_sim[
                movie_to_idx[movie]
            ][
                movie_to_idx[
                    r["title"]
                ]
            ]

            for r in recs[:5]

        ])

        results.append({

            "Movie":
                movie,

            "AvgSim@5":
                similarity,

            "Diversity@5":
                diversity,

            "VoteQuality@5":
                quality
        })

    return pd.DataFrame(
        results
    )