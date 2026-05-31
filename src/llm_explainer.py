from openai import OpenAI
import streamlit as st

from src.recommender import (
    movie_to_idx,
    movies,
    content_sim
)

client = OpenAI(
    api_key=st.secrets[
        "OPENROUTER_API_KEY"
    ],
    base_url=
    "https://openrouter.ai/api/v1"
)

MODEL = "openai/gpt-oss-120b:free"


def get_movie_overview(
    title,
    max_chars=200
):

    idx = movie_to_idx[title]

    overview = str(
        movies.iloc[idx]["overview"]
    )

    if len(overview) > max_chars:

        overview = (
            overview[:max_chars]
            + "..."
        )

    return overview


def explain_movie(
    source,
    target
):

    src_idx = movie_to_idx[source]
    tgt_idx = movie_to_idx[target]

    similarity = round(
        content_sim[src_idx][tgt_idx]
        * 100
    )

    prompt = f"""
Explain why {target}
is recommended to a fan
of {source}.

Similarity:
{similarity}%

Write 3 concise sentences.

Mention:

1. Themes
2. Genres
3. Audience appeal
"""

    try:

        response = client.chat.completions.create(

            model=MODEL,

            messages=[
                {
                    "role":"user",
                    "content":prompt
                }
            ],

            max_tokens=180,
            temperature=0.6
        )

        return response\
            .choices[0]\
            .message\
            .content

    except Exception as e:

        return (
            f"Explanation unavailable: {e}"
        )