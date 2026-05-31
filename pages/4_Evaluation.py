import streamlit as st
import plotly.express as px

from src.evaluation import (
    run_evaluation
)

st.title("📊 Evaluation Dashboard")

if st.button(
    "Run Evaluation"
):

    df = run_evaluation()

    st.dataframe(
        df,
        use_container_width=True
    )

    metrics = [

        "AvgSim@5",
        "Diversity@5",
        "VoteQuality@5"
    ]

    for metric in metrics:

        fig = px.bar(
            df,
            x="Movie",
            y=metric,
            title=metric
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )