import streamlit as st
import plotly.express as px
import pandas as pd

from src.evaluation import (
    run_evaluation
)

st.title("📊 Evaluation Dashboard")

st.markdown("""
### Model Performance Metrics

- **AvgSim@5**: Average similarity score of top 5 recommendations
- **Diversity@5**: Diversity of genres in top 5 recommendations
- **VoteQuality@5**: Average vote quality of recommended movies
""")

if st.button(
    "🚀 Run Full Evaluation",
    use_container_width=True,
    type="primary"
):
    
    with st.spinner("Running evaluation on test queries..."):
        try:
            df = run_evaluation()
            
            st.success("✅ Evaluation complete!")
            
            # Display metrics table
            st.subheader("📈 Evaluation Results")
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )
            
            # Visualizations
            st.subheader("📊 Performance Visualizations")
            
            metrics = [
                "AvgSim@5",
                "Diversity@5",
                "VoteQuality@5"
            ]
            
            # Check which metrics exist in dataframe
            available_metrics = [m for m in metrics if m in df.columns]
            
            if available_metrics:
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
            
            # Summary statistics
            st.subheader("📋 Summary Statistics")
            summary_stats = df[available_metrics].describe()
            st.dataframe(summary_stats, use_container_width=True)
            
        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")
            st.info("Please ensure all model files and data are properly loaded.")
else:
    st.info("Click the button above to run the model evaluation.")

# Display explanation of metrics
with st.expander("📖 Understanding the Metrics"):
    st.markdown("""
    - **AvgSim@5**: Higher is better (1.0 = perfect similarity)
    - **Diversity@5**: Higher is better (more varied recommendations)
    - **VoteQuality@5**: Higher is better (higher rated movies)
    
    The hybrid model aims to balance all three metrics for optimal recommendations.
    """)
