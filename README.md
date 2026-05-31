# 🎬 Hybrid Movie Recommendation System

An AI-powered Movie Recommendation System that combines **Content-Based Filtering**, **Latent Semantic Analysis (SVD)**, **XGBoost Re-Ranking**, **Reciprocal Rank Fusion (RRF)**, and **Large Language Model (LLM) Explanations** to generate accurate, explainable, and personalized movie recommendations.

Built using **Python**, **Machine Learning**, **Generative AI**, and deployed through **Streamlit**.

---

## 🚀 Features

### 🎯 Hybrid Recommendation Engine

* Content-Based Filtering using Sentence-BERT embeddings
* Latent Semantic Similarity using TruncatedSVD
* Reciprocal Rank Fusion (RRF)
* XGBoost Re-Ranking
* Hybrid weighted recommendation scoring

### 🤖 Explainable AI Recommendations

* OpenRouter LLM Integration
* Natural language explanations for each recommendation
* Explains similarities in:

  * Genre
  * Themes
  * Audience preferences
  * Storytelling style

### 📊 Interactive Dashboard

* Streamlit-based web application
* Recommendation explorer
* Model comparison dashboard
* Evaluation dashboard
* AI recommendation explanations

### 📈 Evaluation Metrics

* Average Similarity @ K
* Diversity @ K
* Vote Quality @ K
* Recommendation Coverage

---

# 🏗️ System Architecture

Movie Query

↓

Sentence-BERT Similarity

↓

Latent Semantic Analysis (SVD)

↓

Reciprocal Rank Fusion (RRF)

↓

XGBoost Re-Ranking

↓

Hybrid Scoring

↓

Top-N Recommendations

↓

OpenRouter AI Explanation

---

# 🛠️ Technologies Used

| Category             | Technology     |
| -------------------- | -------------- |
| Programming Language | Python         |
| Dashboard            | Streamlit      |
| Machine Learning     | Scikit-Learn   |
| Ranking Model        | XGBoost        |
| NLP Embeddings       | Sentence-BERT  |
| Latent Features      | TruncatedSVD   |
| Explainable AI       | OpenRouter API |
| Data Processing      | Pandas, NumPy  |
| Visualization        | Plotly         |

---

# 📂 Project Structure

```text
Movie-Recommender/

├── app.py
├── requirements.txt
│
├── models/
│   ├── movies_processed.pkl
│   ├── content_sim.npy
│   ├── latent_sim.npy
│   ├── embeddings.npy
│   ├── xgb_model.pkl
│   └── metadata.pkl
│
├── src/
│   ├── recommender.py
│   ├── llm_explainer.py
│   └── evaluation.py
│
├── pages/
│   ├── 1_Home.py
│   ├── 2_Model_Explorer.py
│   ├── 3_Hybrid_Recommender.py
│   └── 4_Evaluation.py
│
└── .streamlit/
    └── secrets.toml
```

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/movie-recommendation-system.git

cd movie-recommendation-system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🔑 OpenRouter API Setup

Create:

```text
.streamlit/secrets.toml
```

Add:

```toml
OPENROUTER_API_KEY="YOUR_OPENROUTER_API_KEY"
```

---

# ▶️ Run Locally

```bash
streamlit run app.py
```

Application will start at:

```text
http://localhost:8501
```

---

# 🌐 Deployment

This project is designed for deployment on Streamlit Community Cloud.

1. Push project to GitHub
2. Connect repository to Streamlit Cloud
3. Add OpenRouter API Key in Secrets
4. Deploy

---

# 📊 Dataset

Movie metadata dataset containing:

* Movie Title
* Overview
* Genres
* Keywords
* Runtime
* Popularity
* Vote Average
* Release Year

The dataset is processed into machine-learning-ready features and embedding representations.

---

# 🎯 Future Improvements

* User-based collaborative filtering
* Personalized user profiles
* Real-time recommendation updates
* Movie poster integration (TMDB API)
* User authentication
* Recommendation history tracking
* Feedback-based learning system

---

# 👨‍💻 Author

**Baizid Yaldram**

Master of Data Science
University of Malaya

### Research Interests

* Recommender Systems
* Machine Learning
* Generative AI
* Large Language Models
* Data Science

---

# 📜 License

This project is intended for educational, research, and portfolio purposes.

