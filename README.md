# 🎬 Hybrid Movie Recommendation System

> An AI-powered recommendation engine combining Content-Based Filtering, Latent Semantic Analysis, Reciprocal Rank Fusion, XGBoost Re-Ranking, and Generative AI Explanations — delivering accurate, explainable, and personalized movie recommendations.

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-Educational-green?style=flat-square)](#license)
[![Dataset](https://img.shields.io/badge/Dataset-4%2C375%20Movies-orange?style=flat-square)](#dataset)

---

## 📌 Overview

Users can either **describe their preferences in natural language** or **select a movie they already enjoy** to receive intelligent, context-aware recommendations with AI-generated explanations.

The system fuses multiple ML techniques into a single hybrid pipeline, then uses a large language model to explain *why* each movie was recommended — making it both accurate and interpretable.

---

## 🚀 Features

### 🎯 Dual Recommendation Modes

| Mode | How It Works |
|------|-------------|
| **✍️ Text-Based Search** | Describe what you want: *"sci-fi thrillers like Inception"*, *"romantic dramas from the 2010s"* |
| **🎬 Movie-Based Search** | Pick a movie you love and get similar recommendations based on content and latent patterns |

### 🧠 Hybrid Recommendation Pipeline

```
User Input
    │
    ├── Text Query ──► SBERT Encoding ──► Semantic Similarity
    │
    └── Movie Title ─► Content Matching ─► Latent Similarity (SVD)
                                │
                         RRF Score Fusion
                                │
                       XGBoost Re-Ranking
                                │
                      Hybrid Weighted Score
                                │
                     Top-N Recommendations
                                │
                  OpenRouter GPT Explanation
                                │
                   Results + Movie Posters (TMDB)
```

### ⚖️ Hybrid Weighted Scoring

| Component | Weight |
|-----------|--------|
| SBERT Content Similarity | **50%** |
| XGBoost Re-Ranking Score | **25%** |
| Latent Similarity (SVD) | **15%** |
| RRF Score | **10%** |

### 🤖 Explainable AI

Powered by **OpenRouter GPT-OSS-120B**, each recommendation includes a human-readable explanation covering:

- Genre and tone similarity
- Plot and storyline connections
- Atmosphere matching
- Audience appeal reasoning

### 📊 Interactive Dashboard

The Streamlit dashboard includes:
- 🎬 Movie Spotlight section
- 📈 Exploratory Data Analysis (EDA)
- ⭐ Rating distribution visualizations
- 📅 Release year trend analysis
- 📊 Model performance metrics

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | **85%+** |
| Precision | **0.83** |
| Recall | **0.81** |
| F1 Score | **0.82** |
| AUC-ROC | **0.90** |

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.9+ |
| Web Framework | Streamlit |
| NLP Embeddings | Sentence-BERT (`all-MiniLM-L6-v2`) |
| Latent Features | TruncatedSVD (Scikit-Learn) |
| Re-Ranking | XGBoost |
| Rank Fusion | Reciprocal Rank Fusion (RRF) |
| Explainable AI | OpenRouter GPT-OSS-120B |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Movie Posters | TMDB API |

---

## 📂 Project Structure

```
Movie-Recommender/
│
├── app.py                          # Streamlit application entry point
├── requirements.txt
│
├── models/
│   ├── movies_processed.pkl        # Preprocessed movie dataset
│   ├── content_sim.npy             # SBERT content similarity matrix
│   ├── latent_sim.npy              # SVD latent similarity matrix
│   ├── xgb_model.pkl               # Trained XGBoost re-ranker
│   └── metadata.pkl                # Movie metadata
│
├── src/
│   ├── recommender.py              # Core hybrid recommendation logic
│   ├── llm_explainer.py            # OpenRouter AI explanation generator
│   └── evaluation.py              # Evaluation metrics & ablation studies
│
└── .streamlit/
    └── secrets.toml                # API keys (not committed to repo)
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/baizidyaldram/movieapplication.git
cd movieapplication
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 API Configuration

### OpenRouter API *(Required — for AI explanations)*

1. Create an account at [openrouter.ai](https://openrouter.ai)
2. Generate an API key
3. Create `.streamlit/secrets.toml` and add:

```toml
OPENROUTER_API_KEY = "your-openrouter-api-key"
TMDB_API_KEY = "your-tmdb-api-key"
```

### TMDB API *(Optional — for movie posters)*

1. Create an account at [themoviedb.org](https://www.themoviedb.org)
2. Generate an API key and add it to `secrets.toml`

---

## 📥 Required Model Files

Place the following pre-trained files inside the `models/` directory before running:

```
models/
├── movies_processed.pkl
├── content_sim.npy
├── latent_sim.npy
├── xgb_model.pkl
└── metadata.pkl
```

---

## ▶️ Running Locally

```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## 🌐 Deployment

The app is deployment-ready for several platforms:

### Streamlit Community Cloud *(Recommended)*
1. Push the project to GitHub
2. Connect the repository at [share.streamlit.io](https://share.streamlit.io)
3. Add your API keys in the Secrets manager
4. Deploy

### Other Platforms
- **Hugging Face Spaces** — supports Streamlit apps natively
- **Railway / Render** — containerized deployment
- **Heroku** — via `Procfile` configuration

---

## 📊 Dataset

The system uses a processed movie metadata dataset of approximately **4,375 movies**.

| Feature | Description |
|---------|-------------|
| Title | Movie title |
| Overview | Plot summary |
| Genres | Genre categories |
| Keywords | Story keywords |
| Vote Average | Average rating (0–10) |
| Vote Count | Number of votes |
| Popularity | TMDB popularity score |
| Release Year | Year of release |
| Runtime | Movie duration (minutes) |

**Dataset Statistics:**

| Metric | Value |
|--------|-------|
| Total Movies | ~4,375 |
| Year Range | 1916 – 2016 |
| Genres | 19+ |
| Rating Scale | 0 – 10 |

---

## 🎯 Example Queries

### Text-Based Search
```
"funny action movies with high ratings"
"romantic dramas from the 2010s"
"sci-fi thrillers like Inception"
"scary horror movies with strong plots"
```

### Movie-Based Search
```
The Dark Knight  →  similar funny films
Inception        →  with a romantic element
Interstellar     →  mystery-focused matches
Titanic          →  adventure alternatives
```

---

## 🔭 Future Improvements

- [ ] User-based collaborative filtering
- [ ] Personalized watch history & authentication
- [ ] Like / Dislike feedback loop
- [ ] Recommendation history tracking
- [ ] Export recommendations to CSV
- [ ] Movie trailer integration
- [ ] Social sharing features
- [ ] Real-time recommendation updates

---

## 🤝 Contributing

Contributions are welcome!

```bash
# Fork the repository

# Create a feature branch
git checkout -b feature/YourFeatureName

# Commit your changes
git commit -m "Add YourFeatureName"

# Push to your branch
git push origin feature/YourFeatureName

# Open a Pull Request
```

---

## 👨‍💻 Author

**Baizid Yaldram**
Master of Data Science — University of Malaya

*Research interests: Recommender Systems · Machine Learning · Generative AI · LLMs · NLP*

[![GitHub](https://img.shields.io/badge/GitHub-baizidyaldram-181717?style=flat-square&logo=github)](https://github.com/baizidyaldram)

---

## 🙏 Acknowledgments

Special thanks to the open-source tools and APIs that made this project possible:

[OpenRouter](https://openrouter.ai) · [TMDB](https://www.themoviedb.org) · [Streamlit](https://streamlit.io) · [Sentence-BERT](https://www.sbert.net) · [Scikit-Learn](https://scikit-learn.org) · [XGBoost](https://xgboost.ai)

---

## 📜 License

This project is intended for **educational**, **research**, and **portfolio demonstration** purposes.

---

*If you found this project useful, please consider giving it a ⭐ on GitHub — it helps others discover the work!*
