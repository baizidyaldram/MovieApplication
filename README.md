---
title: MovieApplication
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# 🎬 Hybrid Movie Recommendation System

> An AI-powered recommendation engine combining Content-Based Filtering, Latent Semantic Analysis, Reciprocal Rank Fusion, XGBoost Re-Ranking, and Generative AI Explanations — delivering accurate, explainable, and personalized movie recommendations.

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue?style=for-the-badge)](https://huggingface.co/spaces/Baizid122/MovieApplication)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](#-deploy-to-hugging-face-spaces-docker)
[![License](https://img.shields.io/badge/License-Educational-green?style=for-the-badge)](#-license)
[![Dataset](https://img.shields.io/badge/Dataset-4%2C375%20Movies-orange?style=for-the-badge)](#-dataset)

[**Live Demo on Hugging Face**](https://huggingface.co/spaces/Baizid122/MovieApplication) · [**GitHub Repository**](https://github.com/baizidyaldram/MovieApplication)

</div>

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
| Containerization | Docker |
| CI/CD | GitHub Actions |

---

## 📂 Project Structure

```
MovieApplication/
│
├── app.py                          # Streamlit application entry point
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker config for Hugging Face Spaces
├── .dockerignore                   # Files excluded from Docker build
├── movies.csv                      # Raw movie dataset
│
├── models/
│   ├── movies_processed.pkl        # Preprocessed movie dataset
│   ├── content_sim.npy             # SBERT content similarity matrix
│   ├── latent_sim.npy              # SVD latent similarity matrix
│   ├── embeddings.npy              # Sentence-BERT embeddings
│   ├── xgb_model.pkl               # Trained XGBoost re-ranker
│   └── metadata.pkl                # Movie metadata
│
├── src/
│   ├── recommender.py              # Core hybrid recommendation logic
│   ├── llm_explainer.py            # OpenRouter AI explanation generator
│   └── evaluation.py               # Evaluation metrics & ablation studies
│
├── .streamlit/
│   └── secrets.toml                # API keys for local dev (not committed)
│
└── .github/
    └── workflows/
        └── huggingface_sync.yml    # CI/CD: auto-deploy to Hugging Face on push
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/baizidyaldram/MovieApplication.git
cd MovieApplication
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 API Configuration

The app requires two API keys. It automatically reads from **Streamlit secrets** (`.streamlit/secrets.toml`) when running locally, or from **environment variables** when deployed in Docker.

### OpenRouter API *(Required — for AI explanations)*

1. Create an account at [openrouter.ai](https://openrouter.ai)
2. Generate an API key

### TMDB API *(Optional — for movie posters)*

1. Create an account at [themoviedb.org](https://www.themoviedb.org)
2. Go to **Settings → API** and generate a key (V3 key or V4 Read Access Token both work)

### Local Development — `secrets.toml`

Create `.streamlit/secrets.toml` in the project root:

```toml
OPENROUTER_API_KEY = "your-openrouter-api-key"
TMDB_API_KEY = "your-tmdb-api-key"
```

> ⚠️ **Do not commit** `secrets.toml` to version control.

### Cloud Deployment — Environment Variables

When deploying to Streamlit Cloud or Hugging Face, add the same keys as environment variables or secrets in the platform's settings panel (see Deployment sections below).

---

## ▶️ Running Locally

```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## 🌐 Deployment

This project supports **two production deployment paths** out of the box.

### 🟣 Option 1 — Deploy to Streamlit Community Cloud

Best for a quick, zero-config deploy directly from GitHub.

1. Push the project to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repository.
3. Set the **Main file path** to `app.py`.
4. Add your API keys in **Advanced Settings → Secrets**:
   ```toml
   OPENROUTER_API_KEY = "sk-..."
   TMDB_API_KEY = "..."
   ```
5. Click **Deploy** — the app will be live in minutes.

#### Streamlit Cloud Features
- Automatic Git-based deploys on every push
- Built-in secrets manager
- Free tier available
- Custom subdomain (e.g., `your-app.streamlit.app`)

---

### 🐳 Option 2 — Deploy to Hugging Face Spaces (Docker)

Best for full control, GPU support, and the Hugging Face ecosystem.

This project includes a `Dockerfile` and a **GitHub Actions CI/CD workflow** that automatically syncs every push to a Hugging Face Space.

#### One-Time Setup

1. **Create a Hugging Face Space:**
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space)
   - Set the **SDK** to **Docker**
   - Set the name (e.g., `MovieApplication`)

2. **Generate a Hugging Face Token:**
   - Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Create a **Write** token

3. **Add Secrets to GitHub:**
   - Go to your GitHub repo → **Settings → Secrets and variables → Actions**
   - Add these repository secrets:

   | Secret Name | Value |
   |-------------|-------|
   | `HF_TOKEN` | Your Hugging Face write token |

4. **Add API Keys to Hugging Face Space:**
   - Go to your Space → **Settings → Variables and secrets**
   - Add these as **secrets**:

   | Secret Name | Value |
   |-------------|-------|
   | `OPENROUTER_API_KEY` | Your OpenRouter API key |
   | `TMDB_API_KEY` | Your TMDB API key |

#### How CI/CD Works

Every push to `main` triggers the GitHub Actions workflow (`.github/workflows/huggingface_sync.yml`), which:

1. Checks out the full repository (including Git LFS files)
2. Force-pushes to the Hugging Face Space via an orphan branch
3. Hugging Face automatically rebuilds the Docker container and redeploys

```
GitHub push → Actions Workflow → Hugging Face Space (auto-rebuild)
```

> 💡 You can also trigger the workflow manually from the **Actions** tab in GitHub.

#### Hugging Face Features
- Docker-based — full control over the runtime
- Supports GPU instances for heavy ML workloads
- Built-in logging and monitoring
- Community visibility and discoverability

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

## 📥 Required Model Files

The following pre-trained files must be in the `models/` directory. They are tracked with **Git LFS** and will be pulled automatically when you clone:

```
models/
├── movies_processed.pkl       # Preprocessed movie dataset (~5.6 MB)
├── content_sim.npy            # SBERT content similarity matrix (~73 MB)
├── latent_sim.npy             # SVD latent similarity matrix (~73 MB)
├── embeddings.npy             # Sentence-BERT embeddings (~6.4 MB)
├── xgb_model.pkl              # Trained XGBoost re-ranker (~519 KB)
└── metadata.pkl               # Movie metadata (~189 KB)
```

> 💡 If Git LFS files aren't pulled, run: `git lfs pull`

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

[![GitHub](https://img.shields.io/badge/GitHub-baizidyaldram-181717?style=for-the-badge&logo=github)](https://github.com/baizidyaldram)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Baizid122-blue?style=for-the-badge)](https://huggingface.co/Baizid122)

---

## 🙏 Acknowledgments

Special thanks to the open-source tools and APIs that made this project possible:

[OpenRouter](https://openrouter.ai) · [TMDB](https://www.themoviedb.org) · [Streamlit](https://streamlit.io) · [Hugging Face](https://huggingface.co) · [Sentence-BERT](https://www.sbert.net) · [Scikit-Learn](https://scikit-learn.org) · [XGBoost](https://xgboost.ai) · [Docker](https://www.docker.com)

---

## 📜 License

This project is intended for **educational**, **research**, and **portfolio demonstration** purposes.

---

<div align="center">

*If you found this project useful, please consider giving it a ⭐ on GitHub — it helps others discover the work!*

</div>
