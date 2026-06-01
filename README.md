🎬 Hybrid Movie Recommendation System

An AI-powered Hybrid Movie Recommendation System that combines Content-Based Filtering, Latent Semantic Analysis (SVD), Reciprocal Rank Fusion (RRF), XGBoost Re-Ranking, and Generative AI Explanations to deliver accurate, explainable, and personalized movie recommendations.

Users can either describe their preferences in natural language or select a movie they already enjoy to receive intelligent, context-aware recommendations with AI-generated explanations.

Built using Python, Machine Learning, Generative AI, and deployed through Streamlit.

🚀 Features
🎯 Dual Recommendation Methods
✍️ Text-Based Search

Describe what you're looking for in natural language.

Examples:

"funny action movies with high ratings"
"romantic dramas from the 2010s"
"sci-fi thrillers like Inception"
🎬 Movie-Based Search

Select a movie you already enjoy and receive similar recommendations based on content and latent patterns.

🧠 Hybrid Recommendation Engine

The recommendation pipeline combines multiple techniques to improve recommendation quality:

Content-Based Filtering
Sentence-BERT embeddings (all-MiniLM-L6-v2)
Semantic understanding of movie plots and overviews
Latent Semantic Analysis
TruncatedSVD for hidden feature extraction
Captures deeper relationships between movies
Reciprocal Rank Fusion (RRF)
Combines rankings from multiple recommendation sources
XGBoost Re-Ranking
Machine learning-based ranking optimization
Improves recommendation ordering
Hybrid Weighted Scoring
Component	Weight
SBERT Content Similarity	50%
XGBoost Score	25%
Latent Similarity (SVD)	15%
RRF Score	10%
🤖 Explainable AI Recommendations

The system integrates OpenRouter GPT-OSS-120B to generate human-readable explanations for each recommendation.

Explanation Highlights
Genre similarity
Plot and storyline connections
Tone and atmosphere matching
Audience appeal
Recommendation reasoning
📊 Interactive Dashboard

The Streamlit dashboard includes:

🎬 Movie Spotlight section
📈 Interactive Exploratory Data Analysis (EDA)
⭐ Rating distribution visualizations
📅 Release year trend analysis
📊 Model performance metrics
ℹ️ About Project section
📈 Model Performance
Metric	Score
Accuracy	85%+
Precision	0.83
Recall	0.81
F1 Score	0.82
AUC-ROC	0.90
🏗️ System Architecture
User Input (Text Description OR Movie Selection)
                    │
                    ▼
        ┌─────────────────────┐
        │ Recommendation Mode │
        └─────────────────────┘
             │           │
             ▼           ▼

     Text-Based      Movie-Based
       Query           Query
         │               │
         ▼               ▼

   SBERT Encoding   Content Similarity
         │               │
         ▼               ▼

  Semantic Match   Latent Similarity
         │               │
         ▼               ▼

  Rating Boost       RRF Fusion
         │               │
         └──────┬────────┘
                ▼

       XGBoost Re-Ranking
                │
                ▼

      Hybrid Weighted Score
                │
                ▼

      Top-N Recommendations
                │
                ▼

     OpenRouter AI Explanation
                │
                ▼

      Results + Movie Posters
🛠️ Technologies Used
Category	Technology
Programming Language	Python 3.9+
Web Framework	Streamlit
Machine Learning	Scikit-Learn
Gradient Boosting	XGBoost
NLP Embeddings	Sentence-BERT
Latent Features	TruncatedSVD
Explainable AI	OpenRouter GPT-OSS-120B
Data Processing	Pandas, NumPy
Visualization	Plotly
Movie Posters	TMDB API
📂 Project Structure
Movie-Recommender/
│
├── app.py
├── requirements.txt
│
├── models/
│   ├── movies_processed.pkl
│   ├── content_sim.npy
│   ├── latent_sim.npy
│   ├── xgb_model.pkl
│   └── metadata.pkl
│
├── src/
│   ├── recommender.py
│   ├── llm_explainer.py
│   └── evaluation.py
│
└── .streamlit/
    └── secrets.toml
📦 Installation
1. Clone the Repository
git clone https://github.com/baizidyaldram/movieapplication.git
cd movieapplication
2. Create Virtual Environment (Optional)
Windows
python -m venv venv
venv\Scripts\activate
Linux / Mac
python -m venv venv
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
🔑 API Configuration
OpenRouter API (Required)
Create an account on OpenRouter
Generate an API key
Create:
.streamlit/secrets.toml

Add:

OPENROUTER_API_KEY = "your-openrouter-api-key"
TMDB_API_KEY = "your-tmdb-api-key"
TMDB API (Optional)

Used for movie posters.

Create a TMDB account
Generate an API key
Add it to secrets.toml
📥 Required Model Files

Place the following files inside the models/ directory:

movies_processed.pkl
content_sim.npy
latent_sim.npy
xgb_model.pkl
metadata.pkl
▶️ Run Locally
streamlit run app.py

Application URL:

http://localhost:8501
🌐 Deployment

The project is deployment-ready for:

Streamlit Community Cloud
Push project to GitHub
Connect repository to Streamlit Cloud
Add secrets
Deploy
Other Platforms
Hugging Face Spaces
Railway
Render
Heroku
📊 Dataset

The recommendation system uses a processed movie metadata dataset containing approximately 4,375 movies.

Dataset Features
Feature	Description
Title	Movie title
Overview	Plot summary
Genres	Genre categories
Keywords	Story keywords
Vote Average	Average rating
Vote Count	Number of votes
Popularity	TMDB popularity score
Release Year	Movie release year
Runtime	Movie duration
Dataset Statistics
Metric	Value
Movies	~4,375
Year Range	1916 – 2016
Genres	19+
Rating Scale	0 – 10
🎯 Example Queries
Text-Based Search
funny action movies with high ratings
romantic dramas from the 2010s
sci-fi thrillers like Inception
scary horror movies with strong plots
Movie-Based Search
The Dark Knight + funny
Inception + romantic
Interstellar + mystery
Titanic + adventure
🧪 Evaluation Results
Metric	Score
Accuracy	85%+
Precision	0.83
Recall	0.81
F1 Score	0.82
AUC-ROC	0.90
🎯 Future Improvements
User-based collaborative filtering
Personalized watch history
User authentication system
Recommendation history tracking
Feedback system (Like/Dislike)
Export recommendations to CSV
Enhanced poster resolution
Movie trailer integration
Social sharing features
Real-time recommendation updates
🤝 Contributing

Contributions are welcome.

Steps
# Fork repository

# Create feature branch
git checkout -b feature/AmazingFeature

# Commit changes
git commit -m "Add AmazingFeature"

# Push changes
git push origin feature/AmazingFeature

# Open Pull Request
👨‍💻 Author

Baizid Yaldram

Master of Data Science
University of Malaya

Research Interests
Recommender Systems
Machine Learning
Generative AI
Large Language Models (LLMs)
Natural Language Processing
🙏 Acknowledgments

Special thanks to:

OpenRouter
TMDB
Streamlit
Sentence-BERT
Scikit-Learn
XGBoost

for providing the tools and resources that made this project possible.

📜 License

This project is intended for:

Educational purposes
Research purposes
Portfolio demonstration
⭐ Support

If you found this project useful, please consider giving the repository a ⭐ Star on GitHub.

It helps others discover the project and supports future development.
