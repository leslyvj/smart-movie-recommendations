# **🎬 Smart Movie Recommender**

> *AI-powered movie recommendations using collaborative filtering and Pearson correlation*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PySpark](https://img.shields.io/badge/PySpark-3.0+-orange.svg)](https://spark.apache.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<p align="center">
  <img src="https://img.shields.io/badge/🎯-User--Based_CF-blueviolet" alt="Collaborative Filtering"/>
  <img src="https://img.shields.io/badge/📊-Pearson_Correlation-success" alt="Pearson"/>
  <img src="https://img.shields.io/badge/⚡-PySpark-yellow" alt="PySpark"/>
</p>

---

## 🌟 Features

- **🎯 Personalized Recommendations** - Get movie suggestions tailored to your taste
- **👥 User Similarity Analysis** - Find users with similar movie preferences
- **📊 Interactive Comparisons** - Compare your ratings with other users
- **🔮 Rating Prediction** - Predict how much you'll like any movie
- **⚡ Distributed Computing** - Powered by Apache Spark for scalability
- **🎨 Beautiful UI** - Intuitive Streamlit interface with real-time updates

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Java 8 or 11 (required for PySpark)
java -version
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/movie-recommender.git
cd movie-recommender

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

Place your data files in the project root:
- `ratings.csv` - User ratings (required)
- `movies.csv` - Movie metadata (optional)

**Expected format:**
```csv
userId,movieId,rating,timestamp
1,1,4.0,964982703
1,3,4.0,964981247
```

### Run the Application

**Option 1: Interactive UI (Recommended)**
```bash
streamlit run streamlit_app.py
```
Then open [http://localhost:8501](http://localhost:8501) in your browser.

**Option 2: Command-Line Pipeline**
```bash
# Run complete pipeline
./run_pipeline.sh

# Or run individual steps
python 01_data_preparation.py
python 02_similarity_calculation.py
python 03_rating_prediction.py
python 04_generate_recommendations.py
```

---

## 📁 Project Structure

```
movie-recommender/
├── 📊 Core Pipeline
│   ├── 01_data_preparation.py        # Data loading & cleaning
│   ├── 02_similarity_calculation.py  # Pearson correlation
│   ├── 03_rating_prediction.py       # Rating predictions
│   └── 04_generate_recommendations.py # Final recommendations
│
├── 🎨 User Interface
│   └── streamlit_app.py              # Interactive Streamlit app
│
├── 🛠️ Configuration & Utilities
│   ├── config.py                     # Settings & parameters
│   └── utils.py                      # Helper functions
│
├── 📂 Data & Outputs
│   ├── ratings.csv                   # Input: User ratings
│   ├── movies.csv                    # Input: Movie metadata
│   └── output/                       # Generated outputs
│       ├── ratings_clean.csv
│       ├── user_averages.csv
│       ├── similarity_scores.csv
│       ├── predicted_ratings.csv
│       └── final_recommendations.csv
│
└── 📄 Documentation
    ├── README.md
    ├── requirements.txt
    └── docs/
```

---

## ⚙️ Configuration

Edit [`config.py`](config.py) to customize behavior:

```python
# Target user for batch processing
TARGET_USER_ID = 10

# Algorithm parameters
MIN_COMMON_MOVIES = 3       # Min overlap for similarity
TOP_K_SIMILAR_USERS = 50    # Similar users to consider
TOP_N_RECOMMENDATIONS = 10  # Number of recommendations

# Data paths
RATINGS_PATH = "ratings.csv"
MOVIES_PATH = "movies.csv"
OUTPUT_DIR = "output"
```

---

## 🧮 How It Works

### Algorithm Overview

The system uses **User-Based Collaborative Filtering** with **Pearson Correlation**:

```
1. Data Preparation
   ├─ Load & clean ratings
   ├─ Calculate user averages
   └─ Build user-item matrix

2. Similarity Calculation
   ├─ Find co-rated movies
   ├─ Compute Pearson correlation
   └─ Select top-K similar users

3. Rating Prediction
   ├─ Identify unseen movies
   ├─ Apply weighted formula
   └─ Generate predictions

4. Recommendations
   ├─ Rank predictions
   ├─ Join with metadata
   └─ Present top-N results
```

### Mathematical Foundation

**Pearson Correlation:**
```
pearson(u,v) = Σ[(rᵤᵢ - r̄ᵤ)(rᵥᵢ - r̄ᵥ)] / √[Σ(rᵤᵢ - r̄ᵤ)² × Σ(rᵥᵢ - r̄ᵥ)²]
```

**Rating Prediction:**
```
p(u,i) = r̄ᵤ + [Σᵥ sim(u,v) × (rᵥ,ᵢ - r̄ᵥ)] / [Σᵥ |sim(u,v)|]
```

Where:
- `r̄ᵤ` = average rating of user u
- `sim(u,v)` = Pearson correlation between users
- `rᵥ,ᵢ` = rating by user v for item i

---

## 📊 Pipeline Details

### 1️⃣ Data Preparation
[`01_data_preparation.py`](01_data_preparation.py)

```python
# Key functions
load_and_explore_data()    # Load CSVs, show statistics
clean_data()               # Remove nulls, duplicates
calculate_user_averages()  # Compute per-user means
```

**Output:** `ratings_clean.csv`, `user_averages.csv`

---

### 2️⃣ Similarity Calculation
[`02_similarity_calculation.py`](02_similarity_calculation.py)

```python
# Key functions
find_common_movies()              # Identify co-rated items
calculate_pearson_correlation()   # Compute similarity
select_top_similar_users()        # Filter top-K
```

**Output:** `similarity_scores.csv`, `top_similar_users.csv`

---

### 3️⃣ Rating Prediction
[`03_rating_prediction.py`](03_rating_prediction.py)

```python
# Key functions
identify_unseen_movies()    # Find unrated items
calculate_predictions()     # Apply prediction formula
```

**Output:** `predicted_ratings.csv`

---

### 4️⃣ Generate Recommendations
[`04_generate_recommendations.py`](04_generate_recommendations.py)

```python
# Key functions
generate_top_recommendations()  # Rank & select top-N
display_recommendations()       # Format output
```

**Output:** `final_recommendations.csv`

---

## 🎨 Streamlit Interface

### Features

**Tab 1: Get My Recommendations**
- Rate 5-10 movies from popular list
- Find your movie soulmates (similar users)
- Get personalized top-10 recommendations
- See detailed explanations for each suggestion

**Tab 2: Compare with Users**
- Compare your taste with any user
- View Pearson correlation scores
- See rating patterns side-by-side
- Discover movies they loved that you haven't seen

**Tab 3: Explore Similar Users**
- Browse all users similar to you
- Filter by similarity threshold
- View user profiles and statistics
- Interactive similarity visualization

### Key Functions

```python
# Core recommendation engine
find_similar_users()          # Find top-K similar users
generate_recommendations()    # Predict ratings & recommend
predict_rating_for_movie()    # Single movie prediction

# User analysis
compare_users()               # Compare two users
get_user_profile()            # Get user statistics
calculate_pearson_correlation() # Compute similarity

# UI helpers
get_star_rating()             # Display star icons
create_spark_session()        # Initialize Spark
```

---

## 🔧 Utilities & Helpers

[`utils.py`](utils.py) - Common functions:

```python
create_spark_session()      # Initialize Spark with config
load_dataframe()            # Load CSV to Spark DataFrame
save_dataframe()            # Save DataFrame to CSV
create_output_directory()   # Setup output folder
calculate_statistics()      # Compute dataset stats
```

---

## 📈 Performance & Scalability

**Optimization Techniques:**
- ✅ Spark DataFrame caching for frequent operations
- ✅ Broadcast joins for small lookup tables
- ✅ Partitioning strategies for large datasets
- ✅ Streamlit caching (`@st.cache_resource`, `@st.cache_data`)

**Benchmarks:**
| Dataset Size | Users | Movies | Processing Time |
|--------------|-------|--------|-----------------|
| Small        | 100   | 1,000  | ~30 seconds     |
| Medium       | 1,000 | 10,000 | ~3 minutes      |
| Large        | 10,000| 50,000 | ~15 minutes     |

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **🐛 Report Bugs** - Open an issue with details
2. **💡 Suggest Features** - Share your ideas
3. **🔧 Submit PRs** - Fork, code, and submit pull requests
4. **📖 Improve Docs** - Help us write better documentation

### Development Setup

```bash
# Fork and clone the repo
git clone https://github.com/yourusername/movie-recommender.git
cd movie-recommender

# Create feature branch
git checkout -b feature/amazing-feature

# Make changes and commit
git commit -m "Add amazing feature"

# Push and create PR
git push origin feature/amazing-feature
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 🐛 Troubleshooting

### Common Issues

**1. Spark/PySpark Environment Issues**
```bash
# Ensure Python version matches
export PYSPARK_PYTHON=$(which python3)
export PYSPARK_DRIVER_PYTHON=$(which python3)
```

**2. Missing Files Error**
```
Error: ratings.csv not found
Solution: Ensure ratings.csv is in project root directory
```

**3. Low Similarity / Few Recommendations**
```python
# In config.py, lower the threshold
MIN_COMMON_MOVIES = 2  # Reduce from 3 to 2
```

**4. Out of Memory**
```python
# In utils.py or streamlit_app.py, increase memory
.config("spark.driver.memory", "8g")  # Increase from 4g
```

**5. Java Not Found**
```bash
# Install Java 8 or 11
# Ubuntu/Debian:
sudo apt install openjdk-11-jdk

# macOS:
brew install openjdk@11

# Set JAVA_HOME
export JAVA_HOME=/path/to/java
```



## 🎓 Learn More

### Research Papers
- Breese, J.S., et al. (1998). *Empirical Analysis of Predictive Algorithms for Collaborative Filtering*
- Resnick, P., et al. (1994). *GroupLens: An Open Architecture for Collaborative Filtering*
- Sarwar, B., et al. (2001). *Item-Based Collaborative Filtering Recommendation Algorithms*

### Related Projects
- [Surprise](https://surpriselib.com/) - Python scikit for recommender systems
- [LightFM](https://github.com/lyst/lightfm) - Hybrid recommendation algorithms
- [Apache Mahout](https://mahout.apache.org/) - Scalable machine learning library

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```



## 🙏 Acknowledgments

- MovieLens dataset by [GroupLens Research](https://grouplens.org/datasets/movielens/)
- Apache Spark community for the distributed computing framework
- Streamlit team for the amazing UI framework
- All contributors who have helped improve this project

---

## 📊 Project Status

![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/movie-recommender)
![GitHub issues](https://img.shields.io/github/issues/yourusername/movie-recommender)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/movie-recommender)
![GitHub stars](https://img.shields.io/github/stars/yourusername/movie-recommender?style=social)

**Current Version:** 1.0.0  
**Status:** Active Development  
**Last Updated:** January 2025

---

## 🗺️ Roadmap

- [x] Core collaborative filtering implementation
- [x] Streamlit interactive interface
- [x] User comparison features
- [ ] Matrix factorization (SVD) support
- [ ] Deep learning recommendations (Neural CF)
- [ ] Content-based filtering integration
- [ ] API endpoint deployment
- [ ] Docker containerization
- [ ] Performance benchmarking suite
- [ ] A/B testing framework



---

<div align="center">

**If you found this project helpful, please consider giving it a ⭐!**

Made with ❤️ and ☕

[Report Bug](https://github.com/yourusername/movie-recommender/issues) · [Request Feature](https://github.com/yourusername/movie-recommender/issues) · [Documentation](https://github.com/yourusername/movie-recommender/wiki)

</div>

---

<p align="center">
  <sub>Built with PySpark, Streamlit, and lots of ☕</sub>
</p>
