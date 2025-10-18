"""
🎬 Smart Movie Recommender - Streamlit Application
A collaborative filtering movie recommendation system with user interaction and transparency.

Features:
1. Get personalized movie recommendations based on your ratings
2. Compare your taste with similar users
3. Explore user similarity patterns and profiles
"""

import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, count, sum as spark_sum, sqrt, lit, when,
    abs as spark_abs, round as spark_round, desc, first
)
from pyspark.sql.types import IntegerType, FloatType
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import os
from typing import Dict, List, Tuple, Optional

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🎬 Smart Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    /* Main styling */
    .main {
        background-color: #F7F7F7;
    }
    
    /* Movie cards */
    .movie-card {
        background: white;
        color: #2d3436;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .movie-card h4 {
        color: #2d3436;
        margin-top: 0;
    }
    
    .movie-card p, .movie-card ul, .movie-card li {
        color: #2d3436;
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    
    /* Similarity badges */
    .similarity-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
    }
    
    .similarity-very-high {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .similarity-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .similarity-medium {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    .similarity-low {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FF6B6B;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    /* Star ratings */
    .star-rating {
        color: #FFD700;
        font-size: 20px;
    }
    
    /* Profile sections */
    .profile-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    /* Confidence badges */
    .confidence-high {
        background-color: #95E1D3;
        color: #2d3436;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
    }
    
    .confidence-medium {
        background-color: #FFD93D;
        color: #2d3436;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
    }
    
    .confidence-low {
        background-color: #FF6B6B;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
    }
    
    /* Rank badges */
    .rank-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px;
        border-radius: 50%;
        font-weight: bold;
        font-size: 18px;
        width: 40px;
        height: 40px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SPARK SESSION & DATA LOADING
# ============================================================================

@st.cache_resource
def create_spark_session():
    """Create and configure a Spark session"""
    import sys
    
    # Set environment variables for Windows compatibility
    os.environ['SPARK_LOCAL_HOSTNAME'] = 'localhost'
    python_exe = sys.executable
    os.environ['PYSPARK_PYTHON'] = python_exe
    os.environ['PYSPARK_DRIVER_PYTHON'] = python_exe
    
    spark = SparkSession.builder \
        .appName("StreamlitMovieRecommender") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "10") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    return spark


@st.cache_resource
def load_ratings_data():
    """Load ratings data from CSV"""
    try:
        spark = create_spark_session()
        
        ratings_df = spark.read.csv(
            "./ratings.csv",
            header=True,
            inferSchema=True
        ).drop("timestamp")
        
        ratings_df = ratings_df \
            .withColumn("userId", col("userId").cast(IntegerType())) \
            .withColumn("movieId", col("movieId").cast(IntegerType())) \
            .withColumn("rating", col("rating").cast(FloatType()))
        
        # Clean data
        ratings_df = ratings_df.where(
            col("userId").isNotNull() & 
            col("movieId").isNotNull() & 
            col("rating").isNotNull()
        ).dropDuplicates(["userId", "movieId"])
        
        ratings_df.cache()
        
        return ratings_df
    
    except Exception as e:
        st.error(f"Error loading ratings data: {str(e)}")
        st.info("Please ensure 'ratings.csv' exists in the current directory.")
        return None


@st.cache_resource
def load_movies_data():
    """Load movies data from CSV (optional)"""
    try:
        spark = create_spark_session()
        
        if not os.path.exists("./movies.csv"):
            return None
        
        movies_df = spark.read.csv(
            "./movies.csv",
            header=True,
            inferSchema=True
        )
        
        movies_df = movies_df.select(
            col("movieId").cast(IntegerType()),
            col("title")
        )
        
        movies_df.cache()
        
        return movies_df
    
    except Exception as e:
        st.warning(f"Could not load movies.csv: {str(e)}")
        return None


@st.cache_resource
def calculate_user_averages(_ratings_df):
    """Calculate average ratings per user"""
    user_averages_df = _ratings_df.groupBy("userId") \
        .agg(
            avg("rating").alias("avg_rating"),
            count("rating").alias("num_ratings")
        ) \
        .withColumn("avg_rating", spark_round(col("avg_rating"), 3))
    
    user_averages_df.cache()
    
    return user_averages_df


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_star_rating(rating: float) -> str:
    """Convert numeric rating to star display"""
    full_stars = int(rating)
    half_star = 1 if (rating - full_stars) >= 0.5 else 0
    empty_stars = 5 - full_stars - half_star
    
    return "⭐" * full_stars + "½⭐" * half_star + "☆" * empty_stars


def get_rating_type(avg_rating: float) -> str:
    """Classify user rating behavior"""
    if avg_rating >= 4.0:
        return "Generous 🌟"
    elif avg_rating <= 2.5:
        return "Critical 🔍"
    else:
        return "Balanced ⚖️"


def get_similarity_label(similarity: float) -> Tuple[str, str]:
    """Get similarity label and CSS class"""
    if similarity >= 0.8:
        return "Very Similar 🎯", "similarity-very-high"
    elif similarity >= 0.6:
        return "Similar 👍", "similarity-high"
    elif similarity >= 0.4:
        return "Somewhat Similar 🤔", "similarity-medium"
    else:
        return "Different Taste 🎲", "similarity-low"


def get_confidence_label(num_contributors: int) -> Tuple[str, str]:
    """Get confidence label and CSS class"""
    if num_contributors >= 20:
        return "High Confidence ✅", "confidence-high"
    elif num_contributors >= 10:
        return "Medium Confidence ⚠️", "confidence-medium"
    else:
        return "Low Confidence ❓", "confidence-low"


def get_popular_movies(_ratings_df, _movies_df, n=30):
    """Get most popular movies"""
    popular = _ratings_df.groupBy("movieId") \
        .agg(
            count("rating").alias("num_ratings"),
            avg("rating").alias("avg_rating")
        ) \
        .orderBy(desc("num_ratings")) \
        .limit(n)
    
    if _movies_df is not None:
        popular = popular.join(_movies_df, "movieId", "left")
    
    return popular.toPandas()


def calculate_pearson_correlation(ratings_u: Dict[int, float], 
                                 ratings_v: Dict[int, float],
                                 avg_u: float,
                                 avg_v: float) -> Optional[float]:
    """
    Calculate Pearson correlation coefficient between two users
    
    Args:
        ratings_u: User u's ratings {movieId: rating}
        ratings_v: User v's ratings {movieId: rating}
        avg_u: User u's average rating
        avg_v: User v's average rating
        
    Returns:
        Pearson correlation coefficient or None if not enough common movies
    """
    # Find common movies
    common_movies = set(ratings_u.keys()) & set(ratings_v.keys())
    
    if len(common_movies) < 3:
        return None
    
    # Calculate deviations
    numerator = 0
    sum_sq_u = 0
    sum_sq_v = 0
    
    for movie_id in common_movies:
        dev_u = ratings_u[movie_id] - avg_u
        dev_v = ratings_v[movie_id] - avg_v
        
        numerator += dev_u * dev_v
        sum_sq_u += dev_u ** 2
        sum_sq_v += dev_v ** 2
    
    denominator = np.sqrt(sum_sq_u * sum_sq_v)
    
    if denominator == 0:
        return None
    
    return numerator / denominator


def find_similar_users(new_user_ratings: Dict[int, float],
                      ratings_df,
                      user_averages_df,
                      top_k: int = 10,
                      min_common: int = 3) -> List[Dict]:
    """
    Find similar users to a new user based on their ratings
    
    Args:
        new_user_ratings: New user's ratings {movieId: rating}
        ratings_df: Spark DataFrame with all ratings
        user_averages_df: Spark DataFrame with user averages
        top_k: Number of top similar users to return
        min_common: Minimum common movies required
        
    Returns:
        List of similar user dictionaries
    """
    new_user_avg = np.mean(list(new_user_ratings.values()))
    new_user_movies = set(new_user_ratings.keys())
    
    # Get all users with sufficient ratings - use where() instead of filter() to avoid conflicts
    candidate_users = user_averages_df.where(col("num_ratings") >= 10).toPandas()
    
    similar_users = []
    
    with st.spinner("Finding your movie soulmates..."):
        progress_bar = st.progress(0)
        total_users = len(candidate_users)
        
        for idx, user_row in candidate_users.iterrows():
            user_id = user_row['userId']
            user_avg = user_row['avg_rating']
            
            # Get user's ratings - use where() instead of filter()
            user_ratings_df = ratings_df.where(col("userId") == user_id).toPandas()
            user_ratings = dict(zip(user_ratings_df['movieId'], user_ratings_df['rating']))
            user_movies = set(user_ratings.keys())
            
            # Find common movies
            common_movies = new_user_movies & user_movies
            
            if len(common_movies) >= min_common:
                # Calculate Pearson correlation
                similarity = calculate_pearson_correlation(
                    new_user_ratings,
                    user_ratings,
                    new_user_avg,
                    user_avg
                )
                
                if similarity is not None and similarity > 0:
                    similar_users.append({
                        'userId': user_id,
                        'similarity': similarity,
                        'common_movies': len(common_movies),
                        'avg_rating': user_avg,
                        'num_ratings': user_row['num_ratings'],
                        'rating_type': get_rating_type(user_avg)
                    })
            
            # Update progress
            progress_bar.progress((idx + 1) / total_users)
        
        progress_bar.empty()
    
    # Sort by similarity and take top K
    similar_users.sort(key=lambda x: x['similarity'], reverse=True)
    
    return similar_users[:top_k]


def predict_rating_for_movie(new_user_ratings: Dict[int, float],
                             movie_id: int,
                             new_user_avg: float,
                             similar_users: List[Dict],
                             ratings_df,
                             user_averages_df,
                             top_contributors: int = 50) -> Optional[Dict]:
    """
    Predict rating for a specific movie using collaborative filtering
    
    Args:
        new_user_ratings: New user's ratings
        movie_id: Movie to predict rating for
        new_user_avg: New user's average rating
        similar_users: List of similar users
        ratings_df: Spark DataFrame with all ratings
        user_averages_df: Spark DataFrame with user averages
        top_contributors: Max number of similar users to use
        
    Returns:
        Dictionary with prediction details or None
    """
    # Get ratings for this movie from similar users
    similar_user_ids = [u['userId'] for u in similar_users[:top_contributors]]
    
    movie_ratings = ratings_df.where(
        (col("movieId") == movie_id) & 
        col("userId").isin(similar_user_ids)
    ).join(
        user_averages_df.select("userId", "avg_rating"),
        "userId"
    ).toPandas()
    
    if len(movie_ratings) == 0:
        return None
    
    # Apply prediction formula
    weighted_sum = 0
    similarity_sum = 0
    contributors = []
    
    for _, row in movie_ratings.iterrows():
        user_id = row['userId']
        rating = row['rating']
        user_avg = row['avg_rating']
        
        # Find similarity for this user
        similarity = next(
            (u['similarity'] for u in similar_users if u['userId'] == user_id),
            None
        )
        
        if similarity is not None:
            deviation = rating - user_avg
            weighted_sum += similarity * deviation
            similarity_sum += abs(similarity)
            
            contributors.append({
                'userId': user_id,
                'similarity': similarity,
                'rating': rating,
                'avg_rating': user_avg,
                'contribution': similarity * deviation
            })
    
    if similarity_sum == 0:
        return None
    
    # Calculate predicted rating
    predicted_rating = new_user_avg + (weighted_sum / similarity_sum)
    predicted_rating = max(0.5, min(5.0, predicted_rating))
    
    # Sort contributors by contribution
    contributors.sort(key=lambda x: abs(x['contribution']), reverse=True)
    
    return {
        'movieId': movie_id,
        'predicted_rating': round(predicted_rating, 2),
        'num_contributors': len(contributors),
        'contributors': contributors[:5]  # Top 5 contributors
    }


def generate_recommendations(new_user_ratings: Dict[int, float],
                            similar_users: List[Dict],
                            ratings_df,
                            user_averages_df,
                            movies_df,
                            top_n: int = 10) -> List[Dict]:
    """
    Generate top N movie recommendations for new user
    
    Args:
        new_user_ratings: New user's ratings
        similar_users: List of similar users
        ratings_df: Spark DataFrame with all ratings
        user_averages_df: Spark DataFrame with user averages
        movies_df: Spark DataFrame with movie titles
        top_n: Number of recommendations to generate
        
    Returns:
        List of recommendation dictionaries
    """
    new_user_avg = np.mean(list(new_user_ratings.values()))
    rated_movies = set(new_user_ratings.keys())
    
    # Get all movies rated by similar users
    similar_user_ids = [u['userId'] for u in similar_users]
    
    candidate_movies = ratings_df.where(
        col("userId").isin(similar_user_ids)
    ).select("movieId").distinct().toPandas()
    
    # Filter out already rated movies
    unseen_movies = [
        m for m in candidate_movies['movieId'] 
        if m not in rated_movies
    ]
    
    recommendations = []
    
    with st.spinner(f"Generating your top {top_n} recommendations..."):
        progress_bar = st.progress(0)
        total_movies = len(unseen_movies)
        
        for idx, movie_id in enumerate(unseen_movies):
            prediction = predict_rating_for_movie(
                new_user_ratings,
                movie_id,
                new_user_avg,
                similar_users,
                ratings_df,
                user_averages_df
            )
            
            if prediction:
                # Get movie title
                if movies_df is not None:
                    title_row = movies_df.where(col("movieId") == movie_id).first()
                    title = title_row['title'] if title_row else f"Movie {movie_id}"
                else:
                    title = f"Movie {movie_id}"
                
                prediction['title'] = title
                recommendations.append(prediction)
            
            # Update progress
            if idx % 10 == 0:
                progress_bar.progress(min((idx + 1) / total_movies, 1.0))
        
        progress_bar.empty()
    
    # Sort by predicted rating
    recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
    
    return recommendations[:top_n]


def get_user_profile(user_id: int, ratings_df, user_averages_df, movies_df) -> Dict:
    """Get detailed profile for a user"""
    # Get user stats
    user_stats = user_averages_df.where(col("userId") == user_id).first()
    
    if not user_stats:
        return None
    
    # Get user's ratings
    user_ratings_df = ratings_df.where(col("userId") == user_id)
    
    # Get top rated movies
    top_rated = user_ratings_df.orderBy(desc("rating")).limit(5)
    
    if movies_df is not None:
        top_rated = top_rated.join(movies_df, "movieId", "left")
    
    top_rated_movies = top_rated.toPandas()
    
    return {
        'userId': user_id,
        'avg_rating': user_stats['avg_rating'],
        'num_ratings': user_stats['num_ratings'],
        'rating_type': get_rating_type(user_stats['avg_rating']),
        'top_rated_movies': top_rated_movies
    }


def compare_users(user1_ratings: Dict[int, float],
                 user2_ratings: Dict[int, float],
                 avg1: float,
                 avg2: float) -> Dict:
    """Compare two users and return comparison metrics"""
    # Find common movies
    common_movies = set(user1_ratings.keys()) & set(user2_ratings.keys())
    
    if len(common_movies) == 0:
        return None
    
    # Calculate Pearson correlation
    similarity = calculate_pearson_correlation(
        user1_ratings,
        user2_ratings,
        avg1,
        avg2
    )
    
    # Calculate agreement
    agreements = 0
    disagreements = []
    both_loved = []
    both_disliked = []
    
    for movie_id in common_movies:
        rating1 = user1_ratings[movie_id]
        rating2 = user2_ratings[movie_id]
        diff = abs(rating1 - rating2)
        
        if diff <= 1.0:
            agreements += 1
        
        disagreements.append({
            'movieId': movie_id,
            'rating1': rating1,
            'rating2': rating2,
            'difference': diff
        })
        
        if rating1 >= 4.0 and rating2 >= 4.0:
            both_loved.append(movie_id)
        
        if rating1 <= 2.5 and rating2 <= 2.5:
            both_disliked.append(movie_id)
    
    # Sort disagreements
    disagreements.sort(key=lambda x: x['difference'], reverse=True)
    
    agreement_pct = (agreements / len(common_movies)) * 100 if common_movies else 0
    
    return {
        'similarity': similarity,
        'common_movies': len(common_movies),
        'agreement_pct': agreement_pct,
        'both_loved': both_loved,
        'both_disliked': both_disliked,
        'biggest_disagreements': disagreements[:10]
    }


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'user_name' not in st.session_state:
    st.session_state.user_name = ""

if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {}

if 'new_user_avg' not in st.session_state:
    st.session_state.new_user_avg = 0.0

if 'similar_users' not in st.session_state:
    st.session_state.similar_users = []

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []

if 'selected_comparison_user' not in st.session_state:
    st.session_state.selected_comparison_user = None

if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "🎯 Get My Recommendations"

# ============================================================================
# LOAD DATA
# ============================================================================

ratings_df = load_ratings_data()
movies_df = load_movies_data()

if ratings_df is None:
    st.error("❌ Failed to load ratings data. Please check that ratings.csv exists.")
    st.stop()

user_averages_df = calculate_user_averages(ratings_df)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("# 🎬 Smart Movie Recommender")
    st.markdown("### Find movies you'll love based on your taste and similar users")
    st.markdown("---")
    
    # Navigation
    st.session_state.current_tab = st.radio(
        "Navigation",
        ["🎯 Get My Recommendations", "👥 Compare with Users", "🔍 Explore Similar Users"],
        index=["🎯 Get My Recommendations", "👥 Compare with Users", "🔍 Explore Similar Users"].index(st.session_state.current_tab)
    )
    
    st.markdown("---")
    
    # Stats
    total_users = user_averages_df.count()
    total_movies = ratings_df.select("movieId").distinct().count()
    total_ratings = ratings_df.count()
    
    st.markdown("### 📊 Dataset Statistics")
    st.metric("Total Users", f"{total_users:,}")
    st.metric("Total Movies", f"{total_movies:,}")
    st.metric("Total Ratings", f"{total_ratings:,}")
    
    st.markdown("---")
    st.markdown("### 🎓 How It Works")
    st.markdown("""
    1. **Rate Movies**: Rate at least 5 movies
    2. **Find Similar Users**: We find users with similar taste
    3. **Get Recommendations**: Predict what you'll love
    4. **Explore**: Compare with other users
    """)

# ============================================================================
# TAB 1: GET MY RECOMMENDATIONS
# ============================================================================

if st.session_state.current_tab == "🎯 Get My Recommendations":
    st.title("🎯 Get Personalized Movie Recommendations")
    st.markdown("Rate some movies to get started. We'll find users with similar taste and recommend movies you'll love!")
    st.markdown("---")
    
    # USER INPUT SECTION
    st.markdown("## 👤 Tell Us About Your Movie Taste")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_name = st.text_input(
            "What's your name?",
            value=st.session_state.user_name,
            placeholder="Enter your name..."
        )
        st.session_state.user_name = user_name
    
    with col2:
        num_rated = len(st.session_state.user_ratings)
        progress = min(num_rated / 5, 1.0)
        st.metric("Movies Rated", f"{num_rated}/5 minimum")
        st.progress(progress)
    
    st.markdown("---")
    
    # Get popular movies
    popular_movies = get_popular_movies(ratings_df, movies_df, n=30)
    
    st.markdown("### 🎬 Rate These Popular Movies")
    st.markdown("Use the sliders to rate movies you've seen (0.5 = worst, 5.0 = best)")
    
    # Display movies in 3 columns
    cols_per_row = 3
    for i in range(0, len(popular_movies), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, column in enumerate(cols):
            if i + j < len(popular_movies):
                movie = popular_movies.iloc[i + j]
                movie_id = movie['movieId']
                
                with column:
                    st.markdown(f"**{movie['title'] if 'title' in movie else f'Movie {movie_id}'}**")
                    st.markdown(
                        f"<small style='color: gray;'>Avg: {movie['avg_rating']:.1f} ⭐ "
                        f"({movie['num_ratings']:,} ratings)</small>",
                        unsafe_allow_html=True
                    )
                    
                    current_rating = st.session_state.user_ratings.get(movie_id, 0.0)
                    
                    rating = st.slider(
                        "Your rating",
                        min_value=0.0,
                        max_value=5.0,
                        value=current_rating,
                        step=0.5,
                        key=f"rating_{movie_id}",
                        label_visibility="collapsed"
                    )
                    
                    if rating > 0:
                        st.session_state.user_ratings[movie_id] = rating
                        st.markdown(f"<div class='star-rating'>{get_star_rating(rating)}</div>", unsafe_allow_html=True)
                    elif movie_id in st.session_state.user_ratings:
                        del st.session_state.user_ratings[movie_id]
    
    # Clear ratings button
    if st.button("🗑️ Clear All Ratings"):
        st.session_state.user_ratings = {}
        st.session_state.similar_users = []
        st.session_state.recommendations = []
        st.rerun()
    
    st.markdown("---")
    
    # FIND SIMILAR USERS
    if len(st.session_state.user_ratings) >= 5:
        st.markdown("## 🔍 Find Your Movie Soulmates")
        
        if st.button("🎯 Find My Movie Soulmates", type="primary"):
            st.session_state.similar_users = find_similar_users(
                st.session_state.user_ratings,
                ratings_df,
                user_averages_df,
                top_k=10
            )
            
            st.session_state.new_user_avg = np.mean(list(st.session_state.user_ratings.values()))
            
            st.success(f"✅ Found {len(st.session_state.similar_users)} users with similar taste!")
        
        # Display similar users
        if st.session_state.similar_users:
            st.markdown("### 👥 Users with Similar Taste")
            
            for i in range(0, len(st.session_state.similar_users), 3):
                cols = st.columns(3)
                
                for j, column in enumerate(cols):
                    if i + j < len(st.session_state.similar_users):
                        user = st.session_state.similar_users[i + j]
                        
                        with column:
                            similarity_label, similarity_class = get_similarity_label(user['similarity'])
                            
                            st.markdown(f"""
                                <div class='movie-card'>
                                    <h4>User {user['userId']}</h4>
                                    <span class='similarity-badge {similarity_class}'>
                                        {similarity_label}
                                    </span>
                                    <p><strong>Similarity:</strong> {user['similarity']:.3f}</p>
                                    <p><strong>Profile:</strong></p>
                                    <ul>
                                        <li>Avg Rating: {user['avg_rating']:.2f} ⭐</li>
                                        <li>Rating Type: {user['rating_type']}</li>
                                        <li>Movies Rated: {user['num_ratings']}</li>
                                        <li>Common Movies: {user['common_movies']}</li>
                                    </ul>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            if st.button(f"👁️ View Details", key=f"view_{user['userId']}"):
                                st.session_state.selected_comparison_user = user['userId']
                                st.session_state.current_tab = "👥 Compare with Users"
                                st.rerun()
            
            st.markdown("---")
            
            # GENERATE RECOMMENDATIONS
            st.markdown("## 🎬 Get Your Top Recommendations")
            
            if st.button("🌟 Get My Top 10 Recommendations", type="primary"):
                st.session_state.recommendations = generate_recommendations(
                    st.session_state.user_ratings,
                    st.session_state.similar_users,
                    ratings_df,
                    user_averages_df,
                    movies_df,
                    top_n=10
                )
                
                st.success("✅ Generated your personalized recommendations!")
            
            # Display recommendations
            if st.session_state.recommendations:
                st.markdown("### 🏆 Your Top 10 Movie Recommendations")
                
                for idx, rec in enumerate(st.session_state.recommendations, 1):
                    confidence_label, confidence_class = get_confidence_label(rec['num_contributors'])
                    
                    col1, col2 = st.columns([1, 4])
                    
                    with col1:
                        st.markdown(f"""
                            <div class='rank-badge'>
                                #{idx}
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                            <div class='movie-card'>
                                <h3>{rec['title']}</h3>
                                <p><strong>Predicted Rating:</strong> {get_star_rating(rec['predicted_rating'])} ({rec['predicted_rating']:.2f})</p>
                                <span class='{confidence_class}'>{confidence_label}</span>
                                <p><small>Based on {rec['num_contributors']} similar users</small></p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("🤔 Why this recommendation?"):
                            st.markdown("**Top Contributing Users:**")
                            
                            contrib_data = []
                            for contrib in rec['contributors']:
                                contrib_data.append({
                                    'User ID': contrib['userId'],
                                    'Similarity': f"{contrib['similarity']:.3f}",
                                    'Their Rating': get_star_rating(contrib['rating']),
                                    'Their Avg': f"{contrib['avg_rating']:.2f}",
                                    'Contribution': f"{contrib['contribution']:.3f}"
                                })
                            
                            st.dataframe(pd.DataFrame(contrib_data), use_container_width=True)
                            
                            st.markdown(f"""
                                **Calculation:**
                                - Your average rating: {st.session_state.new_user_avg:.2f}
                                - Weighted adjustment from similar users: {rec['predicted_rating'] - st.session_state.new_user_avg:.2f}
                                - **Predicted rating: {rec['predicted_rating']:.2f}**
                            """)
    else:
        st.info(f"ℹ️ Please rate at least 5 movies to get recommendations (currently {len(st.session_state.user_ratings)} rated)")

# ============================================================================
# TAB 2: COMPARE WITH USERS
# ============================================================================

elif st.session_state.current_tab == "👥 Compare with Users":
    st.title("👥 Compare Your Taste with Other Users")
    st.markdown("See how your movie preferences align with other users in the system")
    st.markdown("---")
    
    # User selection
    if st.session_state.selected_comparison_user is None and len(st.session_state.user_ratings) >= 5:
        # Get users with sufficient ratings
        all_users = user_averages_df.where(col("num_ratings") >= 10).toPandas()
        
        user_options = [
            f"User {row['userId']} ({row['num_ratings']} ratings, avg: {row['avg_rating']:.2f})"
            for _, row in all_users.iterrows()
        ]
        
        selected = st.selectbox(
            "Select a user to compare with:",
            options=range(len(user_options)),
            format_func=lambda x: user_options[x]
        )
        
        if st.button("Compare"):
            st.session_state.selected_comparison_user = all_users.iloc[selected]['userId']
            st.rerun()
    
    elif st.session_state.selected_comparison_user is not None and len(st.session_state.user_ratings) >= 5:
        comparison_user_id = st.session_state.selected_comparison_user
        
        # Get comparison user's profile
        comparison_profile = get_user_profile(
            comparison_user_id,
            ratings_df,
            user_averages_df,
            movies_df
        )
        
        # Get comparison user's ratings
        comparison_ratings_df = ratings_df.where(col("userId") == comparison_user_id).toPandas()
        comparison_ratings = dict(zip(comparison_ratings_df['movieId'], comparison_ratings_df['rating']))
        
        # Calculate comparison metrics
        comparison_metrics = compare_users(
            st.session_state.user_ratings,
            comparison_ratings,
            st.session_state.new_user_avg,
            comparison_profile['avg_rating']
        )
        
        if comparison_metrics:
            # PROFILE COMPARISON
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                    <div class='profile-section'>
                        <h3>👤 Your Profile</h3>
                        <p><strong>Name:</strong> {st.session_state.user_name or 'You'}</p>
                        <p><strong>Movies Rated:</strong> {len(st.session_state.user_ratings)}</p>
                        <p><strong>Average Rating:</strong> {st.session_state.new_user_avg:.2f} ⭐</p>
                        <p><strong>Rating Type:</strong> {get_rating_type(st.session_state.new_user_avg)}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**Your Top Rated Movies:**")
                sorted_ratings = sorted(
                    st.session_state.user_ratings.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                for movie_id, rating in sorted_ratings:
                    if movies_df is not None:
                        title_row = movies_df.where(col("movieId") == movie_id).first()
                        title = title_row['title'] if title_row else f"Movie {movie_id}"
                    else:
                        title = f"Movie {movie_id}"
                    
                    st.markdown(f"- {title}: {get_star_rating(rating)} ({rating:.1f})")
            
            with col2:
                st.markdown(f"""
                    <div class='profile-section'>
                        <h3>👤 User {comparison_user_id} Profile</h3>
                        <p><strong>Movies Rated:</strong> {comparison_profile['num_ratings']}</p>
                        <p><strong>Average Rating:</strong> {comparison_profile['avg_rating']:.2f} ⭐</p>
                        <p><strong>Rating Type:</strong> {comparison_profile['rating_type']}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**Their Top Rated Movies:**")
                for _, movie in comparison_profile['top_rated_movies'].iterrows():
                    title = movie['title'] if 'title' in movie else f"Movie {movie['movieId']}"
                    st.markdown(f"- {title}: {get_star_rating(movie['rating'])} ({movie['rating']:.1f})")
            
            st.markdown("---")
            
            # SIMILARITY ANALYSIS
            st.markdown("## 📊 Similarity Analysis")
            
            # Check if similarity is valid
            if comparison_metrics['similarity'] is not None:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Pearson Correlation",
                        f"{comparison_metrics['similarity']:.3f}",
                        help="Ranges from -1 (opposite taste) to +1 (identical taste)"
                    )
                
                with col2:
                    st.metric(
                        "Common Movies",
                        comparison_metrics['common_movies']
                    )
                
                with col3:
                    st.metric(
                        "Agreement Rate",
                        f"{comparison_metrics['agreement_pct']:.1f}%",
                        help="% of movies where ratings differ by ≤1 star"
                    )
                
                # Interpretation
                similarity_label, _ = get_similarity_label(comparison_metrics['similarity'])
                st.info(f"**Interpretation:** {similarity_label}")
            else:
                st.warning("⚠️ Cannot calculate similarity - not enough data or identical ratings for all movies")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Common Movies",
                        comparison_metrics['common_movies']
                    )
                
                with col2:
                    st.metric(
                        "Agreement Rate",
                        f"{comparison_metrics['agreement_pct']:.1f}%",
                        help="% of movies where ratings differ by ≤1 star"
                    )
            
            # Scatter plot
            common_movies = set(st.session_state.user_ratings.keys()) & set(comparison_ratings.keys())
            
            if len(common_movies) > 0:
                scatter_data = []
                for movie_id in common_movies:
                    if movies_df is not None:
                        title_row = movies_df.where(col("movieId") == movie_id).first()
                        title = title_row['title'] if title_row else f"Movie {movie_id}"
                    else:
                        title = f"Movie {movie_id}"
                    
                    scatter_data.append({
                        'Movie': title,
                        'Your Rating': st.session_state.user_ratings[movie_id],
                        'Their Rating': comparison_ratings[movie_id]
                    })
                
                scatter_df = pd.DataFrame(scatter_data)
                
                fig = px.scatter(
                    scatter_df,
                    x='Your Rating',
                    y='Their Rating',
                    hover_data=['Movie'],
                    title="Rating Comparison for Common Movies",
                    trendline="ols"
                )
                
                fig.add_shape(
                    type="line",
                    x0=0.5, y0=0.5,
                    x1=5.0, y1=5.0,
                    line=dict(color="red", dash="dash")
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # RECOMMENDATIONS
            st.markdown("## 🎬 Movie Recommendations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### Movies User {comparison_user_id} Loved")
                st.markdown("(That you haven't seen yet)")
                
                their_favorites = [
                    (movie_id, rating)
                    for movie_id, rating in comparison_ratings.items()
                    if rating >= 4.0 and movie_id not in st.session_state.user_ratings
                ]
                
                their_favorites.sort(key=lambda x: x[1], reverse=True)
                
                for movie_id, rating in their_favorites[:10]:
                    if movies_df is not None:
                        title_row = movies_df.where(col("movieId") == movie_id).first()
                        title = title_row['title'] if title_row else f"Movie {movie_id}"
                    else:
                        title = f"Movie {movie_id}"
                    
                    st.markdown(f"- {title}: {get_star_rating(rating)} ({rating:.1f})")
            
            with col2:
                st.markdown(f"### Your Favorites")
                st.markdown(f"(That User {comparison_user_id} hasn't seen)")
                
                your_favorites = [
                    (movie_id, rating)
                    for movie_id, rating in st.session_state.user_ratings.items()
                    if rating >= 4.0 and movie_id not in comparison_ratings
                ]
                
                your_favorites.sort(key=lambda x: x[1], reverse=True)
                
                for movie_id, rating in your_favorites[:10]:
                    if movies_df is not None:
                        title_row = movies_df.where(col("movieId") == movie_id).first()
                        title = title_row['title'] if title_row else f"Movie {movie_id}"
                    else:
                        title = f"Movie {movie_id}"
                    
                    st.markdown(f"- {title}: {get_star_rating(rating)} ({rating:.1f})")
            
            st.markdown("---")
            
            # AGREEMENT/DISAGREEMENT
            st.markdown("## 🤝 Agreement & Disagreement")
            
            tab1, tab2, tab3 = st.tabs(["✅ Both Loved", "❌ Both Disliked", "⚖️ Biggest Disagreements"])
            
            with tab1:
                st.markdown("**Movies you both rated ≥4.0 stars:**")
                
                for movie_id in comparison_metrics['both_loved']:
                    if movies_df is not None:
                        title_row = movies_df.where(col("movieId") == movie_id).first()
                        title = title_row['title'] if title_row else f"Movie {movie_id}"
                    else:
                        title = f"Movie {movie_id}"
                    
                    your_rating = st.session_state.user_ratings[movie_id]
                    their_rating = comparison_ratings[movie_id]
                    
                    st.markdown(f"- {title}: You {get_star_rating(your_rating)}, Them {get_star_rating(their_rating)}")
            
            with tab2:
                st.markdown("**Movies you both rated ≤2.5 stars:**")
                
                for movie_id in comparison_metrics['both_disliked']:
                    if movies_df is not None:
                        title_row = movies_df.where(col("movieId") == movie_id).first()
                        title = title_row['title'] if title_row else f"Movie {movie_id}"
                    else:
                        title = f"Movie {movie_id}"
                    
                    your_rating = st.session_state.user_ratings[movie_id]
                    their_rating = comparison_ratings[movie_id]
                    
                    st.markdown(f"- {title}: You {get_star_rating(your_rating)}, Them {get_star_rating(their_rating)}")
            
            with tab3:
                st.markdown("**Movies with biggest rating differences:**")
                
                for disagree in comparison_metrics['biggest_disagreements'][:10]:
                    movie_id = disagree['movieId']
                    
                    if movies_df is not None:
                        title_row = movies_df.where(col("movieId") == movie_id).first()
                        title = title_row['title'] if title_row else f"Movie {movie_id}"
                    else:
                        title = f"Movie {movie_id}"
                    
                    st.markdown(
                        f"- {title}: You {get_star_rating(disagree['rating1'])}, "
                        f"Them {get_star_rating(disagree['rating2'])} "
                        f"(Diff: {disagree['difference']:.1f})"
                    )
            
            # Back button
            if st.button("🔙 Back to User Selection"):
                st.session_state.selected_comparison_user = None
                st.rerun()
        
        else:
            st.error("Not enough common movies to compare!")
            if st.button("🔙 Select Different User"):
                st.session_state.selected_comparison_user = None
                st.rerun()
    
    else:
        st.info("ℹ️ Please rate at least 5 movies in Tab 1 to compare with other users")

# ============================================================================
# TAB 3: EXPLORE SIMILAR USERS
# ============================================================================

elif st.session_state.current_tab == "🔍 Explore Similar Users":
    st.title("🔍 Explore Similar Users")
    st.markdown("Discover and filter users with similar movie taste")
    st.markdown("---")
    
    if len(st.session_state.user_ratings) >= 5 and st.session_state.similar_users:
        # Filters in sidebar
        with st.sidebar:
            st.markdown("### 🎛️ Filters")
            
            min_similarity = st.slider(
                "Minimum Similarity",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1
            )
            
            min_common = st.number_input(
                "Minimum Common Movies",
                min_value=1,
                max_value=50,
                value=5
            )
            
            rating_types = st.multiselect(
                "Rating Type",
                ["Generous 🌟", "Critical 🔍", "Balanced ⚖️"],
                default=["Generous 🌟", "Critical 🔍", "Balanced ⚖️"]
            )
        
        # Filter users
        filtered_users = [
            u for u in st.session_state.similar_users
            if u['similarity'] >= min_similarity
            and u['common_movies'] >= min_common
            and u['rating_type'] in rating_types
        ]
        
        # Sort options
        sort_by = st.selectbox(
            "Sort by:",
            ["Similarity (Highest)", "Common Movies (Most)", "Rating Difference (Smallest)"]
        )
        
        if sort_by == "Similarity (Highest)":
            filtered_users.sort(key=lambda x: x['similarity'], reverse=True)
        elif sort_by == "Common Movies (Most)":
            filtered_users.sort(key=lambda x: x['common_movies'], reverse=True)
        else:
            filtered_users.sort(key=lambda x: abs(x['avg_rating'] - st.session_state.new_user_avg))
        
        st.markdown(f"### Found {len(filtered_users)} Similar Users")
        
        # Display users in grid
        for i in range(0, len(filtered_users), 4):
            cols = st.columns(4)
            
            for j, column in enumerate(cols):
                if i + j < len(filtered_users):
                    user = filtered_users[i + j]
                    
                    with column:
                        similarity_label, similarity_class = get_similarity_label(user['similarity'])
                        
                        st.markdown(f"""
                            <div class='movie-card'>
                                <h4>User {user['userId']}</h4>
                                <span class='similarity-badge {similarity_class}'>
                                    {user['similarity']:.3f}
                                </span>
                                <p><small>{similarity_label}</small></p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.progress(user['similarity'])
                        
                        st.markdown(f"""
                            - **Common:** {user['common_movies']} movies
                            - **Avg Rating:** {user['avg_rating']:.2f}
                            - **Type:** {user['rating_type']}
                        """)
                        
                        if st.button("Compare", key=f"compare_{user['userId']}"):
                            st.session_state.selected_comparison_user = user['userId']
                            st.session_state.current_tab = "👥 Compare with Users"
                            st.rerun()
        
        st.markdown("---")
        
        # Similarity distribution
        st.markdown("### 📊 Similarity Distribution")
        
        similarity_scores = [u['similarity'] for u in st.session_state.similar_users]
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=similarity_scores,
            nbinsx=20,
            name="Similarity Distribution"
        ))
        
        fig.update_layout(
            title="Distribution of Similarity Scores",
            xaxis_title="Similarity Score",
            yaxis_title="Number of Users",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif len(st.session_state.user_ratings) >= 5:
        st.info("ℹ️ Please find similar users in Tab 1 first!")
    else:
        st.info("ℹ️ Please rate at least 5 movies in Tab 1 to explore similar users")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🎬 Smart Movie Recommender | Built with Streamlit & PySpark</p>
        <p>Powered by Collaborative Filtering & Pearson Correlation</p>
    </div>
""", unsafe_allow_html=True)