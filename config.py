"""
Configuration file for Movie Recommendation System
Contains all configurable parameters and paths
"""

# ============================================================================
# FILE PATHS
# ============================================================================

DATA_PATH = "./ratings.csv"
MOVIES_PATH = "./movies.csv"
OUTPUT_DIR = "./output"

# ============================================================================
# ALGORITHM PARAMETERS
# ============================================================================

# Target user for recommendations
TARGET_USER_ID = 10

# Number of similar users to consider
TOP_K_SIMILAR_USERS = 50

# Minimum number of common movies required for similarity calculation
MIN_COMMON_MOVIES = 3

# Number of recommendations to generate
TOP_N_RECOMMENDATIONS = 5

# ============================================================================
# SPARK CONFIGURATION
# ============================================================================

SPARK_CONFIG = {
    "spark.app.name": "MovieRecommendationPearson",
    "spark.master": "local[*]",
    "spark.driver.memory": "8g",
    "spark.executor.memory": "8g",
    "spark.sql.shuffle.partitions": "10",
    "spark.driver.host": "localhost",
    "spark.driver.bindAddress": "127.0.0.1",
    "spark.ui.enabled": "false",
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true"
}

# ============================================================================
# DISPLAY SETTINGS
# ============================================================================

# Number of rows to display in sample outputs
DISPLAY_ROWS = 20

# Decimal precision for similarity scores
SIMILARITY_PRECISION = 4

# Decimal precision for predicted ratings
RATING_PRECISION = 2