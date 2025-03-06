import os

# Root directory of the project
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MOVIES_PATH = os.path.join(DATA_DIR, "movies.csv")
ANIME_PATH = os.path.join(DATA_DIR, "anime.csv")
