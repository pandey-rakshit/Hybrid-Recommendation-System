# src/modeling/content_based.py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import cosine_similarity
from typing import Set, Optional, Dict
from src.data_preprocessing.cleaning import clean_text

def build_user_vector(
    user_text: str,
    vectorizer,
    numeric_values: dict = None,
    numeric_cols: list = None,
    scaler=None
):
    """
    Builds a (text + numeric) user vector to match the shape of your combined_matrix.
    
    1) Vectorize user_text using the same vectorizer used for your main dataset.
    2) Create a numeric array in the *same order of columns* that were used to build your combined_matrix.
    3) Transform numeric array with the same scaler (if any).
    4) Horizontally stack text and numeric vectors into one final user vector.

    :param user_text: The text describing user preferences
    :param vectorizer: The fitted TfidfVectorizer used to build the combined_matrix
    :param numeric_values: A dictionary of {col_name: value} for numeric features
    :param numeric_cols: The list of numeric columns in the exact order used for combine_text_and_numeric
    :param scaler: The fitted StandardScaler used for scaling numeric columns
    :return: A sparse row vector (1 x N) matching the dimensionality of combined_matrix
    """
    if not user_text:
        user_text = ""
    text_vec = vectorizer.transform([user_text])

    # 2) If no numeric features are used in the main matrix, just return the text vector
    if not numeric_cols:
        return text_vec

    if numeric_values is None:
        numeric_values = {}

    # 3) Build numeric array in the same column order
    user_numeric_list = []
    for col in numeric_cols:
        # default to 0 if user didn't provide a value
        user_numeric_list.append(numeric_values.get(col, 0.0))
    user_numeric_arr = np.array(user_numeric_list).reshape(1, -1)

    # 4) Scale numeric data if a scaler exists
    if scaler is not None:
        user_numeric_arr = scaler.transform(user_numeric_arr)

    num_vec = csr_matrix(user_numeric_arr)

    # 5) Combine text + numeric into a single vector
    user_vector = hstack([text_vec, num_vec])
    return user_vector


def recommend_items(
    df: pd.DataFrame,
    combined_matrix: csr_matrix,
    user_vector: csr_matrix,
    top_k: int = 5,
    title_col: str = "title",
    fallback_col: str = "popularity",
    already_watched: Optional[Set[int]] = None
):
    """
    Compute cosine similarity of user_vector with combined_matrix, return top_k item titles.
    If user_vector results in low similarity or if we want a fallback, use fallback_col to pick popular items.
    """
    if already_watched is None:
        already_watched = set()

    sim_scores = cosine_similarity(user_vector, combined_matrix).flatten()
    sorted_idx = np.argsort(sim_scores)[::-1]

    recommendations = []
    for idx in sorted_idx:
        if idx not in already_watched:
            recommendations.append(idx)
        if len(recommendations) >= top_k:
            break

    # Fallback if not enough recs
    if len(recommendations) < top_k:
        not_watched_df = df[~df.index.isin(already_watched)]
        fallback_sorted = not_watched_df.sort_values(by=fallback_col, ascending=False)
        needed = top_k - len(recommendations)
        recommendations.extend(fallback_sorted.index[:needed].tolist())

    return df.loc[recommendations, title_col].values
