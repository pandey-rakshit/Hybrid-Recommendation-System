# src/feature_engineering/vectorization.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def build_tfidf_matrix(
    df: pd.DataFrame,
    text_col: str = "combined_text"
):
    """
    Builds a TF-IDF matrix from a specified text column in the DataFrame.
    Returns the fitted TfidfVectorizer and the resulting sparse matrix.
    """
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(df[text_col].fillna(""))
    return vectorizer, matrix

def build_count_matrix(
    df: pd.DataFrame,
    text_col: str = "combined_text"
):
    """
    Builds a count-based matrix from a specified text column in the DataFrame.
    Returns the fitted CountVectorizer and the resulting sparse matrix.
    """
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(df[text_col].fillna(""))
    return vectorizer, matrix
