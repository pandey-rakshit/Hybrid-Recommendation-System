# src/modeling/clustering.py
import pandas as pd
from sklearn.cluster import KMeans

def fit_kmeans(matrix, n_clusters=10, random_state=42):
    """
    Fit KMeans on the given matrix (text + numeric) and return model & labels.
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = km.fit_predict(matrix)
    return km, labels

def cluster_recommender(
    df: pd.DataFrame,
    labels,
    target_item: str,
    title_col: str = "title",
    label_col: str = "cluster_label",
    top_k: int = 5
):
    """
    Given a df with a 'cluster_label' column, recommend items
    in the same cluster as target_item.
    """
    # Map titles to indices
    idx_map = pd.Series(df.index, index=df[title_col]).drop_duplicates()
    if target_item not in idx_map:
        return []

    item_idx = idx_map[target_item]
    cluster_id = df.loc[item_idx, label_col]

    cluster_subset = df[df[label_col] == cluster_id]
    cluster_subset = cluster_subset[cluster_subset.index != item_idx]
    return cluster_subset[title_col].head(top_k).values
