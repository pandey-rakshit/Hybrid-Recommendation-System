# src/data_preprocessing/cleaning.py
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from utils.constants import CLEAN_TEXT

stopword = CLEAN_TEXT['STOP_WORDS']

# Ensure nltk stopwords are downloaded in your environment
# nltk.download('stopwords')

stop_words = set(stopwords.words(stopword))

def clean_text(text: str) -> str:
    """
    Cleans a raw string by:
    - Lowercasing
    - Removing punctuation & digits
    - Stripping extra whitespace
    - Removing stopwords

    :param text: Input text (can be NaN or empty)
    :return: Cleaned string
    """
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)
