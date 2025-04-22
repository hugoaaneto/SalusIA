import string
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")


def preprocess(text, to_lower, remove_punct, remove_stop):
    if to_lower:
        text = text.lower()
    if remove_punct:
        text = text.translate(str.maketrans("", "", string.punctuation))
    if remove_stop:
        stop_words = set(stopwords.words("portuguese"))
        tokens = text.split()
        text = " ".join([word for word in tokens if word not in stop_words])
    return text
