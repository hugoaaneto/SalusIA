import string
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("stopwords")

nlp = spacy.load("pt_core_news_sm")


def preprocess(
    text,
    to_lower=False,
    remove_punct=False,
    remove_stop=False,
    apply_stem=False,
    apply_lemma=False,
):
    if to_lower:
        text = text.lower()

    if remove_punct:
        text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = word_tokenize(text, language="portuguese")

    if remove_stop:
        stop_words = set(stopwords.words("portuguese"))
        tokens = [word for word in tokens if word not in stop_words]

    if apply_stem:
        stemmer = RSLPStemmer()
        tokens = [stemmer.stem(word) for word in tokens]

    if apply_lemma:
        doc = nlp(" ".join(tokens))
        tokens = [token.lemma_ for token in doc]

    return " ".join(tokens)
