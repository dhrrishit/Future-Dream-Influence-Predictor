import spacy
from nltk.sentiment import SentimentIntensityAnalyzer

nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()

def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.lemma_ for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ"]]
    return list(set(keywords))

def analyze_sentiment(text):
    scores = sia.polarity_scores(text)
    return scores