import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from collections import Counter
from emotion_detection import detect_emotions
from dream_symbols import identify_symbols

try:
    nltk.data.find('vader_lexicon')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.update(['dream', 'dreams', 'dreamt', 'dreaming', 'saw', 'felt', 'went', 'came'])

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha() and token not in STOP_WORDS]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

def extract_keywords(text):
    doc = nlp(text)
    keywords = []
    for token in doc:
        if token.pos_ in ["NOUN", "VERB", "ADJ"] and not token.is_stop:
            lemma = token.lemma_.lower()
            if len(lemma) > 2 and lemma not in STOP_WORDS:
                keywords.append(lemma)
    
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "LOC", "GPE", "FAC"]:
            keywords.append(ent.text.lower())
    
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) > 1:
            keywords.append(chunk.text.lower())
    
    keyword_counts = Counter(keywords)
    sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
    
    return list(dict.fromkeys([k[0] for k in sorted_keywords]))

def analyze_sentiment(text):
    scores = sia.polarity_scores(text)
    
    doc = nlp(text)
    intensity_words = [token.text for token in doc if token.pos_ == "ADV" and token.dep_ == "advmod"]
    intensity_score = len(intensity_words) / len(doc) if len(doc) > 0 else 0
    scores['intensity'] = intensity_score
    
    emotions = detect_emotions(text)
    scores['emotions'] = emotions['emotions_str']
    scores['primary_emotion'] = emotions['primary_emotions'][0] if emotions['primary_emotions'] else 'neutral'
    
    return scores

def extract_dream_themes(text):
    keywords = extract_keywords(text)
    
    symbols = identify_symbols(text)
    symbol_categories = {}
    
    for symbol, data in symbols.items():
        category = data['category']
        if category in symbol_categories:
            symbol_categories[category] += 1
        else:
            symbol_categories[category] = 1
    
    themes = {
        'adventure': ['travel', 'journey', 'explore', 'discover', 'quest', 'adventure'],
        'conflict': ['fight', 'argue', 'battle', 'struggle', 'conflict', 'war'],
        'escape': ['run', 'flee', 'escape', 'avoid', 'hide', 'chase'],
        'loss': ['lose', 'lost', 'missing', 'gone', 'disappear', 'search'],
        'transformation': ['change', 'transform', 'grow', 'evolve', 'become', 'metamorphosis'],
        'relationships': ['friend', 'family', 'love', 'partner', 'relationship', 'connection'],
        'fear': ['afraid', 'fear', 'terror', 'scary', 'horror', 'nightmare'],
        'success': ['achieve', 'accomplish', 'win', 'success', 'victory', 'triumph']
    }
    
    matched_themes = []
    for theme, related_words in themes.items():
        if any(keyword in related_words for keyword in keywords):
            matched_themes.append(theme)
    
    if symbol_categories:
        top_category = max(symbol_categories.items(), key=lambda x: x[1])[0]
        if top_category == 'nature' and 'nature' not in matched_themes:
            matched_themes.append('nature')
        elif top_category == 'people' and 'relationships' not in matched_themes:
            matched_themes.append('relationships')
    
    if not matched_themes and keywords:
        matched_themes = keywords[:3]
    
    return matched_themes