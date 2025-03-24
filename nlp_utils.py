import os
import sys
try:
    import spacy
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import nltk
    from collections import Counter
except ImportError as e:
    print(f"Error: Required module not found: {e}")
    print("Please install required dependencies using: pip install -r requirements.txt")
    sys.exit(1)

# Import application-specific modules
try:
    from emotion_detection import detect_emotions
    from dream_symbols import identify_symbols
except ImportError as e:
    print(f"Error: Application module not found: {e}")
    print("Please ensure all application files are in the same directory.")
    sys.exit(1)

# Initialize NLTK resources
def initialize_nltk():
    required_resources = [
        'vader_lexicon',
        'tokenizers/punkt',
        'corpora/stopwords',
        'corpora/wordnet'
    ]
    
    for resource in required_resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            print(f"Downloading NLTK resource: {resource}")
            try:
                nltk.download(resource.split('/')[-1], quiet=True)
                print(f"Successfully downloaded {resource.split('/')[-1]}")
            except Exception as e:
                print(f"Error downloading {resource}: {e}")
                print("Some functionality might be limited.")

# Initialize spaCy
def initialize_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy English model...")
        try:
            os.system("python -m spacy download en_core_web_sm")
            return spacy.load("en_core_web_sm")
        except Exception as e:
            print(f"Error downloading spaCy model: {e}")
            print("Text processing capabilities will be limited.")
            return None

# Initialize resources
initialize_nltk()
try:
    nlp = initialize_spacy()
    sia = SentimentIntensityAnalyzer()
    lemmatizer = WordNetLemmatizer()
except Exception as e:
    print(f"Error initializing NLP components: {e}")
    print("Some functionality may be limited.")
    nlp = None
    sia = None
    lemmatizer = None

# Define stop words with dream-specific additions
try:
    STOP_WORDS = set(stopwords.words('english'))
    STOP_WORDS.update(['dream', 'dreams', 'dreamt', 'dreaming', 'saw', 'felt', 'went', 'came', 'seemed', 'appeared'])
except Exception as e:
    print(f"Error loading stop words: {e}")
    STOP_WORDS = set(['dream', 'dreams', 'dreamt', 'dreaming', 'saw', 'felt', 'went', 'came'])

def preprocess_text(text):
    """Preprocess text by tokenizing, removing stopwords, and lemmatizing."""
    if not isinstance(text, str) or not text.strip():
        return []
    
    if not lemmatizer:
        return text.lower().split() if text else []
    
    try:
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalpha() and token not in STOP_WORDS]
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized_tokens
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return []

def extract_keywords(text):
    """Extract important keywords from text using spaCy."""
    if not nlp or not isinstance(text, str) or not text.strip():
        return ["analysis", "unavailable"]
    
    try:
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
        
        # Return unique keywords, limit to top 15
        unique_keywords = list(dict.fromkeys([k[0] for k in sorted_keywords]))
        return unique_keywords[:15] if unique_keywords else ["no", "keywords", "found"]
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return ["analysis", "error"]

def analyze_sentiment(text):
    """Analyze sentiment of text using VADER."""
    default_result = {"compound": 0, "pos": 0, "neu": 0, "neg": 0}
    
    if not sia or not isinstance(text, str) or not text.strip():
        return default_result
    
    try:
        scores = sia.polarity_scores(text)
        
        if nlp:
            doc = nlp(text)
            intensity_words = [token.text for token in doc if token.pos_ == "ADV" and token.dep_ == "advmod"]
            intensity_score = len(intensity_words) / len(doc) if len(doc) > 0 else 0
            scores['intensity'] = intensity_score
        else:
            scores['intensity'] = 0
        
        try:
            emotions = detect_emotions(text)
            scores['emotions'] = emotions['emotions_str']
            scores['primary_emotion'] = emotions['primary_emotions'][0] if emotions['primary_emotions'] else 'neutral'
        except Exception as e:
            print(f"Error detecting emotions: {e}")
            scores['emotions'] = "neutral"
            scores['primary_emotion'] = "neutral"
        
        return scores
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return default_result

def extract_dream_themes(text):
    """Extract high-level themes from dream text."""
    if not nlp or not isinstance(text, str) or not text.strip():
        return ["unknown"]
    
    try:
        keywords = extract_keywords(text)
        
        try:
            symbols = identify_symbols(text)
            symbol_categories = {}
            
            for symbol, data in symbols.items():
                category = data['category']
                if category in symbol_categories:
                    symbol_categories[category] += 1
                else:
                    symbol_categories[category] = 1
        except Exception as e:
            print(f"Error processing symbols: {e}")
            symbol_categories = {}
        
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
        
        return matched_themes if matched_themes else ["unclassified"]
    except Exception as e:
        print(f"Error extracting dream themes: {e}")
        return ["unknown"]