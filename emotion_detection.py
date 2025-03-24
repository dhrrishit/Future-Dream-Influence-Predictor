import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sys
import os

# Initialize NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading required NLTK resources...")
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")
        print("Some functionality may be limited.")

# Initialize spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except (ImportError, OSError) as e:
    print(f"Error loading spaCy model: {e}")
    print("Attempting to download spaCy model...")
    try:
        import os
        os.system(f"{sys.executable} -m spacy download en_core_web_sm")
        try:
            nlp = spacy.load("en_core_web_sm")
            print("Successfully loaded spaCy model.")
        except Exception as e:
            print(f"Failed to load spaCy model after download: {e}")
            nlp = None
    except Exception as e:
        print(f"Failed to download spaCy model: {e}")
        print("Some functionality will be limited.")
        nlp = None

# Initialize lemmatizer and stop words
try:
    lemmatizer = WordNetLemmatizer()
    STOP_WORDS = set(stopwords.words('english'))
    # Add dream-specific stop words
    STOP_WORDS.update(['dream', 'dreams', 'dreamt', 'dreaming', 'saw', 'felt', 'went', 'came', 'seemed', 'appeared'])
except Exception as e:
    print(f"Error initializing NLP components: {e}")
    print("Some functionality may be limited.")
    lemmatizer = None
    STOP_WORDS = set()

# Emotion lexicon maps words to core emotions
EMOTION_LEXICON = {
    'happy': 'joy', 'joy': 'joy', 'delighted': 'joy', 'pleased': 'joy',
    'glad': 'joy', 'satisfied': 'joy', 'excited': 'joy', 'thrilled': 'joy',
    'ecstatic': 'joy', 'content': 'joy', 'cheerful': 'joy', 'merry': 'joy',
    'jubilant': 'joy', 'elated': 'joy', 'blissful': 'joy',
    
    'sad': 'sadness', 'unhappy': 'sadness', 'miserable': 'sadness', 'depressed': 'sadness',
    'gloomy': 'sadness', 'heartbroken': 'sadness', 'downcast': 'sadness', 'grief': 'sadness',
    'sorrow': 'sadness', 'woe': 'sadness', 'melancholy': 'sadness', 'despair': 'sadness',
    'dejected': 'sadness', 'regretful': 'sadness',
    
    'afraid': 'fear', 'fear': 'fear', 'scared': 'fear', 'frightened': 'fear',
    'terrified': 'fear', 'anxious': 'fear', 'worried': 'fear', 'panic': 'fear',
    'horror': 'fear', 'terror': 'fear', 'dread': 'fear', 'phobia': 'fear',
    'nightmare': 'fear', 'alarmed': 'fear',
    
    'angry': 'anger', 'mad': 'anger', 'furious': 'anger', 'outraged': 'anger',
    'enraged': 'anger', 'irritated': 'anger', 'annoyed': 'anger', 'rage': 'anger',
    'fury': 'anger', 'hostile': 'anger', 'bitter': 'anger', 'hatred': 'anger',
    'resentment': 'anger', 'indignant': 'anger',
    
    'surprised': 'surprise', 'amazed': 'surprise', 'astonished': 'surprise', 'shocked': 'surprise',
    'startled': 'surprise', 'stunned': 'surprise', 'unexpected': 'surprise', 'wonder': 'surprise',
    'awe': 'surprise', 'bewildered': 'surprise', 'dumbfounded': 'surprise',
    
    'disgusted': 'disgust', 'revolted': 'disgust', 'repulsed': 'disgust', 'sickened': 'disgust',
    'nauseous': 'disgust', 'loathing': 'disgust', 'abhorrence': 'disgust', 'aversion': 'disgust',
    'repugnance': 'disgust', 'revulsion': 'disgust',
    
    'love': 'love', 'adore': 'love', 'affection': 'love', 'fond': 'love',
    'caring': 'love', 'tenderness': 'love', 'compassion': 'love', 'warmth': 'love',
    'attachment': 'love', 'devotion': 'love', 'passion': 'love', 'infatuation': 'love',
    'desire': 'love', 'longing': 'love',
    
    'confused': 'confusion', 'puzzled': 'confusion', 'perplexed': 'confusion', 'baffled': 'confusion',
    'uncertain': 'confusion', 'unsure': 'confusion', 'disoriented': 'confusion', 'muddled': 'confusion',
    'bewildered': 'confusion', 'mystified': 'confusion',
    
    'peaceful': 'peace', 'calm': 'peace', 'tranquil': 'peace', 'serene': 'peace',
    'relaxed': 'peace', 'composed': 'peace', 'quiet': 'peace', 'still': 'peace',
    'harmony': 'peace', 'balance': 'peace', 'ease': 'peace', 'comfort': 'peace',
    
    'anticipate': 'anticipation', 'expect': 'anticipation', 'await': 'anticipation', 'hope': 'anticipation',
    'looking forward': 'anticipation', 'eager': 'anticipation', 'excited': 'anticipation'
}

# Intensity modifiers affect emotion strength
INTENSITY_MODIFIERS = {
    'very': 1.5, 'extremely': 2.0, 'slightly': 0.5, 'somewhat': 0.7, 'really': 1.5,
    'incredibly': 2.0, 'barely': 0.3, 'hardly': 0.3, 'absolutely': 2.0, 'completely': 1.8,
    'totally': 1.8, 'utterly': 1.8, 'quite': 1.2, 'rather': 1.1, 'almost': 0.8,
    'nearly': 0.9, 'so': 1.5, 'too': 1.3, 'intensely': 1.8, 'deeply': 1.7,
    'profoundly': 1.9, 'mildly': 0.6, 'moderately': 0.8, 'highly': 1.6
}

def preprocess_text(text):
    """Preprocess text by tokenizing, removing stopwords, and lemmatizing."""
    if not isinstance(text, str) or not text.strip() or lemmatizer is None:
        return []
    
    try:
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalpha() and token not in STOP_WORDS]
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized_tokens
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return []

def detect_emotions(text):
    """Detect emotions in text using the emotion lexicon."""
    # Default emotion results for error cases
    default_result = {
        'emotion_scores': {
            'joy': 0, 'sadness': 0, 'fear': 0, 'anger': 0, 'surprise': 0,
            'disgust': 0, 'love': 0, 'confusion': 0, 'peace': 0, 'anticipation': 0
        },
        'primary_emotions': [],
        'emotions_str': "neutral"
    }
    
    if not isinstance(text, str) or not text.strip():
        return default_result
    
    try:
        tokens = preprocess_text(text)
        
        emotion_counts = {
            'joy': 0, 'sadness': 0, 'fear': 0, 'anger': 0, 'surprise': 0,
            'disgust': 0, 'love': 0, 'confusion': 0, 'peace': 0, 'anticipation': 0
        }
        
        current_modifier = 1.0
        
        for i, token in enumerate(tokens):
            if token in INTENSITY_MODIFIERS:
                current_modifier = INTENSITY_MODIFIERS[token]
                continue
                
            if token in EMOTION_LEXICON:
                emotion = EMOTION_LEXICON[token]
                emotion_counts[emotion] += current_modifier
                current_modifier = 1.0
        
        total_emotions = sum(emotion_counts.values())
        if total_emotions > 0:
            emotion_scores = {emotion: count/total_emotions for emotion, count in emotion_counts.items()}
        else:
            emotion_scores = emotion_counts
        
        primary_emotions = [emotion for emotion, score in sorted(
            emotion_scores.items(), key=lambda x: x[1], reverse=True) 
            if score > 0][:3]  # Get top 3 emotions
        
        emotions_str = ", ".join(primary_emotions) if primary_emotions else "neutral"
        
        return {
            'emotion_scores': emotion_scores,
            'primary_emotions': primary_emotions,
            'emotions_str': emotions_str
        }
    except Exception as e:
        print(f"Error detecting emotions: {e}")
        return default_result

def analyze_emotion_patterns(dream_history):
    """Analyze patterns in emotions across dream history."""
    if not isinstance(dream_history, pd.DataFrame) or len(dream_history) < 3:
        return "Not enough dream data to analyze emotion patterns."
    
    if 'emotions' not in dream_history.columns:
        return "No emotion data available in dream history."
    
    try:
        all_emotions = []
        for emotions_str in dream_history['emotions']:
            if isinstance(emotions_str, str) and emotions_str != "neutral":
                emotions = [e.strip() for e in emotions_str.split(',')]
                all_emotions.extend(emotions)
        
        if not all_emotions:
            return "No specific emotions detected in your dream records."
        
        from collections import Counter
        emotion_counts = Counter(all_emotions)
        common_emotions = emotion_counts.most_common(3)
        
        recent_emotions = []
        for emotions_str in dream_history['emotions'].tail(3):
            if isinstance(emotions_str, str) and emotions_str != "neutral":
                recent_emotions.extend([e.strip() for e in emotions_str.split(',')])
        
        recent_emotion_counts = Counter(recent_emotions)
        recent_common = recent_emotion_counts.most_common(2) if recent_emotions else []
        
        analysis = f"Your most common dream emotions overall are: {', '.join([e[0] for e in common_emotions])}. "
        
        if recent_common:
            analysis += f"Recently, you've been experiencing more {recent_common[0][0]} in your dreams."
        
        if len(dream_history) >= 5:
            earlier_emotions = []
            for emotions_str in dream_history['emotions'].head(len(dream_history) - 3):
                if isinstance(emotions_str, str) and emotions_str != "neutral":
                    earlier_emotions.extend([e.strip() for e in emotions_str.split(',')])
            
            earlier_emotion_counts = Counter(earlier_emotions)
            
            if earlier_emotions and recent_emotions and earlier_emotion_counts and recent_emotion_counts:
                try:
                    earlier_most_common = earlier_emotion_counts.most_common(1)[0][0]
                    recent_most_common = recent_emotion_counts.most_common(1)[0][0]
                    
                    if earlier_most_common != recent_most_common:
                        analysis += f" There appears to be a shift in your emotional patterns from {earlier_most_common} to {recent_most_common}."
                except IndexError:
                    pass  # Not enough data for this analysis
        
        return analysis
    except Exception as e:
        return f"Error analyzing emotion patterns: {str(e)}"

def get_emotion_recommendations(emotion_scores, personality):
    """Generate personalized recommendations based on emotions and personality."""
    if not emotion_scores or not personality:
        return "Continue tracking your dreams to receive personalized recommendations."
    
    try:
        recommendations = []
        
        dominant_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        
        for emotion, score in dominant_emotions:
            if score > 0:
                if emotion == 'fear':
                    recommendations.append("Consider journaling about your fears to better understand and process them.")
                    if personality.get('stress', 5) > 7:
                        recommendations.append("Your high stress levels might be contributing to fear-based dreams. Try relaxation techniques before bed.")
                
                elif emotion == 'joy':
                    recommendations.append("Your positive dream emotions suggest good emotional processing. Continue your current practices.")
                
                elif emotion == 'sadness':
                    recommendations.append("Consider exploring what might be causing feelings of sadness in your waking life.")
                    if personality.get('analytical', 5) > 7:
                        recommendations.append("Try analyzing patterns between daily events and your emotional dreams.")
                
                elif emotion == 'anger':
                    recommendations.append("Look for healthy ways to express and process anger in your waking life.")
                    if personality.get('creativity', 5) > 7:
                        recommendations.append("Channel any angry emotions into creative expression.")
                
                elif emotion == 'confusion':
                    recommendations.append("Your dreams suggest you might be processing complex situations. Take time to reflect on current life changes.")
                
                elif emotion == 'peace':
                    recommendations.append("Your calm dreams suggest good emotional balance. Maintain your current stress management practices.")
        
        if not recommendations:
            recommendations.append("Continue tracking your dream emotions to reveal more patterns over time.")
        
        return "\n".join(recommendations)
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"