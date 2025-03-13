import pandas as pd
import numpy as np
import re
from collections import Counter

DREAM_SYMBOLS = {
    'water': {
        'meaning': 'Emotions, unconscious mind, purification',
        'category': 'nature',
        'associations': ['emotions', 'cleansing', 'life']
    },
    'fire': {
        'meaning': 'Transformation, passion, destruction or rebirth',
        'category': 'nature',
        'associations': ['change', 'energy', 'danger']
    },
    'mountain': {
        'meaning': 'Obstacle, achievement, perspective',
        'category': 'nature',
        'associations': ['challenge', 'journey', 'accomplishment']
    },
    'tree': {
        'meaning': 'Growth, connection, life path',
        'category': 'nature',
        'associations': ['stability', 'roots', 'family']
    },
    'sky': {
        'meaning': 'Possibilities, spirituality, higher consciousness',
        'category': 'nature',
        'associations': ['freedom', 'aspiration', 'perspective']
    },
    'dog': {
        'meaning': 'Loyalty, friendship, protection',
        'category': 'animals',
        'associations': ['companionship', 'instinct', 'devotion']
    },
    'cat': {
        'meaning': 'Independence, mystery, feminine energy',
        'category': 'animals',
        'associations': ['intuition', 'grace', 'detachment']
    },
    'bird': {
        'meaning': 'Freedom, perspective, spirituality',
        'category': 'animals',
        'associations': ['transcendence', 'messages', 'aspiration']
    },
    'snake': {
        'meaning': 'Transformation, healing, hidden fears',
        'category': 'animals',
        'associations': ['rebirth', 'wisdom', 'deception']
    },
    'spider': {
        'meaning': 'Creativity, patience, feminine energy',
        'category': 'animals',
        'associations': ['weaving', 'entrapment', 'complexity']
    },
    'house': {
        'meaning': 'Self, mind, personal space',
        'category': 'objects',
        'associations': ['security', 'identity', 'shelter']
    },
    'door': {
        'meaning': 'Opportunity, transition, access',
        'category': 'objects',
        'associations': ['choices', 'threshold', 'beginning']
    },
    'key': {
        'meaning': 'Access, solution, knowledge',
        'category': 'objects',
        'associations': ['discovery', 'power', 'unlocking']
    },
    'mirror': {
        'meaning': 'Self-reflection, truth, identity',
        'category': 'objects',
        'associations': ['perception', 'reality', 'appearance']
    },
    'book': {
        'meaning': 'Knowledge, memory, life story',
        'category': 'objects',
        'associations': ['learning', 'wisdom', 'record']
    },
    'child': {
        'meaning': 'Innocence, potential, vulnerability',
        'category': 'people',
        'associations': ['new beginnings', 'playfulness', 'dependency']
    },
    'stranger': {
        'meaning': 'Unknown aspects of self, new possibilities',
        'category': 'people',
        'associations': ['mystery', 'unfamiliarity', 'projection']
    },
    'teacher': {
        'meaning': 'Guidance, wisdom, authority',
        'category': 'people',
        'associations': ['learning', 'respect', 'knowledge']
    },
    'falling': {
        'meaning': 'Insecurity, failure, letting go',
        'category': 'actions',
        'associations': ['fear', 'loss of control', 'surrender']
    },
    'flying': {
        'meaning': 'Freedom, perspective, transcendence',
        'category': 'actions',
        'associations': ['liberation', 'ambition', 'escape']
    },
    'running': {
        'meaning': 'Escape, avoidance, pursuit of goals',
        'category': 'actions',
        'associations': ['fear', 'determination', 'urgency']
    },
    'searching': {
        'meaning': 'Seeking answers, purpose, identity',
        'category': 'actions',
        'associations': ['quest', 'uncertainty', 'desire']
    },
    'school': {
        'meaning': 'Learning, social anxiety, evaluation',
        'category': 'settings',
        'associations': ['testing', 'development', 'preparation']
    },
    'forest': {
        'meaning': 'Unconscious mind, unknown, mystery',
        'category': 'settings',
        'associations': ['exploration', 'danger', 'growth']
    },
    'ocean': {
        'meaning': 'Emotions, unconscious, vastness',
        'category': 'settings',
        'associations': ['depth', 'mystery', 'power']
    },
    'bridge': {
        'meaning': 'Transition, connection, crossing over',
        'category': 'settings',
        'associations': ['journey', 'decision', 'progress']
    }
}

def identify_symbols(dream_text):
    text_lower = dream_text.lower()
    found_symbols = {}
    for symbol, data in DREAM_SYMBOLS.items():
        pattern = r'\b' + re.escape(symbol) + r'\b'
        matches = re.findall(pattern, text_lower)
        if matches:
            found_symbols[symbol] = {
                'count': len(matches),
                'meaning': data['meaning'],
                'category': data['category'],
                'associations': data['associations']
            }
    return found_symbols

def get_symbol_frequencies(dream_history):
    if 'dream' not in dream_history.columns or len(dream_history) == 0:
        return {}
    
    all_symbols = {}
    
    for dream_text in dream_history['dream']:
        if isinstance(dream_text, str):
            symbols = identify_symbols(dream_text)
            for symbol, data in symbols.items():
                if symbol in all_symbols:
                    all_symbols[symbol] += data['count']
                else:
                    all_symbols[symbol] = data['count']
    
    return all_symbols

def analyze_dream_symbols(dream_text, personality=None):
    symbols = identify_symbols(dream_text)
    
    if not symbols:
        return {
            'symbols_found': [],
            'interpretation': "No common dream symbols were identified in your dream.",
            'recommendations': "Consider adding more details to your dream description for better analysis."
        }
    
    symbols_found = list(symbols.keys())
    categories = {}
    interpretations = []
    recommendations = []
    
    for symbol, data in symbols.items():
        category = data['category']
        categories[category] = categories.get(category, 0) + 1
        interpretations.append(f"{symbol.title()}: {data['meaning']}")
    
    dominant_category = max(categories.items(), key=lambda x: x[1])[0]
    
    if personality:
        if personality.get('intuition', 5) > 7:
            recommendations.append("Pay special attention to the symbolic meanings as your high intuition suggests a strong connection to symbolic understanding.")
        if personality.get('analytical', 5) > 7:
            recommendations.append("Consider how these symbols might relate to specific situations in your waking life.")
    
    if dominant_category == 'nature':
        recommendations.append("The natural symbols suggest a connection to fundamental life forces and emotions.")
    elif dominant_category == 'animals':
        recommendations.append("The animal symbols indicate strong instinctual or primal energies at work.")
    elif dominant_category == 'objects':
        recommendations.append("The presence of significant objects suggests practical matters or tools for change in your life.")
    elif dominant_category == 'people':
        recommendations.append("The human symbols indicate important relationships or aspects of yourself.")
    elif dominant_category == 'actions':
        recommendations.append("The action symbols suggest important processes or changes occurring in your life.")
    elif dominant_category == 'settings':
        recommendations.append("The settings in your dream point to different aspects of your life situation or emotional state.")
    
    return {
        'symbols_found': symbols_found,
        'interpretation': "\n".join(interpretations),
        'recommendations': "\n".join(recommendations)
    }

def generate_symbol_insights(dream_history, personality=None):
    symbol_frequencies = get_symbol_frequencies(dream_history)
    
    if not symbol_frequencies:
        return "No recurring symbols found in your dream history."
    
    top_symbols = sorted(symbol_frequencies.items(), key=lambda x: x[1], reverse=True)[:5]
    
    insights = ["Your most common dream symbols:"]
    for symbol, count in top_symbols:
        if symbol in DREAM_SYMBOLS:
            insights.append(f"{symbol.title()} (appears {count} times): {DREAM_SYMBOLS[symbol]['meaning']}")
    
    categories = {}
    for symbol, _ in top_symbols:
        if symbol in DREAM_SYMBOLS:
            category = DREAM_SYMBOLS[symbol]['category']
            categories[category] = categories.get(category, 0) + 1
    
    if categories:
        dominant_category = max(categories.items(), key=lambda x: x[1])[0]
        insights.append(f"\nYour dreams show a strong focus on {dominant_category}-related symbols, which might indicate:")
        
        if dominant_category == 'nature':
            insights.append("- A deep connection to natural forces and fundamental life energies")
            insights.append("- Processing of raw emotions and primal experiences")
        elif dominant_category == 'animals':
            insights.append("- Strong instinctual drives and natural wisdom")
            insights.append("- Connection to specific animal traits or energies")
        elif dominant_category == 'objects':
            insights.append("- Focus on practical tools and solutions")
            insights.append("- Attention to material aspects of life")
        elif dominant_category == 'people':
            insights.append("- Emphasis on relationships and social connections")
            insights.append("- Processing of various aspects of self")
        elif dominant_category == 'actions':
            insights.append("- Focus on movement and change in your life")
            insights.append("- Processing of dynamic life situations")
        elif dominant_category == 'settings':
            insights.append("- Attention to context and environment")
            insights.append("- Processing of different life situations")
    
    return "\n".join(insights)