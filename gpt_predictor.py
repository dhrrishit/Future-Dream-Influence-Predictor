import google.generativeai as genai
import os
from emotion_detection import detect_emotions
from dream_symbols import analyze_dream_symbols, generate_symbol_insights

GOOGLE_API_KEY = "YOUR_API_KEY_HERE"  # yate jba for api https://makersuite.google.com/app/apikey

genai.configure(api_key=GOOGLE_API_KEY)

MODEL_NAME = "gemini-pro" 
MODEL = genai.GenerativeModel(MODEL_NAME)

def predict_future_impact(dream_themes, sentiment, personality, dream_text=None, emotions=None, symbols=None):
    if dream_text and not emotions:
        emotion_data = detect_emotions(dream_text)
        emotions = emotion_data['emotion_scores']
    
    if dream_text and not symbols:
        symbol_data = analyze_dream_symbols(dream_text, personality)
        symbols = symbol_data['symbols_found']
# basically pre-prompt diya ase yate
    prompt_content = f"""
You are an expert dream analyst with deep knowledge of psychology, symbolism, and predictive analysis.

A user has provided the following dream information:
- Dream themes: {', '.join(dream_themes)}
- Overall sentiment score: {sentiment:.2f} (ranges from -1 to 1, where negative is negative emotion, positive is positive emotion)

The user's personality profile:
- Intuition-driven decision-making: {personality.get('intuition', 'N/A')}/10
- Stress level: {personality.get('stress', 'N/A')}/10
- Creativity: {personality.get('creativity', 'N/A')}/10
- Analytical thinking: {personality.get('analytical', 'N/A')}/10
- Emotional sensitivity: {personality.get('emotional', 'N/A')}/10
- Preference for routine: {personality.get('routine', 'N/A')}/10
"""

    if emotions:
        top_emotions = sorted([(e, s) for e, s in emotions.items() if s > 0], key=lambda x: x[1], reverse=True)[:3]
        if top_emotions:
            prompt_content += "\nDominant emotions detected in the dream:\n"
            for emotion, score in top_emotions:
                prompt_content += f"- {emotion.capitalize()}: {score:.2f}\n"
    
    if symbols and len(symbols) > 0:
        prompt_content += f"\nSignificant symbols identified: {', '.join(symbols[:5])}\n"
    
    prompt_content += """

Based on these details, provide a creative and thoughtful prediction of:
1. How these dream elements might influence the user's future decisions
2. What subconscious patterns might be emerging
3. What potential opportunities or challenges the dream might be highlighting
4. One specific actionable insight the user could apply to their waking life

Make your response insightful, personalized, and psychologically sound while avoiding generic interpretations.
    """
    
    try:
        response = MODEL.generate_content(prompt_content)
        prediction = response.text.strip()
        return prediction
    
    except Exception as e:
        return f"Error during prediction: {e}"

def analyze_dream_patterns(dream_history, personality):
    if len(dream_history) < 3:
        return "Need at least 3 dreams to analyze patterns effectively."
    
    symbol_insights = generate_symbol_insights(dream_history, personality)
    
    prompt_content = f"""
You are an expert in dream pattern analysis and psychological insight.

A user has provided their dream history with {len(dream_history)} recorded dreams.

Key insights from symbol analysis:
{symbol_insights}

The user's personality profile:
- Intuition-driven decision-making: {personality.get('intuition', 'N/A')}/10
- Stress level: {personality.get('stress', 'N/A')}/10
- Creativity: {personality.get('creativity', 'N/A')}/10
- Analytical thinking: {personality.get('analytical', 'N/A')}/10

Based on this information, provide an analysis of:
1. Potential recurring patterns in the user's dreams
2. How these patterns might relate to their waking life
3. What psychological processes might be occurring
4. How their personality traits might be influencing their dream patterns

Make your analysis insightful, personalized, and psychologically sound.
    """
    
    try:
        response = MODEL.generate_content(prompt_content)
        analysis = response.text.strip()
        return analysis
    
    except Exception as e:
        return f"Error during pattern analysis: {e}"