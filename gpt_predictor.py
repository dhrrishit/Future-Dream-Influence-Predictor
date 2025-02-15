import google.generativeai as genai

GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
genai.configure(api_key=GOOGLE_API_KEY)

MODEL_NAME = "gemini-pro" 
MODEL = genai.GenerativeModel(MODEL_NAME)


def predict_future_impact(dream_themes, sentiment, personality):
    prompt_content = f"""
You are an expert dream analyst. A user has provided their dream themes: {', '.join(dream_themes)}.
The overall dream sentiment score is {sentiment}.
The user's personality data is:
- Intuition-driven decision-making score: {personality.get('intuition', 'N/A')}
- Stress level score: {personality.get('stress', 'N/A')}

Based on these details, provide a creative and thoughtful prediction of how these dream elements might influence the user's future decisions.
    """
    try:
        response = MODEL.generate_content(prompt_content)
        prediction = response.text.strip()
        return prediction
    
    except Exception as e:
        return f"Error during prediction: {e}"