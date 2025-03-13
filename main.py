import nltk
nltk.download('vader_lexicon')
import pandas as pd
import matplotlib.pyplot as plt
from nlp_utils import extract_keywords, analyze_sentiment, extract_dream_themes
from gpt_predictor import predict_future_impact, analyze_dream_patterns
from personality import get_personality_data, get_personality_profile, get_dream_processing_style
from visualization import generate_wordcloud, plot_sentiment_over_time, plot_emotion_distribution
from emotion_detection import detect_emotions, analyze_emotion_patterns, get_emotion_recommendations
from dream_symbols import analyze_dream_symbols, generate_symbol_insights

def main():
    print("=== Future Dream Influence Predictor ===\n")

    dream_text = input("Enter your dream description: ")

    themes = extract_keywords(dream_text)
    sentiment_scores = analyze_sentiment(dream_text)
    compound_sentiment = sentiment_scores.get('compound', 0)
    dream_themes = extract_dream_themes(dream_text)

    emotions = detect_emotions(dream_text)
    primary_emotions = emotions['primary_emotions']
    emotion_scores = emotions['emotion_scores']

    symbol_analysis = analyze_dream_symbols(dream_text)
    symbols_found = symbol_analysis['symbols_found']

    print("\n=== Dream Analysis ===")
    print("\nExtracted Dream Themes:", themes)
    print("Higher-Level Dream Themes:", dream_themes)
    print("\nDream Sentiment:")
    print(f"  Compound Score: {compound_sentiment:.2f}")
    print(f"  Positive: {sentiment_scores.get('pos', 0):.2f}")
    print(f"  Neutral: {sentiment_scores.get('neu', 0):.2f}")
    print(f"  Negative: {sentiment_scores.get('neg', 0):.2f}")
    
    print("\nPrimary Emotions Detected:", ", ".join(primary_emotions) if primary_emotions else "Neutral")
    
    if symbols_found:
        print("\nSignificant Dream Symbols:", ", ".join(symbols_found[:5]))
        print("\nSymbol Interpretation:")
        print(symbol_analysis['interpretation'])

    print("\nGenerating word cloud of dream themes...")
    fig = generate_wordcloud(themes)
    plt.show()

    print("\n=== Personality Assessment ===\n")
    personality = get_personality_data()
    
    personality_profile = get_personality_profile(personality)
    print("\n" + personality_profile)
    
    processing_style = get_dream_processing_style(personality)
    print("\n" + processing_style)

    print("\n=== Future Influence Prediction ===\n")
    prediction = predict_future_impact(dream_themes, compound_sentiment, personality, 
                                     dream_text=dream_text, emotions=emotion_scores, 
                                     symbols=symbols_found)
    print(prediction)

    print("\n=== Personalized Recommendations ===\n")
    recommendations = get_emotion_recommendations(emotion_scores, personality)
    print(recommendations)
    
    print("\n=== Symbol-Based Recommendations ===\n")
    print(symbol_analysis['recommendations'])

    log_entry = {"date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "dream": dream_text,
                 "themes": ", ".join(themes),
                 "sentiment": compound_sentiment,
                 "emotions": emotions['emotions_str'],
                 "symbols": ", ".join(symbols_found) if symbols_found else ""}
    try:
        dream_log = pd.read_csv("dream_log.csv")
        dream_log = pd.concat([dream_log, pd.DataFrame([log_entry])], ignore_index=True)
    except FileNotFoundError:
        dream_log = pd.DataFrame([log_entry])
    dream_log.to_csv("dream_log.csv", index=False)

    if len(dream_log) > 1:
        print("\n=== Dream History Analysis ===\n")
        print("Plotting sentiment over time...")
        fig = plot_sentiment_over_time(dream_log)
        plt.show()
        
        if len(dream_log) >= 3:
            pattern_analysis = analyze_dream_patterns(dream_log, personality)
            print("\nDream Pattern Analysis:")
            print(pattern_analysis)
            
            if 'emotions' in dream_log.columns:
                emotion_analysis = analyze_emotion_patterns(dream_log)
                print("\nEmotion Pattern Analysis:")
                print(emotion_analysis)
                
                fig = plot_emotion_distribution(dream_log)
                if fig:
                    plt.show()

if __name__ == "__main__":
    main()